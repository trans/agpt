module MicroGPT
  module AGPT
    # Memory-efficient BFS trie-walk trainer.
    #
    # Forward (BFS depth 0 → max):
    #   Walk the trie level by level. At each depth, every node extends its
    #   parent's KV cache by one token. Stores only each node's K/V contribution
    #   (~1 KB) plus loss info. KV caches are ephemeral — freed per depth level.
    #
    # Backward (BFS depth max → 0):
    #   For each node: reconstruct the full KV cache from stored K/V rows by
    #   walking the parent chain, re-run one forward step to regenerate
    #   BlockStepState, then backward. Gradient accumulators (dK/dV from
    #   descendants) persist across depth levels.
    #
    # Memory: O(total_nodes × 1 KB) for K/V store + O(total_nodes × 512 B)
    # for grad accumulators. No full NodeForwardState or KV caches retained.
    class TrieWalkTrainer
      getter corpus : TrieCorpus
      getter loss_fn : WeightedNextTokenLoss
      getter observed_count : Int32
      property debug_verify : Bool = false
      property entropy_lambda : Float64 = 0.0  # structure-aware loss weighting

      def initialize(@corpus : TrieCorpus, @loss_fn = WeightedNextTokenLoss.new)
        @observed_count = 0
        @corpus.each_observed_node do |node|
          @observed_count += 1 unless node.depth == 0
        end
      end

      # Depth-progressive subtrie training with local-depth backward.
      #
      # At each stage d (depth 1→max):
      #   1. Forward depth d (batched matmuls), storing K/V in kv_store
      #   2. Partition nodes at depth d into subtries (by root-level ancestor)
      #   3. For each subtrie: backward depth d only, normalize, update
      #
      # This gives (D × branching_factor) updates per epoch — comparable to
      # window training's update frequency while preserving trie prefix sharing.
      #
      # Returns {mean_loss, nodes_trained}.
      def train_epoch(model : MiniGPT) : {Float64, Int32}
        epoch_started = Time.instant if MicroGPT::PerfTrace.enabled?
        seq_len = model.config.seq_len
        head_dims = model.blocks.first.attn.head_dims
        n_layers = model.config.n_layers

        total_loss = 0.0
        nodes_trained = 0

        # Compact per-node K/V storage (~1 KB/node) — persists across entire epoch
        kv_store = NodeKVStore.new

        # Per-node metadata
        node_ancestor_ids = {} of Int32 => Array(Int32)
        node_positions = {} of Int32 => Int32
        node_root_child = {} of Int32 => Int32  # maps node_id → root child id (for partitioning)
        node_ancestor_ids[@corpus.root.id] = [] of Int32

        prev_caches : Hash(Int32, Array(AGPT::LayerKVCache))? = nil

        @corpus.each_depth_level do |depth, nodes|
          next if depth == 0

          depth_started = Time.instant if MicroGPT::PerfTrace.enabled?
          eligible = Array(TrieNode).new
          nodes.each do |node|
            parent = node.parent.not_nil!
            next unless node_ancestor_ids.has_key?(parent.id)
            next if parent.depth >= seq_len
            eligible << node
          end
          next if eligible.empty?

          eligible.each do |node|
            node_positions[node.id] = depth - 1
            # Track which root child each node descends from
            if depth == 1
              node_root_child[node.id] = node.id
            else
              node_root_child[node.id] = node_root_child[node.parent.not_nil!.id]
            end
          end

          # Batched forward for ALL nodes at this depth — shared projections
          forward_started = Time.instant if MicroGPT::PerfTrace.enabled?
          results, this_caches = BatchedDepthForward.forward_depth(
            eligible, node_ancestor_ids, node_positions, kv_store, model, @corpus, prev_caches
          )
          prev_caches = this_caches
          MicroGPT::PerfTrace.observe_max("agpt.forward_stage_bytes", Mat.allocated_bytes)
          MicroGPT::PerfTrace.add_time("agpt.epoch.forward", Time.instant - forward_started.not_nil!) if forward_started

          # Compute loss: batch softmax across all nodes, download once, then
          # compute per-node weighted CE loss and gradient on CPU.
          loss_started = Time.instant if MicroGPT::PerfTrace.enabled?
          loss_grads = {} of Int32 => Mat
          result_map = {} of Int32 => BatchedDepthForward::NodeResult
          MicroGPT::PerfTrace.with_scope("agpt.loss") do
            n_results = results.size
            vocab_size = model.config.vocab_size

            # Stack all logits into [N, vocab] and batch softmax (one GPU op)
            logits_batched = Mat.new(n_results, vocab_size)
            n_results.times do |i|
              vocab_size.times { |j| logits_batched[i, j] = results[i].logits[0, j] }
            end
            probs_batched = MicroGPT.backend.softmax_rows(logits_batched)

            # Single bulk download of all probs to CPU
            all_probs = probs_batched.data  # one sync for all N×vocab

            # Per-node loss and gradient from downloaded probs.
            # Optional entropy weighting: w = 1 + lambda * H_norm, where H_norm
            # is the node's empirical entropy normalized by log(vocab_size).
            # Branching nodes get higher weight; unary/deterministic nodes get w=1.
            log_vocab = Math.log(vocab_size.to_f64)
            lambda = @entropy_lambda
            results.each_with_index do |result, i|
              result_map[result.node_id] = result
              node = @corpus.node_for_id(result.node_id)
              unless node.next_token_counts.empty?
                counts = node.next_token_counts_hash
                total = counts.values.sum(0)
                total_f = total.to_f64

                # Compute empirical entropy H(p) from counts
                entropy = 0.0
                if lambda > 0.0 && counts.size > 1
                  counts.each do |_tok, count|
                    q = count / total_f
                    entropy -= q * Math.log(q) if q > 0.0
                  end
                end
                weight = (lambda > 0.0) ? 1.0 + lambda * (entropy / log_vocab) : 1.0

                # Loss from CPU probs
                loss_value = 0.0
                prob_offset = i * vocab_size
                counts.each do |token_id, count|
                  loss_value -= count * Math.log(all_probs[prob_offset + token_id] + 1e-10)
                end
                loss_value /= total
                loss_value *= weight

                # Gradient: probs - one-hot(weighted), scaled by weight
                grad = Mat.new(1, vocab_size)
                weight_f32 = weight.to_f32
                vocab_size.times { |j| grad[0, j] = all_probs[prob_offset + j] * weight_f32 }
                counts.each do |token_id, count|
                  grad[0, token_id] -= (count.to_f32 / total) * weight_f32
                end

                loss_grads[result.node_id] = grad
                total_loss += loss_value
                nodes_trained += 1
              end
            end
          end
          MicroGPT::PerfTrace.add_time("agpt.epoch.loss", Time.instant - loss_started.not_nil!) if loss_started

          # Partition into subtries by root child
          partition_started = Time.instant if MicroGPT::PerfTrace.enabled?
          subtries = {} of Int32 => Array(BatchedDepthForward::NodeResult)
          eligible.each do |node|
            root_id = node_root_child[node.id]
            (subtries[root_id] ||= [] of BatchedDepthForward::NodeResult) << result_map[node.id]
          end
          if partition_started
            MicroGPT::PerfTrace.add_time("agpt.epoch.partition", Time.instant - partition_started.not_nil!)
            MicroGPT::PerfTrace.increment("agpt.epoch.subtries", subtries.size.to_i64)
          end

          # Process each subtrie: backward uses forward results directly
          # (no re-forward needed — local-depth backward is at the same depth
          # we just forwarded, so BlockStepState is already in results)
          subtries.each do |_root_id, subtrie_results|
            backward_started = Time.instant if MicroGPT::PerfTrace.enabled?
            MicroGPT::PerfTrace.with_scope("agpt.zero_gradients") do
              zero_gradients(model)
            end
            grad_accums = {} of Int32 => NodeGradAccum

            subtrie_grads = nil.as(Array(Mat)?)
            MicroGPT::PerfTrace.with_scope("agpt.subtrie_loss_grads") do
              subtrie_grads = subtrie_results.map do |result|
                if d_logits = loss_grads.delete(result.node_id)
                  # Reuse the per-node logits gradient computed during the
                  # loss pass instead of recomputing weighted loss a second time.
                  d_logits
                else
                  Mat.new(1, model.config.vocab_size)
                end
              end
            end
            subtrie_grads = subtrie_grads.not_nil!

            BatchedDepthBackward.backward_depth(
              subtrie_results, subtrie_grads, grad_accums, kv_store, model, @corpus, this_caches
            )
            MicroGPT::PerfTrace.observe_max("agpt.backward_stage_bytes", Mat.allocated_bytes)
            MicroGPT::PerfTrace.add_time("agpt.epoch.backward", Time.instant - backward_started.not_nil!) if backward_started

            # Normalize by subtrie size and update
            if subtrie_results.size > 0
              update_started = Time.instant if MicroGPT::PerfTrace.enabled?
              MicroGPT::PerfTrace.with_scope("agpt.update") do
                scale_gradients(model, 1.0 / subtrie_results.size)
                lr = model.config.learning_rate
                model.embedding.update(lr)
                model.blocks.each &.update(lr)
                model.final_norm.update(lr)
                model.output.update(lr)
              end
              MicroGPT::PerfTrace.observe_max("agpt.update_stage_bytes", Mat.allocated_bytes)
              MicroGPT::PerfTrace.add_time("agpt.epoch.update", Time.instant - update_started.not_nil!) if update_started
            end
          end

          MicroGPT::PerfTrace.add_time("agpt.epoch.depth_total", Time.instant - depth_started.not_nil!) if depth_started
        end

        mean_loss = nodes_trained > 0 ? total_loss / nodes_trained : 0.0
        MicroGPT::PerfTrace.add_time("agpt.epoch.total", Time.instant - epoch_started.not_nil!) if epoch_started
        {mean_loss, nodes_trained}
      end

      # Scatter ancestor dK/dV from one node's backward to its ancestors' accumulators.
      private def scatter_ancestor_grads(
        ancestor_grads : IncrementalBackward::AncestorGrads,
        state : NodeForwardState,
        n_layers : Int32,
        head_dims : Array(Int32),
        grad_accums : Hash(Int32, NodeGradAccum)
      )
        # ancestor_grads[layer][head] = {dk_ancestors, dv_ancestors}
        # dk_ancestors has rows for positions 0..prefix_len-2
        # state.ancestor_ids maps position index to trie node id
        prefix_len = state.position + 1
        return if prefix_len <= 1  # no ancestors

        n_layers.times do |li|
          head_dims.size.times do |hi|
            dk_anc, dv_anc = ancestor_grads[li][hi]
            next if dk_anc.rows == 0

            hd = head_dims[hi]
            dk_anc.rows.times do |pos|
              # Position pos in the prefix corresponds to ancestor_ids[pos]
              ancestor_id = state.ancestor_ids[pos]
              acc = grad_accums[ancestor_id]? || begin
                a = NodeGradAccum.new(n_layers, head_dims)
                grad_accums[ancestor_id] = a
                a
              end

              # Add this row to the ancestor's accumulator
              row_dk = Mat.new(1, hd)
              row_dv = Mat.new(1, hd)
              hd.times do |j|
                row_dk[0, j] = dk_anc[pos, j]
                row_dv[0, j] = dv_anc[pos, j]
              end
              acc.add_dk(li, hi, row_dk)
              acc.add_dv(li, hi, row_dv)
            end
          end
        end
      end

      # Numerical gradient check: perturb a specific weight and measure loss change
      private def numerical_grad_check(
        model : MiniGPT,
        node_losses : Hash(Int32, {Hash(Int32, Int32)})
      )
        eps = 1e-3_f32

        # Pick a specific weight to check (wq.dw[5, 1])
        weight_mat = model.blocks[0].attn.wq.w
        grad_mat = model.blocks[0].attn.wq.dw
        row, col = 5, 1
        label = "wq.w[#{row},#{col}]"

        analytical_grad = grad_mat[row, col]

        # Total loss function (sum over all observed nodes)
        total_loss = ->{
          loss = 0.0_f64
          node_losses.each do |node_id, loss_info|
            counts = loss_info[0]
            node = find_node(node_id)
            next unless node
            prefix = @corpus.prefix_for(node)
            seq_len = model.config.seq_len
            truncated = prefix.size > seq_len ? prefix[-seq_len..] : prefix
            logits = model.forward(truncated)
            last_row = logits.rows - 1
            last_logits = Mat.new(1, logits.cols)
            logits.cols.times { |c| last_logits[0, c] = logits[last_row, c] }
            l, _ = @loss_fn.loss_and_backward(last_logits, counts)
            loss += l
          end
          loss
        }

        original = weight_mat[row, col]

        weight_mat[row, col] = original + eps
        loss_plus = total_loss.call

        weight_mat[row, col] = original - eps
        loss_minus = total_loss.call

        weight_mat[row, col] = original

        numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
        STDERR.puts "[grad check] #{label} analytical=#{"%.6f" % analytical_grad} numerical=#{"%.6f" % numerical_grad} diff=#{"%.6f" % (analytical_grad - numerical_grad).abs}"

        # Also check emb[47, 10]
        emb_mat = model.embedding.token_emb
        emb_grad = model.embedding.d_token_emb
        row2, col2 = 47, 10
        analytical_grad_e = emb_grad[row2, col2]
        original_e = emb_mat[row2, col2]

        emb_mat[row2, col2] = original_e + eps
        loss_plus = total_loss.call
        emb_mat[row2, col2] = original_e - eps
        loss_minus = total_loss.call
        emb_mat[row2, col2] = original_e

        numerical_grad_e = (loss_plus - loss_minus) / (2.0 * eps)
        STDERR.puts "[grad check] emb[#{row2},#{col2}] analytical=#{"%.6f" % analytical_grad_e} numerical=#{"%.6f" % numerical_grad_e} diff=#{"%.6f" % (analytical_grad_e - numerical_grad_e).abs}"
      end

      private def find_node(id : Int32) : TrieNode?
        result = nil
        @corpus.each_observed_node do |node|
          if node.id == id
            result = node
            break
          end
        end
        result
      end

      private def scale_gradients(model : MiniGPT, scale : Float64)
        trace_sync_delta("agpt.epoch.scale_gradients") do
        s = scale.to_f32
        model.embedding.d_token_emb.scale!(s)
        model.blocks.each do |block|
          block.attn.wq.dw.scale!(s); block.attn.wq.db.scale!(s)
          block.attn.wk.dw.scale!(s); block.attn.wk.db.scale!(s)
          block.attn.wv.dw.scale!(s); block.attn.wv.db.scale!(s)
          block.attn.wo.dw.scale!(s); block.attn.wo.db.scale!(s)
          block.ff.l1.dw.scale!(s); block.ff.l1.db.scale!(s)
          block.ff.l2.dw.scale!(s); block.ff.l2.db.scale!(s)
          block.ln1.dgamma.scale!(s); block.ln1.dbeta.scale!(s)
          block.ln2.dgamma.scale!(s); block.ln2.dbeta.scale!(s)
        end
        model.final_norm.dgamma.scale!(s); model.final_norm.dbeta.scale!(s)
        model.output.proj.dw.scale!(s); model.output.proj.db.scale!(s)
        end
      end

      private def zero_gradients(model : MiniGPT)
        model.embedding.d_token_emb.zero!
        model.blocks.each do |block|
          block.attn.wq.dw.zero!; block.attn.wq.db.zero!
          block.attn.wk.dw.zero!; block.attn.wk.db.zero!
          block.attn.wv.dw.zero!; block.attn.wv.db.zero!
          block.attn.wo.dw.zero!; block.attn.wo.db.zero!
          block.ff.l1.dw.zero!; block.ff.l1.db.zero!
          block.ff.l2.dw.zero!; block.ff.l2.db.zero!
          block.ln1.dgamma.zero!; block.ln1.dbeta.zero!
          block.ln2.dgamma.zero!; block.ln2.dbeta.zero!
        end
        model.final_norm.dgamma.zero!; model.final_norm.dbeta.zero!
        model.output.proj.dw.zero!; model.output.proj.db.zero!
      end

      private def trace_sync_delta(section : String, &)
        unless MicroGPT::PerfTrace.enabled?
          yield
          return
        end

        before_calls = MicroGPT::PerfTrace.count("sync_to_cpu.calls")
        before_bytes = MicroGPT::PerfTrace.bytes("sync_to_cpu.calls")
        before_ms = MicroGPT::PerfTrace.millis("sync_to_cpu")
        yield
        call_delta = MicroGPT::PerfTrace.count("sync_to_cpu.calls") - before_calls
        byte_delta = MicroGPT::PerfTrace.bytes("sync_to_cpu.calls") - before_bytes
        ms_delta = MicroGPT::PerfTrace.millis("sync_to_cpu") - before_ms
        MicroGPT::PerfTrace.increment("#{section}.sync", call_delta)
        MicroGPT::PerfTrace.add_bytes("#{section}.sync", byte_delta)
        MicroGPT::PerfTrace.add_millis("#{section}.sync_to_cpu", ms_delta)
      end

    end
  end
end
