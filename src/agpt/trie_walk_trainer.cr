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

      # Two-regime training: shallow depths (1..d_branch) use the existing
      # forward_depth path with sibling-grouped attention. Deep depths
      # (d_branch+1..max) are handled by one packed forward_segments call over
      # all chain tails. Update cadence is preserved: (depth, root_child)
      # subtries drive .update() calls in both regimes.
      def train_epoch_two_regime(model : MiniGPT, d_branch : Int32? = nil) : {Float64, Int32}
        effective_d_branch = d_branch || @corpus.pick_d_branch
        seq_len = model.config.seq_len
        head_dims = model.blocks.first.attn.head_dims
        n_layers = model.config.n_layers

        total_loss = 0.0
        nodes_trained = 0

        kv_store = NodeKVStore.new
        node_ancestor_ids = {} of Int32 => Array(Int32)
        node_positions = {} of Int32 => Int32
        node_root_child = {} of Int32 => Int32
        node_ancestor_ids[@corpus.root.id] = [] of Int32
        prev_caches : Hash(Int32, Array(AGPT::LayerKVCache))? = nil

        # --- Shallow regime: reuse existing depth-major loop semantics ---
        @corpus.each_depth_level do |depth, nodes|
          next if depth == 0
          break if depth > effective_d_branch

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
            if depth == 1
              node_root_child[node.id] = node.id
            else
              node_root_child[node.id] = node_root_child[node.parent.not_nil!.id]
            end
          end

          results, this_caches = BatchedDepthForward.forward_depth(
            eligible, node_ancestor_ids, node_positions, kv_store, model, @corpus, prev_caches
          )
          prev_caches = this_caches

          loss_grads = compute_loss_grads(results, model, pointerof(total_loss), pointerof(nodes_trained))
          run_subtrie_backward_updates(eligible, results, loss_grads, node_root_child, kv_store, model, this_caches)
        end

        shallow_caches = prev_caches || ({} of Int32 => Array(AGPT::LayerKVCache))
        # Seed root with empty caches so segments whose parent is root resolve
        unless shallow_caches.has_key?(@corpus.root.id)
          shallow_caches[@corpus.root.id] = Array.new(n_layers) {
            AGPT::LayerKVCache.new(head_dims, seq_len)
          }
        end
        # Also seed any branching parent at depth d in (0..d_branch) that isn't
        # already in shallow_caches (e.g. when shallow regime processed depths
        # but the segment's parent is at an earlier depth than the last shallow
        # depth's nodes). We rely on shallow regime's prev_caches being keyed
        # by node_id; for parents at intermediate shallow depths, they should
        # have been keyed there too (since each_depth_level returns them). For
        # extra safety in the all-deep regime, walk up and reconstruct on demand.

        # --- Deep regime: pack per root-child ---
        #
        # For each root-child rc, run packed forward over ALL of rc's deep
        # segments (in topological start_depth order, so later segments see
        # earlier chain tails via node_caches_deep). Accumulate results across
        # all of rc's forwards, THEN fire one update per depth in rc's scope.
        #
        # Cadence: each (depth, root_child) bucket fires exactly one update per
        # epoch — same as today. Staleness bounded by max_depth updates per rc.
        node_caches_deep = shallow_caches

        # Partition all segments by root-child of their deep-slice parent
        segments_by_rc = Hash(Int32, Array(TrieCorpus::Segment)).new do |h, k|
          h[k] = [] of TrieCorpus::Segment
        end
        @corpus.build_segments.each do |seg|
          end_depth = seg.start_depth + seg.node_ids.size - 1
          next if end_depth <= effective_d_branch
          start_idx = seg.start_depth > effective_d_branch ? 0 : (effective_d_branch - seg.start_depth + 1)
          parent_id = start_idx == 0 ? seg.parent_id : seg.node_ids[start_idx - 1]
          rc = node_root_child[parent_id]? || derive_root_child(parent_id)
          segments_by_rc[rc] << seg
        end

        segments_by_rc.each do |rc, rc_segments|
          rc_segments.sort_by! { |s| s.start_depth }

          # Group this rc's segments by start_depth (segments at same
          # start_depth are independent; segments at later start_depth may
          # depend on earlier ones via the shared kv_store + node_caches_deep).
          rc_groups = Hash(Int32, Array(TrieCorpus::Segment)).new do |h, k|
            h[k] = [] of TrieCorpus::Segment
          end
          rc_segments.each { |s| rc_groups[s.start_depth] << s }

          rc_results_by_depth = Hash(Int32, Array(BatchedDepthForward::NodeResult)).new do |h, k|
            h[k] = [] of BatchedDepthForward::NodeResult
          end

          # Forward phase for this rc: iterate start_depth groups, packed call
          # per group, register chain-tail caches between groups. No updates
          # fire here — weights are frozen throughout rc's forward phase.
          rc_groups.keys.sort.each do |gdepth|
            segments = rc_groups[gdepth]
            seg_inputs = [] of BatchedDepthForward::SegmentInput
            seg_to_orig = [] of TrieCorpus::Segment

            segments.each do |seg|
              start_idx = seg.start_depth > effective_d_branch ? 0 : (effective_d_branch - seg.start_depth + 1)
              parent_id = start_idx == 0 ? seg.parent_id : seg.node_ids[start_idx - 1]
              parent_cache = node_caches_deep[parent_id]?
              next unless parent_cache

              deep_nodes = [] of TrieNode
              deep_len = seg.node_ids.size - start_idx
              deep_len.times do |k|
                idx = start_idx + k
                d = seg.start_depth + idx
                break if (d - 1) >= seq_len
                deep_nodes << @corpus.node_for_id(seg.node_ids[idx])
              end
              next if deep_nodes.empty?

              anc_base = build_ancestors(parent_id, seg, start_idx, node_ancestor_ids)
              node_ancestor_ids[parent_id] = anc_base unless node_ancestor_ids.has_key?(parent_id)

              deep_start_depth = seg.start_depth + start_idx
              deep_nodes.each_with_index do |node, k|
                node_positions[node.id] = deep_start_depth + k - 1
                node_root_child[node.id] = rc
              end

              seg_inputs << BatchedDepthForward::SegmentInput.new(
                chain_nodes: deep_nodes,
                parent_cache: parent_cache,
                ancestor_ids_base: anc_base,
                start_depth: deep_start_depth
              )
              seg_to_orig << seg
            end

            next if seg_inputs.empty?

            seg_results = BatchedDepthForward.forward_segments(seg_inputs, kv_store, model, @corpus)

            seg_to_orig.each_with_index do |orig, si|
              tail_id = orig.node_ids.last
              node_caches_deep[tail_id] = seg_results[si].extended_cache
            end

            seg_results.each do |sr|
              sr.node_results.each do |nr|
                node_ancestor_ids[nr.node_id] = nr.ancestor_ids
                rc_results_by_depth[nr.position + 1] << nr
              end
            end
          end

          # Backward+update phase for this rc: one update per depth. All
          # results in this bucket share the same rc by construction.
          next if rc_results_by_depth.empty?

          rc_flat = rc_results_by_depth.values.flatten
          rc_loss_grads = compute_loss_grads(rc_flat, model, pointerof(total_loss), pointerof(nodes_trained))

          rc_results_by_depth.keys.sort.each do |d|
            depth_results = rc_results_by_depth[d]
            zero_gradients(model)
            grad_accums = {} of Int32 => NodeGradAccum
            sub_grads = depth_results.map do |nr|
              rc_loss_grads.delete(nr.node_id) || Mat.new(1, model.config.vocab_size)
            end
            BatchedDepthBackward.backward_depth(
              depth_results, sub_grads, grad_accums, kv_store, model, @corpus, nil
            )
            if depth_results.size > 0
              scale_gradients(model, 1.0 / depth_results.size)
              lr = model.config.learning_rate
              model.embedding.update(lr)
              model.blocks.each &.update(lr)
              model.final_norm.update(lr)
              model.output.update(lr)
            end
          end
        end

        mean_loss = nodes_trained > 0 ? total_loss / nodes_trained : 0.0
        {mean_loss, nodes_trained}
      end

      private def compute_loss_grads(
        results : Array(BatchedDepthForward::NodeResult),
        model : MiniGPT,
        total_loss_ptr : Pointer(Float64),
        nodes_trained_ptr : Pointer(Int32)
      ) : Hash(Int32, Mat)
        loss_grads = {} of Int32 => Mat
        n_results = results.size
        return loss_grads if n_results == 0

        vocab_size = model.config.vocab_size
        logits_batched = Mat.new(n_results, vocab_size)
        n_results.times do |i|
          vocab_size.times { |j| logits_batched[i, j] = results[i].logits[0, j] }
        end
        probs_batched = MicroGPT.backend.softmax_rows(logits_batched)
        all_probs = probs_batched.data

        log_vocab = Math.log(vocab_size.to_f64)
        lambda = @entropy_lambda
        results.each_with_index do |result, i|
          node = @corpus.node_for_id(result.node_id)
          next if node.next_token_counts.empty?

          counts = node.next_token_counts_hash
          total = counts.values.sum(0)
          total_f = total.to_f64

          entropy = 0.0
          if lambda > 0.0 && counts.size > 1
            counts.each do |_tok, count|
              q = count / total_f
              entropy -= q * Math.log(q) if q > 0.0
            end
          end
          weight = (lambda > 0.0) ? 1.0 + lambda * (entropy / log_vocab) : 1.0

          loss_value = 0.0
          prob_offset = i * vocab_size
          counts.each do |token_id, count|
            loss_value -= count * Math.log(all_probs[prob_offset + token_id] + 1e-10)
          end
          loss_value /= total
          loss_value *= weight

          grad = Mat.new(1, vocab_size)
          weight_f32 = weight.to_f32
          vocab_size.times { |j| grad[0, j] = all_probs[prob_offset + j] * weight_f32 }
          counts.each do |token_id, count|
            grad[0, token_id] -= (count.to_f32 / total) * weight_f32
          end

          loss_grads[result.node_id] = grad
          total_loss_ptr.value = total_loss_ptr.value + loss_value
          nodes_trained_ptr.value = nodes_trained_ptr.value + 1
        end
        loss_grads
      end

      private def run_subtrie_backward_updates(
        eligible : Array(TrieNode),
        results : Array(BatchedDepthForward::NodeResult),
        loss_grads : Hash(Int32, Mat),
        node_root_child : Hash(Int32, Int32),
        kv_store : NodeKVStore,
        model : MiniGPT,
        this_caches : Hash(Int32, Array(AGPT::LayerKVCache))
      )
        result_map = {} of Int32 => BatchedDepthForward::NodeResult
        results.each { |r| result_map[r.node_id] = r }

        subtries = {} of Int32 => Array(BatchedDepthForward::NodeResult)
        eligible.each do |node|
          root_id = node_root_child[node.id]
          (subtries[root_id] ||= [] of BatchedDepthForward::NodeResult) << result_map[node.id]
        end

        subtries.each do |_rc, subtrie_results|
          zero_gradients(model)
          grad_accums = {} of Int32 => NodeGradAccum
          subtrie_grads = subtrie_results.map do |r|
            loss_grads.delete(r.node_id) || Mat.new(1, model.config.vocab_size)
          end
          BatchedDepthBackward.backward_depth(
            subtrie_results, subtrie_grads, grad_accums, kv_store, model, @corpus, this_caches
          )
          if subtrie_results.size > 0
            scale_gradients(model, 1.0 / subtrie_results.size)
            lr = model.config.learning_rate
            model.embedding.update(lr)
            model.blocks.each &.update(lr)
            model.final_norm.update(lr)
            model.output.update(lr)
          end
        end
      end

      private def build_ancestors(
        parent_id : Int32,
        seg : TrieCorpus::Segment,
        start_idx : Int32,
        node_ancestor_ids : Hash(Int32, Array(Int32))
      ) : Array(Int32)
        return node_ancestor_ids[parent_id] if node_ancestor_ids.has_key?(parent_id)

        # Walk parent chain from root to parent_id
        chain = [] of Int32
        cur = parent_id
        while cur != -1 && cur != @corpus.root.id
          chain << cur
          cur = @corpus.parent_id(cur)
        end
        chain.reverse!
        chain
      end

      private def derive_root_child(node_id : Int32) : Int32
        cur = node_id
        loop do
          p = @corpus.parent_id(cur)
          return cur if p == @corpus.root.id || p == -1
          cur = p
        end
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
