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

      def initialize(@corpus : TrieCorpus, @loss_fn = WeightedNextTokenLoss.new)
        @observed_count = 0
        @corpus.each_observed_node do |node|
          @observed_count += 1 unless node.depth == 0
        end
      end

      # Run one full trie-walk training epoch using batched depth-level operations.
      # Returns {mean_loss, nodes_trained}.
      def train_epoch(model : MiniGPT) : {Float64, Int32}
        seq_len = model.config.seq_len
        head_dims = model.blocks.first.attn.head_dims
        n_layers = model.config.n_layers

        zero_gradients(model)

        total_loss = 0.0
        nodes_trained = 0

        # Compact per-node K/V storage (~1 KB/node) — persists across entire epoch
        kv_store = NodeKVStore.new

        # Per-node metadata (ancestor_ids, positions)
        node_ancestor_ids = {} of Int32 => Array(Int32)
        node_positions = {} of Int32 => Int32

        node_ancestor_ids[@corpus.root.id] = [] of Int32

        # Collect depth-level node lists for backward re-forward.
        # Forward pass stores only K/V entries (~1 KB/node) + lightweight loss info.
        depth_levels = [] of Array(TrieNode)
        depth_loss_info = [] of Hash(Int32, Hash(Int32, Int32))

        # --- Batched Forward BFS: depth 0 → max ---
        # Stores K/V entries into kv_store. Does NOT keep NodeResults in memory.
        prev_caches : Hash(Int32, Array(AGPT::LayerKVCache))? = nil

        @corpus.each_depth_level do |depth, nodes|
          next if depth == 0

          eligible = Array(TrieNode).new
          nodes.each do |node|
            parent = node.parent.not_nil!
            next unless node_ancestor_ids.has_key?(parent.id)
            parent_depth = parent.depth
            next if parent_depth >= seq_len
            eligible << node
          end
          next if eligible.empty?

          eligible.each do |node|
            node_positions[node.id] = depth - 1
          end

          # Batched forward — computes logits and stores K/V entries in kv_store.
          # We keep NodeResults only long enough to compute loss, then discard.
          results, this_caches = BatchedDepthForward.forward_depth(
            eligible, node_ancestor_ids, node_positions, kv_store, model, @corpus, prev_caches
          )
          prev_caches = this_caches

          loss_info = {} of Int32 => Hash(Int32, Int32)
          results.each do |result|
            node = @corpus.node_for_id(result.node_id)
            unless node.next_token_counts.empty?
              counts = node.next_token_counts_hash
              loss_value, _ = @loss_fn.loss_and_backward(result.logits, counts)
              loss_info[result.node_id] = counts
              total_loss += loss_value
              nodes_trained += 1
            end
          end

          depth_levels << eligible
          depth_loss_info << loss_info
        end

        # Free forward caches — only K/V store entries remain
        prev_caches = nil
        GC.collect

        # --- Batched Backward BFS: depth max → 0 ---
        # Re-forward each depth level (batched) to regenerate BlockStepState,
        # then batched backward. Only one depth level's states live at a time.
        grad_accums = {} of Int32 => NodeGradAccum

        (depth_levels.size - 1).downto(0) do |di|
          eligible = depth_levels[di]
          loss_info = depth_loss_info[di]

          # Re-forward this depth (batched matmuls) to regenerate states for backward
          results, _ = BatchedDepthForward.forward_depth(
            eligible, node_ancestor_ids, node_positions, kv_store, model, @corpus
          )

          # Build loss gradients
          loss_grads = results.map do |result|
            if counts = loss_info[result.node_id]?
              logits = model.output.forward(model.final_norm.forward(result.final_x))
              _, dl = @loss_fn.loss_and_backward(logits, counts)
              dl
            else
              Mat.new(1, model.config.vocab_size)
            end
          end

          # Batched backward
          BatchedDepthBackward.backward_depth(
            results, loss_grads, grad_accums, kv_store, model, @corpus
          )

          # Per-depth weight update
          lr = model.config.learning_rate
          model.embedding.update(lr)
          model.blocks.each &.update(lr)
          model.final_norm.update(lr)
          model.output.update(lr)
          zero_gradients(model)
        end

        mean_loss = nodes_trained > 0 ? total_loss / nodes_trained : 0.0
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

    end
  end
end
