module MicroGPT
  module AGPT
    # Batched forward pass for all nodes at a single depth level.
    #
    # Linear operations (embedding, LN, Q/K/V projections, FFN) are batched into
    # single [N, d_model] × W matmuls across all N nodes at the depth level.
    # Attention is per-node (each node has a different KV history from its prefix).
    #
    # This is where the trie's compute savings are realized: the projections
    # dominate FLOPs (~85% at small d_model) and are processed in one shot.
    module BatchedDepthForward
      extend self
      include MathUtils

      # Per-node result from the batched forward: everything needed for backward.
      record NodeResult,
        node_id : Int32,
        logits : Mat,                         # [1, vocab_size]
        # Saved state for backward reconstruction:
        block_states : Array(BlockStepState),
        final_x : Mat,                        # [1, d_model] pre-final-norm
        final_normed : Mat,                   # [1, d_model] normalized (no gamma/beta)
        final_std_inv : Float64,
        final_norm_out : Mat,                 # [1, d_model] post-final-norm
        token_id : Int32,
        position : Int32,
        ancestor_ids : Array(Int32)

      # Forward all nodes at one depth level.
      #
      # nodes:       trie nodes at this depth (already filtered for seq_len)
      # kv_store:    compact K/V store for KV cache reconstruction
      # model:       the MiniGPT model (weights read-only)
      #
      # Returns array of NodeResult (one per node) + extends kv_store with new K/V.
      # parent_caches: live KV caches from the previous depth level, keyed by node_id.
      # Returns {results, this_depth_caches} — caller keeps this_depth_caches for
      # the next depth and frees parent_caches.
      def forward_depth(
        nodes : Array(TrieNode),
        node_ancestor_ids : Hash(Int32, Array(Int32)),
        node_positions : Hash(Int32, Int32),
        kv_store : NodeKVStore,
        model : MiniGPT,
        corpus : TrieCorpus,
        parent_caches : Hash(Int32, Array(LayerKVCache))? = nil
      ) : {Array(NodeResult), Hash(Int32, Array(LayerKVCache))}
        empty_caches = {} of Int32 => Array(LayerKVCache)
        return {[] of NodeResult, empty_caches} if nodes.empty?

        n = nodes.size
        d_model = model.config.d_model
        seq_len = model.config.seq_len
        head_dims = model.blocks.first.attn.head_dims
        n_layers = model.config.n_layers
        n_heads = head_dims.size

        # --- Gather tokens and metadata ---
        tokens = Array(Int32).new(n)
        positions = Array(Int32).new(n)
        ancestors = Array(Array(Int32)).new(n)

        nodes.each do |node|
          parent = node.parent.not_nil!
          token = node.token_id.not_nil!
          position = node_positions[node.id]? || (node.depth - 1)
          parent_ancestors = node_ancestor_ids[parent.id]
          anc = parent_ancestors + [node.id]

          tokens << token
          positions << position
          ancestors << anc

          # Update metadata maps
          node_ancestor_ids[node.id] = anc
          node_positions[node.id] = position
        end

        # --- Batch embedding: gather N rows from token_emb ---
        x = Mat.new(n, d_model)
        n.times do |i|
          d_model.times { |j| x[i, j] = model.embedding.token_emb[tokens[i], j] }
        end

        # --- Per-block forward ---
        # We need per-node BlockStepState for backward, so we track intermediates.
        all_block_states = Array(Array(BlockStepState)).new(n) { [] of BlockStepState }
        # Collect per-node, per-layer KV caches to pass to the next depth level
        this_depth_caches = Hash(Int32, Array(LayerKVCache)).new
        n.times { |i| this_depth_caches[nodes[i].id] = Array(LayerKVCache).new(n_layers) }

        model.blocks.each_with_index do |block, li|
          attn = block.attn

          # Save block input for residual backward
          x_input = copy_mat(x)

          # Batched LN1: [N, d_model] → [N, d_model]
          x_norm, ln1_norm, ln1_std_inv = MicroGPT.backend.layer_norm_forward(x, block.ln1.gamma, block.ln1.beta)
          ln1_out = copy_mat(x_norm)

          # Batched Q/K/V projections: [N, d_model] × W → [N, d_model]
          q_all = x_norm * attn.wq.w
          add_bias_rows!(q_all, attn.wq.b)
          k_all = x_norm * attn.wk.w
          add_bias_rows!(k_all, attn.wk.b)
          v_all = x_norm * attn.wv.w
          add_bias_rows!(v_all, attn.wv.b)

          # Split by heads: each is [N, head_dim]
          q_parts = split_cols(q_all, head_dims)
          k_new_parts = split_cols(k_all, head_dims)
          v_new_parts = split_cols(v_all, head_dims)

          # Apply RoPE per head — all nodes at same depth share position
          n_heads.times do |hi|
            n.times do |i|
              apply_rope_row!(q_parts[hi], i, attn.ropes[hi], positions[i])
              apply_rope_row!(k_new_parts[hi], i, attn.ropes[hi], positions[i])
            end
          end

          # --- Per-node attention (different KV histories) ---
          # Each node attends to its own prefix. When parent_caches are available,
          # we extend the parent's cache by one entry instead of reconstructing
          # from scratch — O(1) per node instead of O(depth).
          attn_outputs = Mat.new(n, d_model)
          per_node_attn_weights = Array(Array(Mat)).new(n)

          n.times do |i|
            GC.collect if i > 0 && i % 500 == 0  # free temporary KV slices

            node_id = nodes[i].id
            parent_id = nodes[i].parent.not_nil!.id

            # Fast path: extend parent's cache if available
            layer_cache = if pc = parent_caches
              if parent_layer_caches = pc[parent_id]?
                parent_lc = parent_layer_caches[li]
                # Unary: reuse parent cache directly; branching: copy first
                if nodes[i].parent.not_nil!.children.size > 1
                  parent_lc.deep_clone
                else
                  parent_lc
                end
              else
                kv_store.reconstruct_layer_cache(node_id, corpus, li, head_dims, seq_len)
              end
            else
              kv_store.reconstruct_layer_cache(node_id, corpus, li, head_dims, seq_len)
            end

            # Extract this node's K/V from the batched projection
            k_row_parts = Array(Mat).new(n_heads)
            v_row_parts = Array(Mat).new(n_heads)
            n_heads.times do |hi|
              hd = head_dims[hi]
              k_row = Mat.new(1, hd)
              v_row = Mat.new(1, hd)
              hd.times do |j|
                k_row[0, j] = k_new_parts[hi][i, j]
                v_row[0, j] = v_new_parts[hi][i, j]
              end
              k_row_parts << k_row
              v_row_parts << v_row
            end

            # Extend cache with this node's K/V
            layer_cache.extend(k_row_parts, v_row_parts)

            # Per-head attention
            head_attn_weights = Array(Mat).new(n_heads)
            col_offset = 0
            n_heads.times do |hi|
              hd = head_dims[hi]

              q_h = Mat.new(1, hd)
              hd.times { |j| q_h[0, j] = q_parts[hi][i, j] }

              k_full = layer_cache.k_slice(hi)  # [prefix_len, hd]
              v_full = layer_cache.v_slice(hi)  # [prefix_len, hd]

              scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32
              scores = q_h * k_full.t  # [1, prefix_len]
              scores.scale!(scale)
              weights = MicroGPT.backend.softmax_rows(scores)
              out = weights * v_full  # [1, hd]

              head_attn_weights << copy_mat(weights)

              hd.times { |j| attn_outputs[i, col_offset + j] = out[0, j] }
              col_offset += hd
            end

            per_node_attn_weights << head_attn_weights
            this_depth_caches[node_id] << layer_cache

            # Store this node's K/V for this layer
            kv_store.store_layer(node_id, li, k_row_parts, v_row_parts)
          end

          # Batched WO: [N, d_model] × W_o → [N, d_model]
          wo_input = copy_mat(attn_outputs)
          attn_proj = attn_outputs * attn.wo.w
          add_bias_rows!(attn_proj, attn.wo.b)

          # Residual 1
          x.add!(attn_proj)
          x_after_attn = copy_mat(x)

          # Batched LN2
          x_norm2, ln2_norm, ln2_std_inv = MicroGPT.backend.layer_norm_forward(x, block.ln2.gamma, block.ln2.beta)
          ln2_out = copy_mat(x_norm2)

          # Batched FFN L1 + ReLU
          h = x_norm2 * block.ff.l1.w
          add_bias_rows!(h, block.ff.l1.b)
          ff_relu_mask = Mat.new(h.rows, h.cols)
          h.rows.times do |r|
            h.cols.times do |c|
              ff_relu_mask[r, c] = h[r, c] > 0 ? 1.0_f32 : 0.0_f32
              h[r, c] *= ff_relu_mask[r, c]
            end
          end
          ff_relu_out = copy_mat(h)

          # Batched FFN L2
          ff_out = h * block.ff.l2.w
          add_bias_rows!(ff_out, block.ff.l2.b)

          # Residual 2
          x.add!(ff_out)

          # Save per-node BlockStepState (extracting rows from batched matrices)
          n.times do |i|
            bss = BlockStepState.new(
              x_input: extract_row(x_input, i),
              ln1_out: extract_row(ln1_out, i),
              ln1_normed: extract_row(ln1_norm, i),
              ln1_std_inv: ln1_std_inv[i, 0].to_f64,
              q_parts: head_dims.map_with_index { |hd, hi| extract_row_slice(q_parts[hi], i) },
              attn_weights: per_node_attn_weights[i],
              wo_input: extract_row(wo_input, i),
              x_after_attn: extract_row(x_after_attn, i),
              ln2_out: extract_row(ln2_out, i),
              ln2_normed: extract_row(ln2_norm, i),
              ln2_std_inv: ln2_std_inv[i, 0].to_f64,
              ff_relu_out: extract_row(ff_relu_out, i),
              ff_relu_mask: extract_row(ff_relu_mask, i)
            )
            all_block_states[i] << bss
          end
        end

        # --- Final norm (batched) ---
        final_x = copy_mat(x)
        final_out, final_norm, final_std_inv = MicroGPT.backend.layer_norm_forward(
          x, model.final_norm.gamma, model.final_norm.beta
        )
        final_norm_out = copy_mat(final_out)

        # --- Output projection (batched): [N, d_model] × W → [N, vocab_size] ---
        logits = final_out * model.output.proj.w
        add_bias_rows!(logits, model.output.proj.b)

        # --- Build per-node results ---
        results = Array(NodeResult).new(n)
        n.times do |i|
          results << NodeResult.new(
            node_id: nodes[i].id,
            logits: extract_row(logits, i),
            block_states: all_block_states[i],
            final_x: extract_row(final_x, i),
            final_normed: extract_row(final_norm, i),
            final_std_inv: final_std_inv[i, 0].to_f64,
            final_norm_out: extract_row(final_norm_out, i),
            token_id: tokens[i],
            position: positions[i],
            ancestor_ids: ancestors[i]
          )
        end

        {results, this_depth_caches}
      end

      # --- Helpers ---

      private def add_bias_rows!(m : Mat, bias : Mat)
        m.rows.times do |r|
          m.cols.times { |c| m[r, c] += bias[0, c] }
        end
      end

      private def extract_row(m : Mat, r : Int32) : Mat
        result = Mat.new(1, m.cols)
        m.cols.times { |c| result[0, c] = m[r, c] }
        result
      end

      private def extract_row_slice(m : Mat, r : Int32) : Mat
        extract_row(m, r)
      end

      private def copy_mat(m : Mat) : Mat
        result = Mat.new(m.rows, m.cols)
        m.rows.times do |r|
          m.cols.times { |c| result[r, c] = m[r, c] }
        end
        result
      end

      private def apply_rope_row!(m : Mat, row : Int32, rope : RoPE, position : Int32)
        half = rope.dim // 2
        half.times do |i|
          c = rope.cos_cache[position, 2 * i]
          s = rope.sin_cache[position, 2 * i]
          x0 = m[row, 2 * i]
          x1 = m[row, 2 * i + 1]
          m[row, 2 * i]     = x0 * c - x1 * s
          m[row, 2 * i + 1] = x0 * s + x1 * c
        end
      end
    end
  end
end
