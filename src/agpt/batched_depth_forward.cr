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

      # Minimal node descriptor for forward_depth.
      # Decouples the hot path from TrieNode/LoadedRecord internals.
      # Build from a TrieNode via NodeProxy.from_trie_node, or from a
      # LoadedRecord via NodeProxy.from_record.
      record NodeProxy,
        id : Int32,
        token_id : Int32,
        depth : Int32

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

      # ------------------------------------------------------------------------
      # Unary chain compression (scaffolding for next implementation phase)
      # ------------------------------------------------------------------------
      #
      # A unary chain is a sequence of trie nodes [n_0, n_1, ..., n_{L-1}] where
      # each n_i has exactly one child n_{i+1} (the tail n_{L-1} may be a leaf
      # or a branching node).
      #
      # Current forward (depth-major): processes one token per node per depth
      # level. L nodes in a chain = L separate 1-token forward calls.
      #
      # Chain-compressed forward: processes the entire chain as one window-style
      # batched forward — equivalent to MiniGPT.forward([t_0, t_1, ..., t_{L-1}])
      # but:
      #   (a) attention starts from the parent's KV cache (not empty),
      #   (b) per-position BlockStepState is saved for later backward,
      #   (c) per-position K/V is stored in the NodeKVStore keyed by chain node id,
      #   (d) per-position logits are returned as NodeResults (one per chain position).
      #
      # Target: reduce L per-node attention ops into a single [L, L] causal
      # attention matmul per head per layer. Projections/FFN become [L, d_model]
      # matmuls.
      #
      # Contract (what this method MUST produce to pass the reference spec):
      #   - logits at each position match the per-node path to float tolerance
      #   - final_x at each position matches
      #   - K/V stored per position matches (post-RoPE)
      #
      # Not yet implemented — this signature is the design target for next session.
      # The implementation should:
      #   1. Embed the chain tokens into a [L, d_model] matrix
      #   2. For each block li:
      #      a. Batched LN1 over [L, d_model]
      #      b. Batched Q/K/V projections into [L, d_model] each
      #      c. Per-head RoPE at each position's absolute depth
      #      d. Attention where Q_i attends to (parent_cache.k_slice + chain_K[0..i])
      #         — this is the causal attention that prefix_len varies per position
      #         — consider: combined matmul [L, parent_len+L] × flattened K, then
      #           mask the forbidden future positions
      #      e. Batched WO projection → [L, d_model]
      #      f. Residual + LN2
      #      g. Batched FFN → back to [L, d_model]
      #   3. Batched final norm + output projection → [L, vocab_size]
      #   4. Build NodeResults per position
      #   5. Append chain K/V entries to kv_store
      #
      # Tricky bits to watch:
      #   - RoPE position for chain node n_i is (start_depth + i - 1), NOT i
      #   - ancestor_ids for position i = parent_chain + [chain[0..i].ids]
      #   - BlockStepState attn_weights must be [1, parent_len + i + 1] per position
      #     (not uniform — different i has different prefix_len)
      #   - Mask: position i in chain must NOT see positions j > i within the chain
      def forward_unary_chain(
        chain_nodes : Array(TrieNode),
        parent_cache : Array(LayerKVCache),
        ancestor_ids_base : Array(Int32),   # ancestor ids of chain[0]'s parent
        start_depth : Int32,
        kv_store : NodeKVStore,
        model : MiniGPT,
        corpus : TrieAccessor
      ) : {Array(NodeResult), Array(LayerKVCache)}
        l = chain_nodes.size
        raise ArgumentError.new("forward_unary_chain: empty chain") if l == 0

        d_model = model.config.d_model
        head_dims = model.blocks.first.attn.head_dims
        n_layers = model.config.n_layers
        n_heads = head_dims.size

        tokens = chain_nodes.map { |n| n.token_id.not_nil! }
        # Position for chain[i] = absolute depth - 1 = (start_depth + i) - 1
        positions = Array(Int32).new(l) { |i| start_depth + i - 1 }

        # --- Embedding [L, d_model] ---
        x = Mat.new(l, d_model)
        l.times do |i|
          d_model.times { |j| x[i, j] = model.embedding.token_emb[tokens[i], j] }
        end

        all_block_states = Array(Array(BlockStepState)).new(l) { [] of BlockStepState }
        extended_cache = Array(LayerKVCache).new(n_layers)

        n_layers.times do |li|
          block = model.blocks[li]
          attn = block.attn
          parent_layer_cache = parent_cache[li]
          prefix_len = parent_layer_cache.len

          x_input = copy_mat(x)

          x_norm, ln1_norm, ln1_std_inv = MicroGPT.backend.layer_norm_forward(x, block.ln1.gamma, block.ln1.beta)
          ln1_out = copy_mat(x_norm)

          q_all = x_norm * attn.wq.w
          add_bias_rows!(q_all, attn.wq.b)
          k_all = x_norm * attn.wk.w
          add_bias_rows!(k_all, attn.wk.b)
          v_all = x_norm * attn.wv.w
          add_bias_rows!(v_all, attn.wv.b)

          q_parts = split_cols(q_all, head_dims)
          k_parts = split_cols(k_all, head_dims)
          v_parts = split_cols(v_all, head_dims)

          # RoPE per head at each position's absolute depth
          n_heads.times do |hi|
            l.times do |i|
              apply_rope_row!(q_parts[hi], i, attn.ropes[hi], positions[i])
              apply_rope_row!(k_parts[hi], i, attn.ropes[hi], positions[i])
            end
          end

          # Store per-position K/V into kv_store and build extended layer cache
          layer_cache = parent_layer_cache.deep_clone
          l.times do |i|
            k_row_parts = Array(Mat).new(n_heads)
            v_row_parts = Array(Mat).new(n_heads)
            n_heads.times do |hi|
              hd = head_dims[hi]
              k_row = Mat.new(1, hd)
              v_row = Mat.new(1, hd)
              hd.times do |j|
                k_row[0, j] = k_parts[hi][i, j]
                v_row[0, j] = v_parts[hi][i, j]
              end
              k_row_parts << k_row
              v_row_parts << v_row
            end
            kv_store.store_layer(chain_nodes[i].id, li, k_row_parts, v_row_parts)
            layer_cache.extend(k_row_parts, v_row_parts)
          end
          extended_cache << layer_cache

          # --- Per-position causal attention over (parent prefix + chain so far) ---
          attn_outputs = Mat.new(l, d_model)
          per_pos_attn_weights = Array(Array(Mat)).new(l) { Array(Mat).new(n_heads) }

          col_offset = 0
          n_heads.times do |hi|
            hd = head_dims[hi]
            scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32

            full_len_max = prefix_len + l
            k_full = Mat.new(full_len_max, hd)
            v_full = Mat.new(full_len_max, hd)
            if prefix_len > 0
              kp = parent_layer_cache.k_slice(hi)
              vp = parent_layer_cache.v_slice(hi)
              prefix_len.times do |r|
                hd.times do |j|
                  k_full[r, j] = kp[r, j]
                  v_full[r, j] = vp[r, j]
                end
              end
            end
            l.times do |i|
              hd.times do |j|
                k_full[prefix_len + i, j] = k_parts[hi][i, j]
                v_full[prefix_len + i, j] = v_parts[hi][i, j]
              end
            end

            l.times do |i|
              eff_len = prefix_len + i + 1
              q_row = Mat.new(1, hd)
              hd.times { |j| q_row[0, j] = q_parts[hi][i, j] }

              k_eff = Mat.new(eff_len, hd)
              v_eff = Mat.new(eff_len, hd)
              eff_len.times do |r|
                hd.times do |j|
                  k_eff[r, j] = k_full[r, j]
                  v_eff[r, j] = v_full[r, j]
                end
              end

              scores = q_row * k_eff.t
              scores.scale!(scale)
              weights = MicroGPT.backend.softmax_rows(scores)
              out = weights * v_eff

              hd.times { |j| attn_outputs[i, col_offset + j] = out[0, j] }
              per_pos_attn_weights[i] << copy_mat(weights)
            end

            col_offset += hd
          end

          # WO + residual 1
          wo_input = copy_mat(attn_outputs)
          attn_proj = attn_outputs * attn.wo.w
          add_bias_rows!(attn_proj, attn.wo.b)
          x.add!(attn_proj)
          x_after_attn = copy_mat(x)

          # LN2
          x_norm2, ln2_norm, ln2_std_inv = MicroGPT.backend.layer_norm_forward(x, block.ln2.gamma, block.ln2.beta)
          ln2_out = copy_mat(x_norm2)

          # FFN L1 + ReLU
          h = x_norm2 * block.ff.l1.w
          if MicroGPT.backend.is_a?(MicroGPT::CuBLASBackend)
            h, ff_relu_mask = MicroGPT.backend.fused_bias_relu(h, block.ff.l1.b)
          else
            add_bias_rows!(h, block.ff.l1.b)
            ff_relu_mask = Mat.new(h.rows, h.cols)
            h.rows.times do |r|
              h.cols.times do |c|
                ff_relu_mask[r, c] = h[r, c] > 0 ? 1.0_f32 : 0.0_f32
                h[r, c] *= ff_relu_mask[r, c]
              end
            end
          end
          ff_relu_out = copy_mat(h)

          # FFN L2 + residual 2
          ff_out = h * block.ff.l2.w
          add_bias_rows!(ff_out, block.ff.l2.b)
          x.add!(ff_out)

          # Per-position BlockStepState
          l.times do |i|
            bss = BlockStepState.new(
              x_input: extract_row(x_input, i),
              ln1_out: extract_row(ln1_out, i),
              ln1_normed: extract_row(ln1_norm, i),
              ln1_std_inv: ln1_std_inv[i, 0].to_f64,
              q_parts: head_dims.map_with_index { |_, hi| extract_row_slice(q_parts[hi], i) },
              attn_weights: per_pos_attn_weights[i],
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

        # --- Final norm + output projection (batched over L) ---
        final_x = copy_mat(x)
        final_out, final_norm, final_std_inv = MicroGPT.backend.layer_norm_forward(
          x, model.final_norm.gamma, model.final_norm.beta
        )
        final_norm_out = copy_mat(final_out)

        logits = final_out * model.output.proj.w
        add_bias_rows!(logits, model.output.proj.b)

        results = Array(NodeResult).new(l)
        running_ancestors = ancestor_ids_base.dup
        l.times do |i|
          running_ancestors = running_ancestors + [chain_nodes[i].id]
          results << NodeResult.new(
            node_id: chain_nodes[i].id,
            logits: extract_row(logits, i),
            block_states: all_block_states[i],
            final_x: extract_row(final_x, i),
            final_normed: extract_row(final_norm, i),
            final_std_inv: final_std_inv[i, 0].to_f64,
            final_norm_out: extract_row(final_norm_out, i),
            token_id: tokens[i],
            position: positions[i],
            ancestor_ids: running_ancestors.dup
          )
        end

        {results, extended_cache}
      end

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
        nodes : Array(NodeProxy),
        node_ancestor_ids : Hash(Int32, Array(Int32)),
        node_positions : Hash(Int32, Int32),
        kv_store : NodeKVStore,
        model : MiniGPT,
        corpus : TrieAccessor,
        parent_caches : Hash(Int32, Array(LayerKVCache))? = nil
      ) : {Array(NodeResult), Hash(Int32, Array(LayerKVCache))}
        scoped_result = nil.as({Array(NodeResult), Hash(Int32, Array(LayerKVCache))}?)
        MicroGPT::PerfTrace.with_scope("agpt.forward") do
          empty_caches = {} of Int32 => Array(LayerKVCache)
          scoped_result = {[] of NodeResult, empty_caches}
          next if nodes.empty?

          n = nodes.size
          d_model = model.config.d_model
          seq_len = model.config.seq_len
          head_dims = model.blocks.first.attn.head_dims
          n_layers = model.config.n_layers
          n_heads = head_dims.size

          # --- Gather tokens and metadata ---
          metadata_started = Time.instant if MicroGPT::PerfTrace.enabled?
          tokens = Array(Int32).new(n)
          positions = Array(Int32).new(n)
          ancestors = Array(Array(Int32)).new(n)

          nodes.each do |node|
            token = node.token_id
            position = node_positions[node.id]? || (node.depth - 1)
            parent_ancestors = node_ancestor_ids[corpus.parent_id(node.id)]
            anc = parent_ancestors + [node.id]

            tokens << token
            positions << position
            ancestors << anc

            # Update metadata maps
            node_ancestor_ids[node.id] = anc
            node_positions[node.id] = position
          end
          MicroGPT::PerfTrace.add_time("agpt.forward.metadata", Time.instant - metadata_started.not_nil!) if metadata_started

          # --- Batch embedding: gather N rows from token_emb ---
          embedding_started = Time.instant if MicroGPT::PerfTrace.enabled?
          x = if backend = MicroGPT.backend.as?(MicroGPT::CuBLASBackend)
            backend.embedding_gather(model.embedding.token_emb, tokens, n, d_model)
          else
            m = Mat.new(n, d_model)
            n.times do |i|
              d_model.times { |j| m[i, j] = model.embedding.token_emb[tokens[i], j] }
            end
            m
          end
          MicroGPT::PerfTrace.add_time("agpt.forward.embedding", Time.instant - embedding_started.not_nil!) if embedding_started

          # --- Per-block forward ---
          # We need per-node BlockStepState for backward, so we track intermediates.
          all_block_states = Array(Array(BlockStepState)).new(n) { [] of BlockStepState }
          # Collect per-node, per-layer KV caches to pass to the next depth level
          this_depth_caches = Hash(Int32, Array(LayerKVCache)).new
          n.times { |i| this_depth_caches[nodes[i].id] = Array(LayerKVCache).new(n_layers) }

          model.blocks.each_with_index do |block, li|
            attn = block.attn
            block_started = Time.instant if MicroGPT::PerfTrace.enabled?

            # Save block input for residual backward
            x_input = copy_mat(x)

            # Batched LN1: [N, d_model] → [N, d_model]
            qkv_started = Time.instant if MicroGPT::PerfTrace.enabled?
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
            q_parts = nil.as(Array(Mat)?)
            k_new_parts = nil.as(Array(Mat)?)
            v_new_parts = nil.as(Array(Mat)?)
            trace_sync_delta("agpt.forward.layer#{li}.split_qkv") do
              q_parts = split_cols(q_all, head_dims)
              k_new_parts = split_cols(k_all, head_dims)
              v_new_parts = split_cols(v_all, head_dims)
            end
            q_parts = q_parts.not_nil!
            k_new_parts = k_new_parts.not_nil!
            v_new_parts = v_new_parts.not_nil!

            # Apply RoPE per head — all nodes at same depth share position
            trace_sync_delta("agpt.forward.layer#{li}.qkv_rope") do
              n_heads.times do |hi|
                n.times do |i|
                  apply_rope_row!(q_parts[hi], i, attn.ropes[hi], positions[i])
                  apply_rope_row!(k_new_parts[hi], i, attn.ropes[hi], positions[i])
                end
              end
            end
            MicroGPT::PerfTrace.add_time("agpt.forward.layer#{li}.qkv_rope", Time.instant - qkv_started.not_nil!) if qkv_started

            # --- Per-node attention (different KV histories) ---
            # Store each node's K/V contribution first, then compute attention.
            # On GPU: pack Q + KV into contiguous buffers, one kernel launch.
            # On CPU: per-node loop with incremental parent caches.

            per_node_attn_weights = Array(Array(Mat)).new(n)

            # First: extract and store each node's K/V from batched projections
            kv_store_started = Time.instant if MicroGPT::PerfTrace.enabled?
            node_k_rows = Array(Array(Mat)).new(n)
            node_v_rows = Array(Array(Mat)).new(n)
            trace_sync_delta("agpt.forward.layer#{li}.kv_store") do
              n.times do |i|
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
                node_k_rows << k_row_parts
                node_v_rows << v_row_parts
                kv_store.store_layer(nodes[i].id, li, k_row_parts, v_row_parts)
              end
            end
            MicroGPT::PerfTrace.add_time("agpt.forward.layer#{li}.kv_store", Time.instant - kv_store_started.not_nil!) if kv_store_started

            attention_started = Time.instant if MicroGPT::PerfTrace.enabled?
            if MicroGPT.backend.is_a?(MicroGPT::CuBLASBackend)
              # GPU path: pack all Q/KV, one kernel launch for all nodes
              attn_outputs = gpu_batched_attention(
                nodes, n, li, head_dims, n_heads, d_model, seq_len,
                q_parts, node_k_rows, node_v_rows,
                kv_store, corpus, per_node_attn_weights,
                this_depth_caches, parent_caches
              )
            else
              # CPU path: grouped by parent (siblings share K/V prefix)
              attn_outputs = Mat.new(n, d_model)
              if ENV["AGPT_ATTN"]? == "per_node"
                cpu_per_node_attention(
                  nodes, n, li, head_dims, n_heads, d_model, seq_len,
                  q_parts, node_k_rows, node_v_rows,
                  kv_store, corpus, parent_caches,
                  attn_outputs, per_node_attn_weights, this_depth_caches
                )
              else
                cpu_grouped_attention(
                  nodes, n, li, head_dims, n_heads, d_model, seq_len,
                  q_parts, node_k_rows, node_v_rows,
                  kv_store, corpus, parent_caches,
                  attn_outputs, per_node_attn_weights, this_depth_caches
                )
              end
            end
            MicroGPT::PerfTrace.add_time("agpt.forward.layer#{li}.attention", Time.instant - attention_started.not_nil!) if attention_started

            # Batched WO: [N, d_model] × W_o → [N, d_model]
            post_attn_started = Time.instant if MicroGPT::PerfTrace.enabled?
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
            if MicroGPT.backend.is_a?(MicroGPT::CuBLASBackend)
              h, ff_relu_mask = MicroGPT.backend.fused_bias_relu(h, block.ff.l1.b)
            else
              add_bias_rows!(h, block.ff.l1.b)
              ff_relu_mask = Mat.new(h.rows, h.cols)
              h.rows.times do |r|
                h.cols.times do |c|
                  ff_relu_mask[r, c] = h[r, c] > 0 ? 1.0_f32 : 0.0_f32
                  h[r, c] *= ff_relu_mask[r, c]
                end
              end
            end
            ff_relu_out = copy_mat(h)

            # Batched FFN L2
            ff_out = h * block.ff.l2.w
            add_bias_rows!(ff_out, block.ff.l2.b)

            # Residual 2
            x.add!(ff_out)
            MicroGPT::PerfTrace.add_time("agpt.forward.layer#{li}.post_attn_ffn", Time.instant - post_attn_started.not_nil!) if post_attn_started

            # Save per-node BlockStepState (extracting rows from batched matrices)
            state_extract_started = Time.instant if MicroGPT::PerfTrace.enabled?
            trace_sync_delta("agpt.forward.layer#{li}.state_extract") do
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
            MicroGPT::PerfTrace.add_time("agpt.forward.layer#{li}.state_extract", Time.instant - state_extract_started.not_nil!) if state_extract_started
            MicroGPT::PerfTrace.add_time("agpt.forward.layer#{li}.total", Time.instant - block_started.not_nil!) if block_started
          end

          # --- Final norm (batched) ---
          final_started = Time.instant if MicroGPT::PerfTrace.enabled?
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
          MicroGPT::PerfTrace.add_time("agpt.forward.final", Time.instant - final_started.not_nil!) if final_started

          scoped_result = {results, this_depth_caches}
        end
        scoped_result.not_nil!
      end

      # --- Helpers ---

      private def add_bias_rows!(m : Mat, bias : Mat)
        if MicroGPT.backend.is_a?(MicroGPT::CuBLASBackend)
          MicroGPT.backend.bias_add(m, bias)
        else
          m.rows.times do |r|
            m.cols.times { |c| m[r, c] += bias[0, c] }
          end
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

      # GPU path: pack Q + KV into flat arrays, one CUDA kernel call.
      private def gpu_batched_attention(
        nodes, n, li, head_dims, n_heads, d_model, seq_len,
        q_parts, node_k_rows, node_v_rows,
        kv_store, corpus, per_node_attn_weights,
        this_depth_caches,
        parent_caches : Hash(Int32, Array(LayerKVCache))? = nil
      ) : Mat
        hd = head_dims[0]  # assumes uniform heads for GPU path

        # Compute prefix lengths and max_len
        prefix_lens = Array(Int32).new(n)
        n.times do |i|
          # depth of this node = number of ancestors (excl root) + self
          prefix_lens << nodes[i].depth
        end
        max_len = prefix_lens.max

        # Pack Q: [N * n_heads, hd] — interleaved by head
        q_pack_started = Time.instant if MicroGPT::PerfTrace.enabled?
        q_flat = Array(Float32).new(n * n_heads * hd, 0.0_f32)
        n.times do |i|
          n_heads.times do |hi|
            base = (i * n_heads + hi) * hd
            hd.times { |j| q_flat[base + j] = q_parts[hi][i, j] }
          end
        end
        MicroGPT::PerfTrace.add_time("agpt.gpu_attn.q_pack", Time.instant - q_pack_started.not_nil!) if q_pack_started

        # Pack KV: for each node, collect ancestor K/V entries + own entry
        # Layout: position p of node i at kv_offset[i]+p, heads interleaved
        # Entry stride: n_heads * hd
        kv_offsets = Array(Int32).new(n, 0)
        total_kv = 0
        n.times do |i|
          kv_offsets[i] = total_kv
          total_kv += prefix_lens[i]
        end

        k_flat = Array(Float32).new(total_kv * n_heads * hd, 0.0_f32)
        v_flat = Array(Float32).new(total_kv * n_heads * hd, 0.0_f32)

        kv_pack_started = Time.instant if MicroGPT::PerfTrace.enabled?
        n.times do |i|
          node_id = nodes[i].id
          offset = kv_offsets[i]
          parent_id = corpus.parent_id(node_id)

          # Fast path: read ancestor K/V from parent_caches (leveled trainer).
          # The parent's LayerKVCache already has all prefix positions packed.
          if pc = parent_caches
            if pcs = pc[parent_id]?
              layer_cache = pcs[li]
              pl = layer_cache.len
              n_heads.times do |hi|
                k_mat = layer_cache.k_parts[hi]
                v_mat = layer_cache.v_parts[hi]
                pl.times do |pos|
                  base = (offset + pos) * n_heads * hd + hi * hd
                  hd.times do |j|
                    k_flat[base + j] = k_mat[pos, j]
                    v_flat[base + j] = v_mat[pos, j]
                  end
                end
              end
              # Own K/V at position pl
              n_heads.times do |hi|
                base = (offset + pl) * n_heads * hd + hi * hd
                hd.times do |j|
                  k_flat[base + j] = node_k_rows[i][hi][0, j]
                  v_flat[base + j] = node_v_rows[i][hi][0, j]
                end
              end
              next
            end
          end

          # Fallback: walk ancestor chain via kv_store (non-leveled trainer).
          chain = [] of Int32
          current = node_id
          while current != -1
            chain << current
            current = corpus.parent_id(current)
          end
          chain.reverse!
          # chain[0] = root (no KV), chain[1..] = ancestors + self

          pos = 0
          (1...chain.size).each do |ci|
            anc_id = chain[ci]
            entry = kv_store.entries[anc_id]?
            next unless entry
            layer_entry = entry[li]?
            next unless layer_entry
            n_heads.times do |hi|
              k_row, v_row = layer_entry[hi]
              base = (offset + pos) * n_heads * hd + hi * hd
              hd.times do |j|
                k_flat[base + j] = k_row[0, j]
                v_flat[base + j] = v_row[0, j]
              end
            end
            pos += 1
          end
        end
        MicroGPT::PerfTrace.add_time("agpt.gpu_attn.kv_pack", Time.instant - kv_pack_started.not_nil!) if kv_pack_started

        # Allocate GPU buffers and copy
        upload_started = Time.instant if MicroGPT::PerfTrace.enabled?
        q_gpu = gpu_alloc_copy(q_flat)
        k_gpu = gpu_alloc_copy(k_flat)
        v_gpu = gpu_alloc_copy(v_flat)
        offsets_gpu = gpu_alloc_copy_int(kv_offsets)
        lengths_gpu = gpu_alloc_copy_int(prefix_lens)
        out_size = n * n_heads * hd
        out_gpu = gpu_alloc(out_size)
        weights_size = n * n_heads * max_len
        weights_gpu = gpu_alloc(weights_size)
        if upload_started
          total_upload_bytes = ((q_flat.size + k_flat.size + v_flat.size) * sizeof(Float32) +
            (kv_offsets.size + prefix_lens.size) * sizeof(Int32)).to_i64
          MicroGPT::PerfTrace.increment("agpt.gpu_attn.upload")
          MicroGPT::PerfTrace.add_bytes("agpt.gpu_attn.upload", total_upload_bytes)
          MicroGPT::PerfTrace.add_time("agpt.gpu_attn.upload", Time.instant - upload_started.not_nil!)
        end

        scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32

        kernel_started = Time.instant if MicroGPT::PerfTrace.enabled?
        LibCUDAKernels.cuda_batched_varlen_attention(
          q_gpu, k_gpu, v_gpu, offsets_gpu, lengths_gpu,
          out_gpu, weights_gpu,
          n, n_heads, hd, max_len, scale
        )
        LibCUDAKernels.cuda_sync
        MicroGPT::PerfTrace.add_time("agpt.gpu_attn.kernel", Time.instant - kernel_started.not_nil!) if kernel_started

        unpacked_gpu = gpu_alloc(n * d_model)
        LibCUDAKernels.cuda_unpack_batched_attn_output(
          out_gpu, unpacked_gpu, n, n_heads, hd
        )

        download_started = Time.instant if MicroGPT::PerfTrace.enabled?
        weights_host = Array(Float32).new(weights_size, 0.0_f32)
        LibCUDA.cudaMemcpy(weights_host.to_unsafe.as(Void*), weights_gpu.as(Void*),
                           weights_size.to_u64 * sizeof(Float32), 2)
        if download_started
          MicroGPT::PerfTrace.increment("agpt.gpu_attn.download")
          MicroGPT::PerfTrace.add_bytes("agpt.gpu_attn.download", (weights_size * sizeof(Float32)).to_i64)
          MicroGPT::PerfTrace.add_time("agpt.gpu_attn.download", Time.instant - download_started.not_nil!)
        end

        attn_outputs = Mat.new(n, d_model, unpacked_gpu.as(Void*))

        n.times do |i|
          head_weights = Array(Mat).new(n_heads)
          n_heads.times do |hi|
            # Unpack weights for backward
            w = Mat.new(1, prefix_lens[i])
            w_base = (i * n_heads + hi) * max_len
            prefix_lens[i].times { |p| w[0, p] = weights_host[w_base + p] }
            head_weights << w
          end
          per_node_attn_weights << head_weights

          # Build this_depth_caches entry: clone parent cache and extend with own K/V.
          # Fast path uses parent_caches (leveled trainer); fallback reconstructs from kv_store.
          node_id = nodes[i].id
          parent_id = corpus.parent_id(node_id)
          lc = if pc = parent_caches
            if pcs = pc[parent_id]?
              pcs[li].deep_clone
            else
              kv_store.reconstruct_layer_cache(node_id, corpus, li, head_dims, seq_len)
            end
          else
            kv_store.reconstruct_layer_cache(node_id, corpus, li, head_dims, seq_len)
          end
          lc.extend(node_k_rows[i], node_v_rows[i])
          this_depth_caches[node_id] << lc
        end

        # Free GPU buffers
        gpu_free(q_gpu); gpu_free(k_gpu); gpu_free(v_gpu)
        gpu_free(offsets_gpu.as(Float32*)); gpu_free(lengths_gpu.as(Float32*))
        gpu_free(out_gpu); gpu_free(weights_gpu)

        attn_outputs
      end

      # CPU path: per-node attention with incremental parent caches
      # Grouped attention: siblings with the same parent share K/V for positions
      # 0..d-1 and differ only at position d (their own token). Batch the shared
      # part into [C, d_prefix] matmuls per group; handle self-position per node.
      #
      # Produces per-sibling outputs and weights identical to cpu_per_node_attention
      # up to float32 ordering, so backward (which reads per-node attn_weights)
      # works unchanged.
      private def cpu_grouped_attention(
        nodes, n, li, head_dims, n_heads, d_model, seq_len,
        q_parts, node_k_rows, node_v_rows,
        kv_store, corpus, parent_caches,
        attn_outputs, per_node_attn_weights, this_depth_caches
      )
        # Build per_node_attn_weights with None placeholders so we can write
        # by index (groups process out of natural node order)
        n.times { per_node_attn_weights << Array(Mat).new(n_heads) }

        # Group node indices by parent id, preserving the iteration order
        groups = Hash(Int32, Array(Int32)).new
        n.times do |i|
          pid = corpus.parent_id(nodes[i].id)
          (groups[pid] ||= [] of Int32) << i
        end

        groups.each do |parent_id, idxs|
          # Resolve parent K/V cache (read-only — we do NOT extend it)
          parent_layer_cache = nil.as(LayerKVCache?)
          if pc = parent_caches
            if pcs = pc[parent_id]?
              parent_layer_cache = pcs[li]
            end
          end
          if parent_layer_cache.nil?
            # Fallback: reconstruct parent's cache via any child's node_id
            # (reconstruct_layer_cache builds the parent chain excluding the node itself)
            first_node_id = nodes[idxs[0]].id
            parent_layer_cache = kv_store.reconstruct_layer_cache(
              first_node_id, corpus, li, head_dims, seq_len
            )
          end
          parent_layer_cache = parent_layer_cache.not_nil!

          c = idxs.size
          prefix_len = parent_layer_cache.len

          # Per-head grouped attention
          col_offset = 0
          n_heads.times do |hi|
            hd = head_dims[hi]
            scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32

            # Gather Q for this group: [C, hd]
            q_group = Mat.new(c, hd)
            c.times do |k|
              row = idxs[k]
              hd.times { |j| q_group[k, j] = q_parts[hi][row, j] }
            end

            # Scores at prefix positions 0..prefix_len-1: [C, prefix_len]
            # Shared across siblings — one matmul
            scores = if prefix_len > 0
              k_prefix = parent_layer_cache.k_slice(hi)  # [prefix_len, hd]
              q_group * k_prefix.t                        # [C, prefix_len]
            else
              Mat.new(c, 0)
            end

            # Self scores at position prefix_len: one per sibling
            self_scores = Array(Float32).new(c)
            c.times do |k|
              row = idxs[k]
              k_self = node_k_rows[row][hi]  # [1, hd]
              dot = 0.0_f32
              hd.times { |j| dot += q_group[k, j] * k_self[0, j] }
              self_scores << dot
            end

            # Full scores: [C, prefix_len + 1], scaled
            full_len = prefix_len + 1
            full_scores = Mat.new(c, full_len)
            c.times do |k|
              prefix_len.times { |p| full_scores[k, p] = scores[k, p] * scale }
              full_scores[k, prefix_len] = self_scores[k] * scale
            end

            # Softmax per row → [C, full_len]
            weights = MicroGPT.backend.softmax_rows(full_scores)

            # Weighted sum for output: weights[:, 0:prefix_len] × V_prefix
            #   + weights[:, prefix_len] * V_self_i per sibling
            if prefix_len > 0
              v_prefix = parent_layer_cache.v_slice(hi)  # [prefix_len, hd]
              # split weights into prefix and self parts
              w_prefix = Mat.new(c, prefix_len)
              c.times do |k|
                prefix_len.times { |p| w_prefix[k, p] = weights[k, p] }
              end
              out_prefix = w_prefix * v_prefix  # [C, hd]

              c.times do |k|
                row = idxs[k]
                v_self = node_v_rows[row][hi]
                w_self = weights[k, prefix_len]
                hd.times do |j|
                  attn_outputs[row, col_offset + j] = out_prefix[k, j] + w_self * v_self[0, j]
                end
              end
            else
              # prefix_len == 0 (root's children): attention is just self
              c.times do |k|
                row = idxs[k]
                v_self = node_v_rows[row][hi]
                w_self = weights[k, 0]
                hd.times { |j| attn_outputs[row, col_offset + j] = w_self * v_self[0, j] }
              end
            end

            # Save per-sibling attention weights as [1, full_len] Mats
            c.times do |k|
              row = idxs[k]
              w_row = Mat.new(1, full_len)
              full_len.times { |p| w_row[0, p] = weights[k, p] }
              per_node_attn_weights[row] << w_row
            end

            col_offset += hd
          end

          # For each sibling, this_depth_caches[node_id] = parent_cache extended by this sibling's K/V.
          # Always deep_clone (even for unary c==1) so prev_caches and this_depth_caches are
          # fully disjoint — this lets the leveled trainer explicitly free prev_caches after
          # forward_depth returns without risk of use-after-free.
          # deep_clone now allocates only @len+1 rows (not seq_len), so unary chains
          # grow depth-proportional caches instead of fixed 128-row ones.
          c.times do |k|
            row = idxs[k]
            nid = nodes[row].id
            sibling_cache = parent_layer_cache.deep_clone
            sibling_cache.extend(node_k_rows[row], node_v_rows[row])
            this_depth_caches[nid] << sibling_cache
          end
        end
      end

      private def cpu_per_node_attention(
        nodes, n, li, head_dims, n_heads, d_model, seq_len,
        q_parts, node_k_rows, node_v_rows,
        kv_store, corpus, parent_caches,
        attn_outputs, per_node_attn_weights, this_depth_caches
      )
        n.times do |i|
          GC.collect if i > 0 && i % 500 == 0

          node_id = nodes[i].id
          parent_id = corpus.parent_id(node_id)

          layer_cache = if pc = parent_caches
            if parent_layer_caches = pc[parent_id]?
              parent_lc = parent_layer_caches[li]
              if corpus.child_count_of(parent_id) > 1
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

          layer_cache.extend(node_k_rows[i], node_v_rows[i])

          head_attn_weights = Array(Mat).new(n_heads)
          col_offset = 0
          n_heads.times do |hi|
            hd = head_dims[hi]
            q_h = Mat.new(1, hd)
            hd.times { |j| q_h[0, j] = q_parts[hi][i, j] }
            k_full = layer_cache.k_slice(hi)
            v_full = layer_cache.v_slice(hi)
            scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32
            scores = q_h * k_full.t
            scores.scale!(scale)
            weights = MicroGPT.backend.softmax_rows(scores)
            out = weights * v_full
            head_attn_weights << copy_mat(weights)
            hd.times { |j| attn_outputs[i, col_offset + j] = out[0, j] }
            col_offset += hd
          end

          per_node_attn_weights << head_attn_weights
          this_depth_caches[node_id] << layer_cache
        end
      end

      private def gpu_alloc_copy(data : Array(Float32)) : Float32*
        ptr = Pointer(Void).null
        LibCUDA.cudaMalloc(pointerof(ptr), data.size.to_u64 * sizeof(Float32))
        LibCUDA.cudaMemcpy(ptr, data.to_unsafe.as(Void*),
                           data.size.to_u64 * sizeof(Float32), 1) # cudaMemcpyHostToDevice=1
        ptr.as(Float32*)
      end

      private def gpu_alloc_copy_int(data : Array(Int32)) : Int32*
        ptr = Pointer(Void).null
        LibCUDA.cudaMalloc(pointerof(ptr), data.size.to_u64 * sizeof(Int32))
        LibCUDA.cudaMemcpy(ptr, data.to_unsafe.as(Void*),
                           data.size.to_u64 * sizeof(Int32), 1)
        ptr.as(Int32*)
      end

      private def gpu_alloc(n : Int32) : Float32*
        ptr = Pointer(Void).null
        LibCUDA.cudaMalloc(pointerof(ptr), n.to_u64 * sizeof(Float32))
        ptr.as(Float32*)
      end

      private def gpu_free(ptr : Float32*)
        LibCUDA.cudaFree(ptr.as(Void*))
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
