module MicroGPT
  module AGPT
    # Incremental single-token forward pass using KV caches.
    #
    # Reads model weights without modifying any backward state. The KV cache is
    # mutated in-place (one new position appended per call).
    #
    # Returns {logits, state} where state captures all intermediates needed for
    # single-step backward.
    module IncrementalForward
      extend self
      include MathUtils

      def forward_token(model : MiniGPT, token_id : Int32, position : Int32,
                        kv_cache : ModelKVCache,
                        ancestor_ids : Array(Int32) = [] of Int32) : {Mat, NodeForwardState}
        d_model = model.config.d_model

        # Embedding: gather single row
        x = Mat.new(1, d_model)
        d_model.times { |j| x[0, j] = model.embedding.token_emb[token_id, j] }

        block_states = Array(BlockStepState).new(model.blocks.size)

        model.blocks.each_with_index do |block, li|
          cache = kv_cache.layer_caches[li]
          attn = block.attn

          x_input = clone_row(x)

          # Pre-norm: LN1
          x_norm, ln1_std_inv = ln_row_with_state(x, block.ln1.gamma, block.ln1.beta)
          ln1_out = clone_row(x_norm)  # WITH gamma/beta — input to WQ/WK/WV
          ln1_normed = ln_normed_row(x, ln1_std_inv)  # WITHOUT gamma/beta — for LN backward

          # Q/K/V projections: [1, d_model] each
          q_all = matmul_bias_row(x_norm, attn.wq.w, attn.wq.b)
          k_new = matmul_bias_row(x_norm, attn.wk.w, attn.wk.b)
          v_new = matmul_bias_row(x_norm, attn.wv.w, attn.wv.b)

          # Split by heads
          q_parts = split_cols(q_all, attn.head_dims)
          k_new_parts = split_cols(k_new, attn.head_dims)
          v_new_parts = split_cols(v_new, attn.head_dims)

          # RoPE at the correct absolute position
          attn.ropes.each_with_index do |rope, hi|
            apply_rope_at!(q_parts[hi], rope, position)
            apply_rope_at!(k_new_parts[hi], rope, position)
          end

          # Extend KV cache with new K, V
          cache.extend(k_new_parts, v_new_parts)

          # Per-head attention: Q [1,hd] attends to all cached K [len,hd]
          head_outputs = Array(Mat).new(attn.heads.size)
          attn_weights_all = Array(Mat).new(attn.heads.size)
          attn.heads.size.times do |hi|
            k_full = cache.k_slice(hi)   # [len, head_dim]
            v_full = cache.v_slice(hi)   # [len, head_dim]
            head_dim = attn.head_dims[hi]

            scores = q_parts[hi] * k_full.t  # [1, len]
            scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
            scores.scale!(scale)
            weights = MicroGPT.backend.softmax_rows(scores)
            out = weights * v_full  # [1, head_dim]
            head_outputs << out
            attn_weights_all << clone_row(weights)
          end

          wo_input = concat_cols(head_outputs)
          attn_out = matmul_bias_row(wo_input, attn.wo.w, attn.wo.b)

          # Residual connection 1
          x.add!(attn_out)
          x_after_attn = clone_row(x)

          # Pre-norm: LN2
          x_norm2, ln2_std_inv = ln_row_with_state(x, block.ln2.gamma, block.ln2.beta)
          ln2_out = clone_row(x_norm2)  # WITH gamma/beta — input to FFN L1
          ln2_normed = ln_normed_row(x, ln2_std_inv)  # WITHOUT gamma/beta — for LN backward

          # FFN: L1 → ReLU → L2
          h = matmul_bias_row(x_norm2, block.ff.l1.w, block.ff.l1.b)
          ff_relu_mask = relu_mask_row(h)
          apply_relu_mask!(h, ff_relu_mask)
          ff_relu_out = clone_row(h)
          ff_out = matmul_bias_row(h, block.ff.l2.w, block.ff.l2.b)

          # Residual connection 2
          x.add!(ff_out)

          block_states << BlockStepState.new(
            x_input: x_input,
            ln1_out: ln1_out,
            ln1_normed: ln1_normed,
            ln1_std_inv: ln1_std_inv,
            q_parts: q_parts.map { |q| clone_row(q) },
            attn_weights: attn_weights_all,
            wo_input: clone_row(wo_input),
            x_after_attn: x_after_attn,
            ln2_out: ln2_out,
            ln2_normed: ln2_normed,
            ln2_std_inv: ln2_std_inv,
            ff_relu_out: ff_relu_out,
            ff_relu_mask: ff_relu_mask
          )
        end

        # Final norm
        final_x = clone_row(x)
        final_out, final_std_inv = ln_row_with_state(x, model.final_norm.gamma, model.final_norm.beta)
        final_normed = ln_normed_row(x, final_std_inv)
        final_norm_out = clone_row(final_out)

        # Output projection → logits [1, vocab_size]
        logits = matmul_bias_row(final_out, model.output.proj.w, model.output.proj.b)

        state = NodeForwardState.new(
          block_states: block_states,
          final_x: final_x,
          final_normed: final_normed,
          final_std_inv: final_std_inv,
          final_norm_out: final_norm_out,
          token_id: token_id,
          position: position,
          ancestor_ids: ancestor_ids
        )

        {logits, state}
      end

      # --- Helpers ---

      def matmul_bias_row(x : Mat, w : Mat, b : Mat) : Mat
        result = x * w  # [1, out_dim]
        b.cols.times { |j| result[0, j] += b[0, j] }
        result
      end

      # Match the backend's Float32 precision exactly.
      def ln_row_with_state(x : Mat, gamma : Mat, beta : Mat) : {Mat, Float64}
        d = x.cols
        mean = 0.0_f32
        d.times { |j| mean += x[0, j] }
        mean /= d

        var = 0.0_f32
        d.times do |j|
          diff = x[0, j] - mean
          var += diff * diff
        end
        var /= d

        std_inv_f32 = (1.0 / Math.sqrt(var + 1e-5_f32)).to_f32
        result = Mat.new(1, d)
        d.times do |j|
          normed = (x[0, j] - mean) * std_inv_f32
          result[0, j] = (normed * gamma[0, j] + beta[0, j]).to_f32
        end
        {result, std_inv_f32.to_f64}
      end

      # Return the normalized values (centered, scaled) WITHOUT gamma/beta.
      def ln_normed_row(x : Mat, std_inv : Float64) : Mat
        d = x.cols
        mean = 0.0_f32
        d.times { |j| mean += x[0, j] }
        mean /= d
        std_inv_f32 = std_inv.to_f32
        result = Mat.new(1, d)
        d.times { |j| result[0, j] = ((x[0, j] - mean) * std_inv_f32).to_f32 }
        result
      end

      def relu_mask_row(x : Mat) : Mat
        mask = Mat.new(1, x.cols)
        x.cols.times { |j| mask[0, j] = x[0, j] > 0 ? 1.0_f32 : 0.0_f32 }
        mask
      end

      def apply_relu_mask!(x : Mat, mask : Mat)
        x.cols.times { |j| x[0, j] *= mask[0, j] }
      end

      def clone_row(x : Mat) : Mat
        result = Mat.new(x.rows, x.cols)
        x.rows.times do |r|
          x.cols.times { |c| result[r, c] = x[r, c] }
        end
        result
      end

      def apply_rope_at!(x : Mat, rope : RoPE, position : Int32)
        half = rope.dim // 2
        half.times do |i|
          c = rope.cos_cache[position, 2 * i]
          s = rope.sin_cache[position, 2 * i]
          x0 = x[0, 2 * i]
          x1 = x[0, 2 * i + 1]
          x[0, 2 * i]     = x0 * c - x1 * s
          x[0, 2 * i + 1] = x0 * s + x1 * c
        end
      end

      def apply_inverse_rope_at!(x : Mat, rope : RoPE, position : Int32)
        half = rope.dim // 2
        half.times do |i|
          c = rope.cos_cache[position, 2 * i]
          s = rope.sin_cache[position, 2 * i]
          x0 = x[0, 2 * i]
          x1 = x[0, 2 * i + 1]
          x[0, 2 * i]     = x0 * c + x1 * s
          x[0, 2 * i + 1] = -x0 * s + x1 * c
        end
      end
    end
  end
end
