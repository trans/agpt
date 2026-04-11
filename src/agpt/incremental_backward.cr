module MicroGPT
  module AGPT
    # Incremental single-step backward through one token position.
    #
    # Given a NodeForwardState (intermediates from forward), a loss gradient,
    # accumulated dK/dV from descendants, and the KV cache, this module:
    #
    # 1. Backprops through: output_proj → final_norm → blocks(reverse) → embedding
    # 2. Accumulates parameter gradients (dW, db) into the model's gradient buffers
    # 3. Computes dK/dV for all positions in the KV cache (cross-position gradients)
    # 4. Returns these dK/dV so the caller can scatter them to ancestor nodes
    #
    # The only cross-position gradient path in a transformer is through attention
    # dK/dV. There is no hidden-state recurrence between positions.
    module IncrementalBackward
      extend self
      include MathUtils

      # Per-layer dK/dV for ancestor positions (excluding current position, which
      # is handled locally).
      #
      # ancestor_dk[layer][head] = [prefix_len-1, head_dim] (rows for positions 0..prefix_len-2)
      # Empty if prefix_len == 1 (no ancestors).
      alias AncestorGrads = Array(Array({Mat, Mat}))  # [layer][head] → {dK, dV}

      # Run single-step backward for one node.
      #
      # d_logits: [1, vocab_size] — gradient from this node's loss (zero if no loss)
      # accum: NodeGradAccum — accumulated dK/dV from descendants at each layer/head
      # kv_cache: the KV cache state at this node (with len = position+1)
      #
      # Returns ancestor_grads for scattering to ancestor nodes.
      def backward_token(
        model : MiniGPT,
        state : NodeForwardState,
        d_logits : Mat,
        accum : NodeGradAccum,
        kv_cache : ModelKVCache
      ) : AncestorGrads
        position = state.position
        prefix_len = position + 1

        # Output projection backward
        # d_logits [1, vocab_size] → d_final_norm_out [1, d_model]
        proj = model.output.proj
        accum_linear_grads(proj, state.final_norm_out, d_logits)
        d_hidden = d_logits * proj.w.t  # [1, d_model]

        # Final norm backward
        d_hidden = ln_backward_row(
          d_hidden, state.final_normed, state.final_std_inv,
          model.final_norm.gamma, model.final_norm.dgamma, model.final_norm.dbeta
        )

        # Blocks in reverse order
        ancestor_grads = Array(Array({Mat, Mat})).new(model.blocks.size)

        model.blocks.reverse_each.with_index do |block, rev_idx|
          li = model.blocks.size - 1 - rev_idx
          bs = state.block_states[li]
          attn = block.attn
          cache = kv_cache.layer_caches[li]

          # --- Reverse of residual 2: out = a + ff_out ---
          d_ff_out = clone_mat(d_hidden)

          # FFN L2 backward (input to L2 = relu output)
          accum_linear_grads(block.ff.l2, bs.ff_relu_out, d_ff_out)
          d_ff_relu = d_ff_out * block.ff.l2.w.t  # [1, d_ff]

          # ReLU backward
          d_ff_relu.cols.times { |j| d_ff_relu[0, j] *= bs.ff_relu_mask[0, j] }

          # FFN L1 backward (input to L1 = LN2 output WITH gamma/beta)
          accum_linear_grads(block.ff.l1, bs.ln2_out, d_ff_relu)
          d_ln2_out = d_ff_relu * block.ff.l1.w.t  # [1, d_model]

          # LN2 backward
          d_ln2 = ln_backward_row(
            d_ln2_out, bs.ln2_normed, bs.ln2_std_inv,
            block.ln2.gamma, block.ln2.dgamma, block.ln2.dbeta
          )
          d_hidden.add!(d_ln2)  # residual 2

          # --- Reverse of residual 1: a = x + attn_out ---
          d_attn_proj = clone_mat(d_hidden)

          # WO backward
          accum_linear_grads(attn.wo, bs.wo_input, d_attn_proj)
          d_concat = d_attn_proj * attn.wo.w.t  # [1, d_model]

          # Split d_concat by heads
          d_head_outs = split_cols(d_concat, attn.head_dims)

          # Per-head attention backward + accumulate descendant dK/dV
          dq_parts = Array(Mat).new(attn.heads.size)
          dk_current_parts = Array(Mat).new(attn.heads.size)
          dv_current_parts = Array(Mat).new(attn.heads.size)
          layer_ancestor_grads = Array({Mat, Mat}).new(attn.heads.size)

          attn.heads.size.times do |hi|
            head_dim = attn.head_dims[hi]
            d_out_h = d_head_outs[hi]             # [1, head_dim]
            w_h = bs.attn_weights[hi]              # [1, prefix_len]
            q_h = bs.q_parts[hi]                   # [1, head_dim] post-RoPE
            k_h = cache.k_slice(hi)                # [prefix_len, head_dim]
            v_h = cache.v_slice(hi)                # [prefix_len, head_dim]

            # Attention backward
            dv_full = w_h.t * d_out_h              # [prefix_len, head_dim]
            d_weights = d_out_h * v_h.t            # [1, prefix_len]
            scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
            d_scores = softmax_backward_row(w_h, d_weights)
            d_scores.scale!(scale)
            dq_h = d_scores * k_h                  # [1, head_dim]
            dk_full = d_scores.t * q_h             # [prefix_len, head_dim]

            # Add accumulated dK/dV from descendants at current position (last row)
            last = prefix_len - 1
            head_dim.times do |j|
              dk_full[last, j] += accum.dk[li][hi][0, j]
              dv_full[last, j] += accum.dv[li][hi][0, j]
            end

            # Extract current position's dK/dV (with accum included)
            dk_current = extract_row(dk_full, last)
            dv_current = extract_row(dv_full, last)

            # Extract ancestor positions' dK/dV (for scattering)
            if prefix_len > 1
              dk_ancestors = extract_rows(dk_full, 0, prefix_len - 1)
              dv_ancestors = extract_rows(dv_full, 0, prefix_len - 1)
            else
              dk_ancestors = Mat.new(0, head_dim)
              dv_ancestors = Mat.new(0, head_dim)
            end

            # Inverse RoPE on dQ and dK_current (at this position)
            IncrementalForward.apply_inverse_rope_at!(dq_h, attn.ropes[hi], position)
            IncrementalForward.apply_inverse_rope_at!(dk_current, attn.ropes[hi], position)

            dq_parts << dq_h
            dk_current_parts << dk_current
            dv_current_parts << dv_current

            layer_ancestor_grads << {dk_ancestors, dv_ancestors}
          end

          ancestor_grads << layer_ancestor_grads

          # Concatenate per-head dQ/dK/dV into fused [1, d_model] vectors
          dq_all = concat_cols(dq_parts)
          dk_current_all = concat_cols(dk_current_parts)
          dv_current_all = concat_cols(dv_current_parts)

          # Accumulate WQ, WK, WV weight gradients (fused, not per-head)
          # Input to WQ/WK/WV is LN1 output (WITH gamma/beta)
          accum_linear_grads(attn.wq, bs.ln1_out, dq_all)
          accum_linear_grads(attn.wk, bs.ln1_out, dk_current_all)
          accum_linear_grads(attn.wv, bs.ln1_out, dv_current_all)

          # Backprop through WQ/WK/WV → d_ln1_out
          d_from_q = dq_all * attn.wq.w.t  # [1, d_model]
          d_from_k = dk_current_all * attn.wk.w.t
          d_from_v = dv_current_all * attn.wv.w.t

          d_ln1_out = d_from_q
          d_ln1_out.add!(d_from_k)
          d_ln1_out.add!(d_from_v)

          # LN1 backward
          d_ln1 = ln_backward_row(
            d_ln1_out, bs.ln1_normed, bs.ln1_std_inv,
            block.ln1.gamma, block.ln1.dgamma, block.ln1.dbeta
          )
          d_hidden.add!(d_ln1)  # residual 1
        end

        # Embedding gradient
        model.embedding.d_token_emb.cols.times do |j|
          model.embedding.d_token_emb[state.token_id, j] += d_hidden[0, j]
        end

        # Reorder ancestor_grads: currently in reverse block order, fix to forward order
        ancestor_grads.reverse!
        ancestor_grads
      end

      # --- Helpers ---

      private def accum_linear_grads(layer : Linear, input : Mat, grad : Mat)
        # dW += input^T × grad (outer product for single row)
        accum_outer(layer.dw, input, grad)
        accum_bias(layer.db, grad)
      end

      private def accum_outer(dw : Mat, input : Mat, grad : Mat)
        # input [1, d_in], grad [1, d_out] → dW [d_in, d_out] += input^T × grad
        d_in = input.cols
        d_out = grad.cols
        d_in.times do |i|
          d_out.times { |j| dw[i, j] += input[0, i] * grad[0, j] }
        end
      end

      private def accum_bias(db : Mat, grad : Mat)
        grad.cols.times { |j| db[0, j] += grad[0, j] }
      end

      private def ln_backward_row(grad : Mat, normed : Mat, std_inv : Float64,
                                   gamma : Mat, dgamma : Mat, dbeta : Mat) : Mat
        d = grad.cols
        # Accumulate gamma/beta gradients
        d.times do |j|
          dgamma[0, j] += grad[0, j] * normed[0, j]
          dbeta[0, j] += grad[0, j]
        end

        # Compute input gradient
        # d_normed = grad ⊙ gamma
        d_normed = Mat.new(1, d)
        d.times { |j| d_normed[0, j] = grad[0, j] * gamma[0, j] }

        # mean_dn = mean(d_normed)
        # mean_dn_n = mean(d_normed ⊙ normed)
        mean_dn = 0.0_f64
        mean_dn_n = 0.0_f64
        d.times do |j|
          mean_dn += d_normed[0, j]
          mean_dn_n += d_normed[0, j] * normed[0, j]
        end
        mean_dn /= d
        mean_dn_n /= d

        dx = Mat.new(1, d)
        d.times do |j|
          dx[0, j] = ((d_normed[0, j] - mean_dn - normed[0, j] * mean_dn_n) * std_inv).to_f32
        end
        dx
      end

      private def softmax_backward_row(s : Mat, ds : Mat) : Mat
        # s [1, cols], ds [1, cols] → d_scores [1, cols]
        cols = s.cols
        dot = 0.0_f64
        cols.times { |j| dot += ds[0, j] * s[0, j] }
        result = Mat.new(1, cols)
        cols.times { |j| result[0, j] = (s[0, j] * (ds[0, j] - dot)).to_f32 }
        result
      end

      private def clone_mat(x : Mat) : Mat
        IncrementalForward.clone_row(x)
      end

      private def extract_row(m : Mat, r : Int32) : Mat
        result = Mat.new(1, m.cols)
        m.cols.times { |c| result[0, c] = m[r, c] }
        result
      end

      private def extract_rows(m : Mat, from : Int32, count : Int32) : Mat
        result = Mat.new(count, m.cols)
        count.times do |r|
          m.cols.times { |c| result[r, c] = m[from + r, c] }
        end
        result
      end
    end
  end
end
