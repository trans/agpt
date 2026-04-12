module MicroGPT
  module AGPT
    # Batched backward pass for all nodes at a single depth level.
    #
    # Linear operations (output projection, LN, Q/K/V projection gradients, FFN)
    # are batched into single [N, d] matmuls. Attention backward is per-node
    # (each node has different KV history). Gradient accumulation into model
    # weight buffers (dW, db) happens via batched outer products.
    module BatchedDepthBackward
      extend self
      include MathUtils

      # Backward for one depth level's worth of nodes.
      #
      # results:     NodeResults from BatchedDepthForward
      # loss_grads:  per-node d_logits [1, vocab_size] (zero for unobserved nodes)
      # grad_accums: accumulated dK/dV from deeper levels (mutated: new ancestor grads scattered)
      # kv_store:    for reconstructing per-node KV caches for attention backward
      # model:       the MiniGPT model (weight gradients accumulated in place)
      def backward_depth(
        results : Array(BatchedDepthForward::NodeResult),
        loss_grads : Array(Mat),
        grad_accums : Hash(Int32, NodeGradAccum),
        kv_store : NodeKVStore,
        model : MiniGPT,
        corpus : TrieCorpus
      )
        return if results.empty?

        n = results.size
        d_model = model.config.d_model
        vocab_size = model.config.vocab_size
        head_dims = model.blocks.first.attn.head_dims
        n_layers = model.config.n_layers
        n_heads = head_dims.size
        seq_len = model.config.seq_len

        # --- Batched output projection backward ---
        # Stack d_logits into [N, vocab_size]
        d_logits_batched = Mat.new(n, vocab_size)
        n.times do |i|
          vocab_size.times { |j| d_logits_batched[i, j] = loss_grads[i][0, j] }
        end

        # Stack final_norm_out into [N, d_model] (input to output projection)
        final_norm_out_batched = Mat.new(n, d_model)
        n.times do |i|
          d_model.times { |j| final_norm_out_batched[i, j] = results[i].final_norm_out[0, j] }
        end

        # dW_proj += final_norm_out^T × d_logits  [d_model, vocab_size]
        proj = model.output.proj
        proj.dw.add!(final_norm_out_batched.t * d_logits_batched)
        # db_proj += sum of d_logits rows
        n.times do |i|
          vocab_size.times { |j| proj.db[0, j] += d_logits_batched[i, j] }
        end

        # d_hidden = d_logits × W_proj^T  [N, d_model]
        d_hidden = d_logits_batched * proj.w.t

        # --- Batched final norm backward ---
        final_normed_batched = Mat.new(n, d_model)
        final_std_inv_batched = Mat.new(n, 1)
        n.times do |i|
          d_model.times { |j| final_normed_batched[i, j] = results[i].final_normed[0, j] }
          final_std_inv_batched[i, 0] = results[i].final_std_inv.to_f32
        end

        d_hidden = batched_ln_backward(
          d_hidden, final_normed_batched, final_std_inv_batched,
          model.final_norm.gamma, model.final_norm.dgamma, model.final_norm.dbeta
        )

        # --- Per-block backward (reverse order) ---
        model.blocks.reverse_each.with_index do |block, rev_idx|
          li = model.blocks.size - 1 - rev_idx
          attn = block.attn

          # Gather per-node block states for this layer into batched matrices
          ln2_out_b = Mat.new(n, d_model)
          ln2_norm_b = Mat.new(n, d_model)
          ln2_sinv_b = Mat.new(n, 1)
          ff_relu_out_b = Mat.new(n, model.config.d_ff)
          ff_relu_mask_b = Mat.new(n, model.config.d_ff)
          wo_input_b = Mat.new(n, d_model)
          ln1_out_b = Mat.new(n, d_model)
          ln1_norm_b = Mat.new(n, d_model)
          ln1_sinv_b = Mat.new(n, 1)

          n.times do |i|
            bs = results[i].block_states[li]
            d_model.times do |j|
              ln2_out_b[i, j] = bs.ln2_out[0, j]
              ln2_norm_b[i, j] = bs.ln2_normed[0, j]
              wo_input_b[i, j] = bs.wo_input[0, j]
              ln1_out_b[i, j] = bs.ln1_out[0, j]
              ln1_norm_b[i, j] = bs.ln1_normed[0, j]
            end
            ln2_sinv_b[i, 0] = bs.ln2_std_inv.to_f32
            ln1_sinv_b[i, 0] = bs.ln1_std_inv.to_f32
            model.config.d_ff.times do |j|
              ff_relu_out_b[i, j] = bs.ff_relu_out[0, j]
              ff_relu_mask_b[i, j] = bs.ff_relu_mask[0, j]
            end
          end

          # --- Residual 2 backward: d_ff_out = d_hidden ---
          d_ff_out = copy_mat(d_hidden)

          # Batched FFN L2 backward: dW += ff_relu_out^T × d_ff_out
          block.ff.l2.dw.add!(ff_relu_out_b.t * d_ff_out)
          n.times { |i| d_ff_out.cols.times { |j| block.ff.l2.db[0, j] += d_ff_out[i, j] } }
          d_ff_relu = d_ff_out * block.ff.l2.w.t  # [N, d_ff]

          # ReLU backward
          n.times { |i| model.config.d_ff.times { |j| d_ff_relu[i, j] *= ff_relu_mask_b[i, j] } }

          # Batched FFN L1 backward: dW += ln2_out^T × d_ff_relu
          block.ff.l1.dw.add!(ln2_out_b.t * d_ff_relu)
          n.times { |i| model.config.d_ff.times { |j| block.ff.l1.db[0, j] += d_ff_relu[i, j] } }
          d_ln2_out = d_ff_relu * block.ff.l1.w.t  # [N, d_model]

          # Batched LN2 backward
          d_ln2 = batched_ln_backward(
            d_ln2_out, ln2_norm_b, ln2_sinv_b,
            block.ln2.gamma, block.ln2.dgamma, block.ln2.dbeta
          )
          d_hidden.add!(d_ln2)  # residual 2

          # --- Residual 1 backward ---
          d_attn_proj = copy_mat(d_hidden)

          # Batched WO backward: dW += wo_input^T × d_attn_proj
          attn.wo.dw.add!(wo_input_b.t * d_attn_proj)
          n.times { |i| d_model.times { |j| attn.wo.db[0, j] += d_attn_proj[i, j] } }
          d_concat = d_attn_proj * attn.wo.w.t  # [N, d_model]

          # Split d_concat by heads: each [N, head_dim]
          d_head_outs = split_cols(d_concat, head_dims)

          # --- Per-node attention backward (different KV histories) ---
          dq_all = Mat.new(n, d_model)
          dk_current_all = Mat.new(n, d_model)
          dv_current_all = Mat.new(n, d_model)

          n.times do |i|
            GC.collect if i > 0 && i % 500 == 0  # free temporary KV slices

            result = results[i]
            bs = result.block_states[li]
            position = result.position
            prefix_len = position + 1

            accum = grad_accums[result.node_id]? || NodeGradAccum.new(n_layers, head_dims)

            # Reconstruct single-layer KV cache for this node
            layer_cache = kv_store.reconstruct_layer_cache(
              result.node_id, corpus, li, head_dims, seq_len
            )
            # Extend with this node's own K/V (stored in kv_store)
            node_kv = kv_store.entries[result.node_id]
            k_parts_node = Array(Mat).new(n_heads)
            v_parts_node = Array(Mat).new(n_heads)
            n_heads.times do |hi|
              k_row, v_row = node_kv[li][hi]
              k_parts_node << k_row
              v_parts_node << v_row
            end
            layer_cache.extend(k_parts_node, v_parts_node)

            col_offset = 0
            n_heads.times do |hi|
              hd = head_dims[hi]
              d_out_h = Mat.new(1, hd)
              hd.times { |j| d_out_h[0, j] = d_head_outs[hi][i, j] }

              w_h = bs.attn_weights[hi]              # [1, prefix_len]
              q_h = bs.q_parts[hi]                   # [1, head_dim] post-RoPE
              k_h = layer_cache.k_slice(hi)          # [prefix_len, head_dim]
              v_h = layer_cache.v_slice(hi)          # [prefix_len, head_dim]

              dv_full = w_h.t * d_out_h              # [prefix_len, head_dim]
              d_weights = d_out_h * v_h.t            # [1, prefix_len]
              scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32
              d_scores = softmax_backward_row(w_h, d_weights)
              d_scores.scale!(scale)
              dq_h = d_scores * k_h                  # [1, head_dim]
              dk_full = d_scores.t * q_h             # [prefix_len, head_dim]

              # Add accumulated dK/dV from descendants at current position
              last = prefix_len - 1
              hd.times do |j|
                dk_full[last, j] += accum.dk[li][hi][0, j]
                dv_full[last, j] += accum.dv[li][hi][0, j]
              end

              dk_current = extract_row(dk_full, last)
              dv_current = extract_row(dv_full, last)

              # Scatter ancestor dK/dV
              if prefix_len > 1
                (prefix_len - 1).times do |pos|
                  ancestor_id = result.ancestor_ids[pos]
                  acc = grad_accums[ancestor_id]? || begin
                    a = NodeGradAccum.new(n_layers, head_dims)
                    grad_accums[ancestor_id] = a
                    a
                  end
                  row_dk = Mat.new(1, hd)
                  row_dv = Mat.new(1, hd)
                  hd.times do |j|
                    row_dk[0, j] = dk_full[pos, j]
                    row_dv[0, j] = dv_full[pos, j]
                  end
                  acc.add_dk(li, hi, row_dk)
                  acc.add_dv(li, hi, row_dv)
                end
              end

              # Inverse RoPE on dQ and dK_current
              IncrementalForward.apply_inverse_rope_at!(dq_h, attn.ropes[hi], position)
              IncrementalForward.apply_inverse_rope_at!(dk_current, attn.ropes[hi], position)

              # Write back to batched matrices
              hd.times do |j|
                dq_all[i, col_offset + j] = dq_h[0, j]
                dk_current_all[i, col_offset + j] = dk_current[0, j]
                dv_current_all[i, col_offset + j] = dv_current[0, j]
              end
              col_offset += hd
            end

            # Free this node's grad accumulator
            grad_accums.delete(result.node_id)
          end

          # Batched WQ/WK/WV backward: dW += ln1_out^T × dq/dk/dv
          attn.wq.dw.add!(ln1_out_b.t * dq_all)
          attn.wk.dw.add!(ln1_out_b.t * dk_current_all)
          attn.wv.dw.add!(ln1_out_b.t * dv_current_all)
          n.times do |i|
            d_model.times do |j|
              attn.wq.db[0, j] += dq_all[i, j]
              attn.wk.db[0, j] += dk_current_all[i, j]
              attn.wv.db[0, j] += dv_current_all[i, j]
            end
          end

          # d_ln1_out = dq × Wq^T + dk × Wk^T + dv × Wv^T  [N, d_model]
          d_ln1_out = dq_all * attn.wq.w.t
          d_ln1_out.add!(dk_current_all * attn.wk.w.t)
          d_ln1_out.add!(dv_current_all * attn.wv.w.t)

          # Batched LN1 backward
          d_ln1 = batched_ln_backward(
            d_ln1_out, ln1_norm_b, ln1_sinv_b,
            block.ln1.gamma, block.ln1.dgamma, block.ln1.dbeta
          )
          d_hidden.add!(d_ln1)  # residual 1
        end

        # --- Batched embedding gradient ---
        n.times do |i|
          token = results[i].token_id
          d_model.times { |j| model.embedding.d_token_emb[token, j] += d_hidden[i, j] }
        end
      end

      # --- Helpers ---

      private def copy_mat(m : Mat) : Mat
        result = Mat.new(m.rows, m.cols)
        m.rows.times { |r| m.cols.times { |c| result[r, c] = m[r, c] } }
        result
      end

      private def extract_row(m : Mat, r : Int32) : Mat
        result = Mat.new(1, m.cols)
        m.cols.times { |c| result[0, c] = m[r, c] }
        result
      end

      # Batched layer norm backward: accumulates dgamma/dbeta across N rows.
      private def batched_ln_backward(
        grad : Mat, normed : Mat, std_inv : Mat,
        gamma : Mat, dgamma : Mat, dbeta : Mat
      ) : Mat
        n = grad.rows
        d = grad.cols
        dx = Mat.new(n, d)

        n.times do |i|
          sinv = std_inv[i, 0].to_f64
          # Accumulate gamma/beta gradients
          d.times do |j|
            dgamma[0, j] += grad[i, j] * normed[i, j]
            dbeta[0, j] += grad[i, j]
          end

          # Per-row input gradient
          mean_dn = 0.0_f64
          mean_dn_n = 0.0_f64
          d.times do |j|
            dn = grad[i, j] * gamma[0, j]
            mean_dn += dn
            mean_dn_n += dn * normed[i, j]
          end
          mean_dn /= d
          mean_dn_n /= d

          d.times do |j|
            dn = grad[i, j] * gamma[0, j]
            dx[i, j] = ((dn - mean_dn - normed[i, j] * mean_dn_n) * sinv).to_f32
          end
        end

        dx
      end

      private def softmax_backward_row(s : Mat, ds : Mat) : Mat
        cols = s.cols
        dot = 0.0_f64
        cols.times { |j| dot += ds[0, j] * s[0, j] }
        result = Mat.new(1, cols)
        cols.times { |j| result[0, j] = (s[0, j] * (ds[0, j] - dot)).to_f32 }
        result
      end
    end
  end
end
