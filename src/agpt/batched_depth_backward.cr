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
      # forward_caches: if provided, reuses KV caches from the forward pass
      # instead of reconstructing from kv_store. Key = node_id → Array(LayerKVCache).
      def backward_depth(
        results : Array(BatchedDepthForward::NodeResult),
        loss_grads : Array(Mat),
        grad_accums : Hash(Int32, NodeGradAccum),
        kv_store : NodeKVStore,
        model : MiniGPT,
        corpus : TrieAccessor,
        forward_caches : Hash(Int32, Array(LayerKVCache))? = nil
      )
        MicroGPT::PerfTrace.with_scope("agpt.backward") do
          return if results.empty?

          n = results.size
          d_model = model.config.d_model
          vocab_size = model.config.vocab_size
          head_dims = model.blocks.first.attn.head_dims
          n_layers = model.config.n_layers
          n_heads = head_dims.size
          seq_len = model.config.seq_len

          # --- Batched output projection backward ---
          output_started = Time.instant if MicroGPT::PerfTrace.enabled?
          # Stack d_logits into [N, vocab_size]
          d_logits_batched = Mat.new(n, vocab_size)
          trace_sync_delta("agpt.backward.output_stack") do
            n.times do |i|
              vocab_size.times { |j| d_logits_batched[i, j] = loss_grads[i][0, j] }
            end
          end

          # Stack final_norm_out into [N, d_model] (input to output projection)
          final_norm_out_batched = Mat.new(n, d_model)
          trace_sync_delta("agpt.backward.output_norm_out") do
            n.times do |i|
              d_model.times { |j| final_norm_out_batched[i, j] = results[i].final_norm_out[0, j] }
            end
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
          MicroGPT::PerfTrace.add_time("agpt.backward.output", Time.instant - output_started.not_nil!) if output_started

          # --- Batched final norm backward ---
          final_norm_started = Time.instant if MicroGPT::PerfTrace.enabled?
          final_normed_batched = Mat.new(n, d_model)
          final_std_inv_batched = Mat.new(n, 1)
          trace_sync_delta("agpt.backward.final_norm_prep") do
            n.times do |i|
              d_model.times { |j| final_normed_batched[i, j] = results[i].final_normed[0, j] }
              final_std_inv_batched[i, 0] = results[i].final_std_inv.to_f32
            end
          end

          d_hidden = batched_ln_backward(
            d_hidden, final_normed_batched, final_std_inv_batched,
            model.final_norm.gamma, model.final_norm.dgamma, model.final_norm.dbeta
          )
          MicroGPT::PerfTrace.add_time("agpt.backward.final_norm", Time.instant - final_norm_started.not_nil!) if final_norm_started

          # --- Per-block backward (reverse order) ---
          model.blocks.reverse_each.with_index do |block, rev_idx|
          li = model.blocks.size - 1 - rev_idx
          attn = block.attn
          block_started = Time.instant if MicroGPT::PerfTrace.enabled?

          # Gather per-node block states for this layer into batched matrices
          gather_started = Time.instant if MicroGPT::PerfTrace.enabled?
          ln2_out_b = Mat.new(n, d_model)
          ln2_norm_b = Mat.new(n, d_model)
          ln2_sinv_b = Mat.new(n, 1)
          ff_relu_out_b = Mat.new(n, model.config.d_ff)
          ff_relu_mask_b = Mat.new(n, model.config.d_ff)
          wo_input_b = Mat.new(n, d_model)
          ln1_out_b = Mat.new(n, d_model)
          ln1_norm_b = Mat.new(n, d_model)
          ln1_sinv_b = Mat.new(n, 1)

          trace_sync_delta("agpt.backward.layer#{li}.gather_state") do
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
          end
          MicroGPT::PerfTrace.add_time("agpt.backward.layer#{li}.gather_state", Time.instant - gather_started.not_nil!) if gather_started

          # --- Residual 2 backward: d_ff_out = d_hidden ---
          ffn_started = Time.instant if MicroGPT::PerfTrace.enabled?
          d_ff_out = nil.as(Mat?)
          trace_sync_delta("agpt.backward.layer#{li}.residual2_copy") do
            d_ff_out = copy_mat(d_hidden)
          end
          d_ff_out = d_ff_out.not_nil!

          # Batched FFN L2 backward: dW += ff_relu_out^T × d_ff_out
          block.ff.l2.dw.add!(ff_relu_out_b.t * d_ff_out)
          n.times { |i| d_ff_out.cols.times { |j| block.ff.l2.db[0, j] += d_ff_out[i, j] } }
          d_ff_relu = d_ff_out * block.ff.l2.w.t  # [N, d_ff]

          # ReLU backward
          trace_sync_delta("agpt.backward.layer#{li}.relu_backward") do
            n.times { |i| model.config.d_ff.times { |j| d_ff_relu[i, j] *= ff_relu_mask_b[i, j] } }
          end

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
          MicroGPT::PerfTrace.add_time("agpt.backward.layer#{li}.ffn_ln2", Time.instant - ffn_started.not_nil!) if ffn_started

          # --- Residual 1 backward ---
          attn_started = Time.instant if MicroGPT::PerfTrace.enabled?
          d_attn_proj = nil.as(Mat?)
          trace_sync_delta("agpt.backward.layer#{li}.residual1_copy") do
            d_attn_proj = copy_mat(d_hidden)
          end
          d_attn_proj = d_attn_proj.not_nil!

          # Batched WO backward: dW += wo_input^T × d_attn_proj
          attn.wo.dw.add!(wo_input_b.t * d_attn_proj)
          n.times { |i| d_model.times { |j| attn.wo.db[0, j] += d_attn_proj[i, j] } }
          d_concat = d_attn_proj * attn.wo.w.t  # [N, d_model]

          # Split d_concat by heads: each [N, head_dim]
          d_head_outs = nil.as(Array(Mat)?)
          trace_sync_delta("agpt.backward.layer#{li}.split_heads") do
            d_head_outs = split_cols(d_concat, head_dims)
          end
          d_head_outs = d_head_outs.not_nil!

          # --- Per-node attention backward (different KV histories) ---
          dq_all = Mat.new(n, d_model)
          dk_current_all = Mat.new(n, d_model)
          dv_current_all = Mat.new(n, d_model)
          dq_all_data = dq_all.raw_data
          dk_current_all_data = dk_current_all.raw_data
          dv_current_all_data = dv_current_all.raw_data

          # Pre-allocated scratch buffers — reused across all N nodes × n_heads
          # to avoid per-call Array allocations (was ~10M allocs/epoch).
          max_head_dim = head_dims.max
          scratch_d_weights = Array(Float32).new(seq_len, 0.0_f32)
          scratch_d_scores = Array(Float32).new(seq_len, 0.0_f32)
          scratch_rope = Array(Float32).new(max_head_dim, 0.0_f32)

          # GPU backward path: pack inputs, launch kernel, download gradients,
          # CPU-side scatter + inverse RoPE. Fallback to per-node CPU path if
          # not enabled or not on CuBLAS backend.
          if ENV["AGPT_GPU_BACKWARD"]? == "1" && MicroGPT.backend.is_a?(MicroGPT::CuBLASBackend)
            gpu_attention_backward_depth(
              results, li, n, d_model, head_dims, n_heads, n_layers, seq_len,
              d_head_outs, kv_store, corpus, attn,
              dq_all_data, dk_current_all_data, dv_current_all_data,
              grad_accums, scratch_rope, forward_caches
            )
            MicroGPT::PerfTrace.add_time("agpt.backward.layer#{li}.attention", Time.instant - attn_started.not_nil!) if attn_started

            # Skip the CPU per-node loop below
            proj_started = Time.instant if MicroGPT::PerfTrace.enabled?
            attn.wq.dw.add!(ln1_out_b.t * dq_all)
            attn.wk.dw.add!(ln1_out_b.t * dk_current_all)
            attn.wv.dw.add!(ln1_out_b.t * dv_current_all)
            trace_sync_delta("agpt.backward.layer#{li}.qkv_bias") do
              n.times do |i|
                d_model.times do |j|
                  attn.wq.db[0, j] += dq_all[i, j]
                  attn.wk.db[0, j] += dk_current_all[i, j]
                  attn.wv.db[0, j] += dv_current_all[i, j]
                end
              end
            end
            d_ln1_out = dq_all * attn.wq.w.t
            d_ln1_out.add!(dk_current_all * attn.wk.w.t)
            d_ln1_out.add!(dv_current_all * attn.wv.w.t)
            d_ln1 = batched_ln_backward(
              d_ln1_out, ln1_norm_b, ln1_sinv_b,
              block.ln1.gamma, block.ln1.dgamma, block.ln1.dbeta
            )
            d_hidden.add!(d_ln1)
            MicroGPT::PerfTrace.add_time("agpt.backward.layer#{li}.qkv_ln1", Time.instant - proj_started.not_nil!) if proj_started
            MicroGPT::PerfTrace.add_time("agpt.backward.layer#{li}.total", Time.instant - block_started.not_nil!) if block_started
            next
          end

          n.times do |i|
            GC.collect if i > 0 && i % 2000 == 0

            result = results[i]
            bs = result.block_states[li]
            position = result.position

            accum = grad_accums[result.node_id]? || NodeGradAccum.new(n_layers, head_dims)

            # Reuse KV cache from forward pass if available (fast path),
            # otherwise reconstruct from kv_store (slow path).
            layer_cache = if fc = forward_caches
              if node_caches = fc[result.node_id]?
                node_caches[li]  # already includes this node's K/V
              else
                reconstruct_node_layer_cache(result.node_id, li, kv_store, corpus, head_dims, seq_len, n_heads)
              end
            else
              reconstruct_node_layer_cache(result.node_id, li, kv_store, corpus, head_dims, seq_len, n_heads)
            end

            col_offset = 0
            n_heads.times do |hi|
              hd = head_dims[hi]
              d_out_data = d_head_outs[hi].raw_data
              d_out_base = i * hd
              row_base = i * d_model + col_offset
              optimized_attention_backward_head(
                position: position,
                ancestor_ids: result.ancestor_ids,
                layer: li,
                head: hi,
                head_dims: head_dims,
                n_layers: n_layers,
                d_out_data: d_out_data,
                d_out_base: d_out_base,
                attn_weights: bs.attn_weights[hi],
                q_part: bs.q_parts[hi],
                layer_cache: layer_cache,
                accum: accum,
                grad_accums: grad_accums,
                rope: attn.ropes[hi],
                scratch_d_weights: scratch_d_weights,
                scratch_d_scores: scratch_d_scores,
                scratch_rope: scratch_rope,
                dq_all_data: dq_all_data,
                dk_current_all_data: dk_current_all_data,
                dv_current_all_data: dv_current_all_data,
                out_offset: row_base
              )
              col_offset += hd
            end

            # Free this node's grad accumulator
            grad_accums.delete(result.node_id)
          end
          MicroGPT::PerfTrace.add_time("agpt.backward.layer#{li}.attention", Time.instant - attn_started.not_nil!) if attn_started

          # Batched WQ/WK/WV backward: dW += ln1_out^T × dq/dk/dv
          proj_started = Time.instant if MicroGPT::PerfTrace.enabled?
          attn.wq.dw.add!(ln1_out_b.t * dq_all)
          attn.wk.dw.add!(ln1_out_b.t * dk_current_all)
          attn.wv.dw.add!(ln1_out_b.t * dv_current_all)
          trace_sync_delta("agpt.backward.layer#{li}.qkv_bias") do
            n.times do |i|
              d_model.times do |j|
                attn.wq.db[0, j] += dq_all[i, j]
                attn.wk.db[0, j] += dk_current_all[i, j]
                attn.wv.db[0, j] += dv_current_all[i, j]
              end
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
          MicroGPT::PerfTrace.add_time("agpt.backward.layer#{li}.qkv_ln1", Time.instant - proj_started.not_nil!) if proj_started
          MicroGPT::PerfTrace.add_time("agpt.backward.layer#{li}.total", Time.instant - block_started.not_nil!) if block_started
          end

          # --- Batched embedding gradient ---
          embedding_started = Time.instant if MicroGPT::PerfTrace.enabled?
          trace_sync_delta("agpt.backward.embedding") do
            n.times do |i|
              token = results[i].token_id
              d_model.times { |j| model.embedding.d_token_emb[token, j] += d_hidden[i, j] }
            end
          end
          MicroGPT::PerfTrace.add_time("agpt.backward.embedding", Time.instant - embedding_started.not_nil!) if embedding_started
        end
      end

      # --- Helpers ---

      # GPU-batched per-node attention backward for one layer.
      # Packs inputs (Q, K, V, attn_weights, d_out) into flat buffers matching
      # the forward kernel layout, launches cuda_batched_varlen_attention_backward,
      # downloads gradients, and performs CPU-side ancestor scatter + inverse RoPE.
      private def gpu_attention_backward_depth(
        results, li, n, d_model, head_dims, n_heads, n_layers, seq_len,
        d_head_outs, kv_store, corpus, attn,
        dq_all_data, dk_current_all_data, dv_current_all_data,
        grad_accums, scratch_rope,
        forward_caches : Hash(Int32, Array(LayerKVCache))? = nil
      )
        hd = head_dims[0]  # assumes uniform head dims (matches forward kernel)

        # Compute prefix lengths and max_len (same as forward)
        prefix_lens = Array(Int32).new(n)
        n.times { |i| prefix_lens << results[i].ancestor_ids.size }
        max_len = prefix_lens.max

        # Pack Q: [N * n_heads, hd]
        q_flat = Array(Float32).new(n * n_heads * hd, 0.0_f32)
        n.times do |i|
          bs = results[i].block_states[li]
          n_heads.times do |hi|
            base = (i * n_heads + hi) * hd
            q_src = bs.q_parts[hi].raw_data
            hd.times { |j| q_flat[base + j] = q_src[j] }
          end
        end

        # Pack attn_weights: [N * n_heads * max_len]
        w_flat = Array(Float32).new(n * n_heads * max_len, 0.0_f32)
        n.times do |i|
          bs = results[i].block_states[li]
          pl = prefix_lens[i]
          n_heads.times do |hi|
            w_src = bs.attn_weights[hi].raw_data
            w_base = (i * n_heads + hi) * max_len
            pl.times { |p| w_flat[w_base + p] = w_src[p] }
          end
        end

        # Pack d_out: [N * n_heads, hd] (from d_head_outs per head)
        d_out_flat = Array(Float32).new(n * n_heads * hd, 0.0_f32)
        n.times do |i|
          n_heads.times do |hi|
            d_src = d_head_outs[hi].raw_data
            base = (i * n_heads + hi) * hd
            hd.times { |j| d_out_flat[base + j] = d_src[i * hd + j] }
          end
        end

        # Pack K/V: for each node, collect its ancestor chain K/V (same as forward)
        kv_offsets = Array(Int32).new(n, 0)
        total_kv = 0
        n.times do |i|
          kv_offsets[i] = total_kv
          total_kv += prefix_lens[i]
        end
        k_flat = Array(Float32).new(total_kv * n_heads * hd, 0.0_f32)
        v_flat = Array(Float32).new(total_kv * n_heads * hd, 0.0_f32)
        n.times do |i|
          node_id = results[i].node_id
          offset = kv_offsets[i]
          pl = prefix_lens[i]

          # Fast path: read from forward_caches (the full accumulated LayerKVCache).
          # This is already assembled during forward so no ancestor walk needed.
          # Used by the leveled trainer (no_op kv_store) and whenever forward_caches
          # is available (which is always in the interleaved design).
          if fc = forward_caches
            if node_caches = fc[node_id]?
              layer_cache = node_caches[li]
              n_heads.times do |hi|
                k_mat = layer_cache.k_slice(hi)  # [pl, hd]
                v_mat = layer_cache.v_slice(hi)
                pl.times do |pos|
                  base = (offset + pos) * n_heads * hd + hi * hd
                  hd.times do |j|
                    k_flat[base + j] = k_mat[pos, j]
                    v_flat[base + j] = v_mat[pos, j]
                  end
                end
              end
              next
            end
          end

          # Fallback: reconstruct from kv_store by walking ancestor chain.
          # Used when forward_caches is nil (e.g. non-interleaved training).
          chain = [] of Int32
          current = node_id
          while current != -1
            chain << current
            current = corpus.parent_id(current)
          end
          chain.reverse!
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

        # Upload to GPU
        q_gpu = gpu_alloc_copy_local(q_flat)
        k_gpu = gpu_alloc_copy_local(k_flat)
        v_gpu = gpu_alloc_copy_local(v_flat)
        w_gpu = gpu_alloc_copy_local(w_flat)
        d_out_gpu = gpu_alloc_copy_local(d_out_flat)
        offsets_gpu = gpu_alloc_copy_int_local(kv_offsets)
        lengths_gpu = gpu_alloc_copy_int_local(prefix_lens)

        # Allocate output buffers on GPU
        dq_size = n * n_heads * hd
        dk_size = total_kv * n_heads * hd
        dv_size = total_kv * n_heads * hd
        dq_gpu = gpu_alloc_local(dq_size)
        dk_gpu = gpu_alloc_local(dk_size)
        dv_gpu = gpu_alloc_local(dv_size)

        scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32
        LibCUDAKernels.cuda_batched_varlen_attention_backward(
          q_gpu, k_gpu, v_gpu, w_gpu, d_out_gpu,
          offsets_gpu, lengths_gpu,
          dq_gpu, dk_gpu, dv_gpu,
          n, n_heads, hd, max_len, scale
        )
        LibCUDAKernels.cuda_sync

        # Download gradients
        dq_host = Array(Float32).new(dq_size, 0.0_f32)
        dk_host = Array(Float32).new(dk_size, 0.0_f32)
        dv_host = Array(Float32).new(dv_size, 0.0_f32)
        LibCUDA.cudaMemcpy(dq_host.to_unsafe.as(Void*), dq_gpu.as(Void*),
                           dq_size.to_u64 * sizeof(Float32), 2)
        LibCUDA.cudaMemcpy(dk_host.to_unsafe.as(Void*), dk_gpu.as(Void*),
                           dk_size.to_u64 * sizeof(Float32), 2)
        LibCUDA.cudaMemcpy(dv_host.to_unsafe.as(Void*), dv_gpu.as(Void*),
                           dv_size.to_u64 * sizeof(Float32), 2)

        # Free GPU buffers
        LibCUDA.cudaFree(q_gpu.as(Void*))
        LibCUDA.cudaFree(k_gpu.as(Void*))
        LibCUDA.cudaFree(v_gpu.as(Void*))
        LibCUDA.cudaFree(w_gpu.as(Void*))
        LibCUDA.cudaFree(d_out_gpu.as(Void*))
        LibCUDA.cudaFree(offsets_gpu.as(Void*))
        LibCUDA.cudaFree(lengths_gpu.as(Void*))
        LibCUDA.cudaFree(dq_gpu.as(Void*))
        LibCUDA.cudaFree(dk_gpu.as(Void*))
        LibCUDA.cudaFree(dv_gpu.as(Void*))

        # Write dq into dq_all_data, apply inverse RoPE
        n.times do |i|
          bs = results[i].block_states[li]
          position = results[i].position
          col_offset = 0
          n_heads.times do |hi|
            hd_h = head_dims[hi]
            dq_base = (i * n_heads + hi) * hd_h
            out_offset = i * d_model + col_offset
            hd_h.times { |j| dq_all_data[out_offset + j] = dq_host[dq_base + j] }
            apply_inverse_rope_slice!(dq_all_data, out_offset, hd_h, attn.ropes[hi], position, scratch_rope)
            col_offset += hd_h
          end

          # Last position (pos == prefix_len - 1) is THIS node's own K/V; write
          # its dk/dv into dk_current_all, dv_current_all + apply inverse RoPE
          # on dk.
          result = results[i]
          accum = grad_accums[result.node_id]? || NodeGradAccum.new(n_layers, head_dims)
          offset = kv_offsets[i]
          last_p = prefix_lens[i] - 1
          col_offset = 0
          n_heads.times do |hi|
            hd_h = head_dims[hi]
            kv_base = (offset + last_p) * n_heads * hd_h + hi * hd_h
            out_offset = i * d_model + col_offset
            accum_dk = accum.dk[li][hi].raw_data
            accum_dv = accum.dv[li][hi].raw_data
            hd_h.times do |j|
              dk_current_all_data[out_offset + j] = dk_host[kv_base + j] + accum_dk[j]
              dv_current_all_data[out_offset + j] = dv_host[kv_base + j] + accum_dv[j]
            end
            apply_inverse_rope_slice!(dk_current_all_data, out_offset, hd_h, attn.ropes[hi], position, scratch_rope)
            col_offset += hd_h
          end

          # Ancestor scatter: positions 0..last_p-1 contribute to ancestor grad_accums
          ancestor_ids = result.ancestor_ids
          (0...last_p).each do |pos|
            anc_id = ancestor_ids[pos]
            acc = grad_accums[anc_id]? || begin
              a = NodeGradAccum.new(n_layers, head_dims)
              grad_accums[anc_id] = a
              a
            end
            col_offset2 = 0
            n_heads.times do |hi|
              hd_h = head_dims[hi]
              kv_base = (offset + pos) * n_heads * hd_h + hi * hd_h
              acc_dk = acc.dk[li][hi].raw_data
              acc_dv = acc.dv[li][hi].raw_data
              hd_h.times do |j|
                acc_dk[j] += dk_host[kv_base + j]
                acc_dv[j] += dv_host[kv_base + j]
              end
              col_offset2 += hd_h
            end
          end

          grad_accums.delete(result.node_id)
        end
      end

      private def gpu_alloc_copy_local(data : Array(Float32)) : Float32*
        ptr = Pointer(Void).null
        LibCUDA.cudaMalloc(pointerof(ptr), data.size.to_u64 * sizeof(Float32))
        LibCUDA.cudaMemcpy(ptr, data.to_unsafe.as(Void*),
                           data.size.to_u64 * sizeof(Float32), 1)
        ptr.as(Float32*)
      end

      private def gpu_alloc_copy_int_local(data : Array(Int32)) : Int32*
        ptr = Pointer(Void).null
        LibCUDA.cudaMalloc(pointerof(ptr), data.size.to_u64 * sizeof(Int32))
        LibCUDA.cudaMemcpy(ptr, data.to_unsafe.as(Void*),
                           data.size.to_u64 * sizeof(Int32), 1)
        ptr.as(Int32*)
      end

      private def gpu_alloc_local(n : Int32) : Float32*
        ptr = Pointer(Void).null
        LibCUDA.cudaMalloc(pointerof(ptr), n.to_u64 * sizeof(Float32))
        ptr.as(Float32*)
      end

      private def reconstruct_node_layer_cache(node_id, li, kv_store, corpus, head_dims, seq_len, n_heads)
        layer_cache = kv_store.reconstruct_layer_cache(node_id, corpus, li, head_dims, seq_len)
        node_kv = kv_store.entries[node_id]
        k_parts_node = Array(Mat).new(n_heads)
        v_parts_node = Array(Mat).new(n_heads)
        n_heads.times do |hi|
          k_row, v_row = node_kv[li][hi]
          k_parts_node << k_row
          v_parts_node << v_row
        end
        layer_cache.extend(k_parts_node, v_parts_node)
        layer_cache
      end

      # Writes dq/dk_current/dv_current directly into pre-allocated output
      # buffers (dq_all_data, dk_current_all_data, dv_current_all_data) at the
      # given row offset. Uses caller-owned scratch buffers for d_weights,
      # d_scores to avoid per-call allocations.
      def optimized_attention_backward_head(
        position : Int32,
        ancestor_ids : Array(Int32),
        layer : Int32,
        head : Int32,
        head_dims : Array(Int32),
        n_layers : Int32,
        d_out_data : Array(Float32),
        d_out_base : Int32,
        attn_weights : Mat,
        q_part : Mat,
        layer_cache : LayerKVCache,
        accum : NodeGradAccum,
        grad_accums : Hash(Int32, NodeGradAccum),
        rope : RoPE,
        scratch_d_weights : Array(Float32),
        scratch_d_scores : Array(Float32),
        scratch_rope : Array(Float32),
        dq_all_data : Array(Float32),
        dk_current_all_data : Array(Float32),
        dv_current_all_data : Array(Float32),
        out_offset : Int32
      )
        hd = head_dims[head]
        prefix_len = layer_cache.len
        w_data = attn_weights.raw_data
        q_data = q_part.raw_data
        k_data = layer_cache.k_parts[head].raw_data
        v_data = layer_cache.v_parts[head].raw_data
        scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32
        dot = 0.0_f64

        # d_weights[pos] = dOut · V[pos]; accumulate dot = Σ w[pos]*d_weights[pos]
        prefix_len.times do |pos|
          base = pos * hd
          sum = 0.0_f32
          hd.times do |j|
            sum += d_out_data[d_out_base + j] * v_data[base + j]
          end
          scratch_d_weights[pos] = sum
          dot += sum * w_data[pos]
        end

        # d_scores[pos] = w[pos] * (d_weights[pos] - dot) * scale
        prefix_len.times do |pos|
          scratch_d_scores[pos] = (w_data[pos] * (scratch_d_weights[pos] - dot) * scale).to_f32
        end

        # dq = d_scores × K — write directly into dq_all_data at out_offset
        hd.times do |j|
          sum = 0.0_f32
          prefix_len.times do |pos|
            sum += scratch_d_scores[pos] * k_data[pos * hd + j]
          end
          dq_all_data[out_offset + j] = sum
        end

        last = prefix_len - 1
        accum_dk = accum.dk[layer][head].raw_data
        accum_dv = accum.dv[layer][head].raw_data

        # dk/dv per position: current node uses accum'd ancestor contributions;
        # ancestor positions scatter into their grad_accums
        prefix_len.times do |pos|
          score_grad = scratch_d_scores[pos]
          weight = w_data[pos]
          if pos == last
            hd.times do |j|
              dk_current_all_data[out_offset + j] = score_grad * q_data[j] + accum_dk[j]
              dv_current_all_data[out_offset + j] = weight * d_out_data[d_out_base + j] + accum_dv[j]
            end
          else
            ancestor_id = ancestor_ids[pos]
            acc = grad_accums[ancestor_id]? || begin
              a = NodeGradAccum.new(n_layers, head_dims)
              grad_accums[ancestor_id] = a
              a
            end
            acc_dk = acc.dk[layer][head].raw_data
            acc_dv = acc.dv[layer][head].raw_data
            hd.times do |j|
              acc_dk[j] += score_grad * q_data[j]
              acc_dv[j] += weight * d_out_data[d_out_base + j]
            end
          end
        end

        # Inverse RoPE on dq and dk_current, operating in-place on output buffers
        apply_inverse_rope_slice!(dq_all_data, out_offset, hd, rope, position, scratch_rope)
        apply_inverse_rope_slice!(dk_current_all_data, out_offset, hd, rope, position, scratch_rope)
      end

      # Applies inverse RoPE to a slice of a flat array (data[offset..offset+hd])
      # using scratch buffer to avoid allocation.
      private def apply_inverse_rope_slice!(
        data : Array(Float32), offset : Int32, hd : Int32,
        rope : RoPE, position : Int32, scratch : Array(Float32)
      )
        hd.times { |j| scratch[j] = data[offset + j] }
        apply_inverse_rope_values!(scratch, rope, position)
        hd.times { |j| data[offset + j] = scratch[j] }
      end

      private def copy_mat(m : Mat) : Mat
        result = Mat.new(m.rows, m.cols)
        m.rows.times { |r| m.cols.times { |c| result[r, c] = m[r, c] } }
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

      private def extract_row(m : Mat, r : Int32) : Mat
        result = Mat.new(1, m.cols)
        m.cols.times { |c| result[0, c] = m[r, c] }
        result
      end

      private def apply_inverse_rope_values!(values : Array(Float32), rope : RoPE, position : Int32)
        half = values.size // 2
        half.times do |i|
          c = rope.cos_cache[position, 2 * i]
          s = rope.sin_cache[position, 2 * i]
          x0 = values[2 * i]
          x1 = values[2 * i + 1]
          values[2 * i]     = x0 * c + x1 * s
          values[2 * i + 1] = -x0 * s + x1 * c
        end
      end

      # Batched layer norm backward: accumulates dgamma/dbeta across N rows.
      private def batched_ln_backward(
        grad : Mat, normed : Mat, std_inv : Mat,
        gamma : Mat, dgamma : Mat, dbeta : Mat
      ) : Mat
        if MicroGPT.backend.is_a?(MicroGPT::CuBLASBackend)
          dx = nil.as(Mat?)
          trace_sync_delta("agpt.backward.batched_ln_backward") do
            dx_tmp, dgamma_tmp, dbeta_tmp = MicroGPT.backend.layer_norm_backward(
              grad, normed, std_inv, gamma
            )
            dgamma.add!(dgamma_tmp)
            dbeta.add!(dbeta_tmp)
            dx = dx_tmp
          end
          return dx.not_nil!
        end

        n = grad.rows
        d = grad.cols
        dx = Mat.new(n, d)
        trace_sync_delta("agpt.backward.batched_ln_backward") do
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
