module MicroGPT
  module AGPT
    # BFS trie-walk trainer backed by a LeveledTrieReader (disk-paged corpus).
    #
    # Identical training loop to TrieWalkTrainer but drives depth iteration
    # from LeveledTrieReader.nodes_at_depth rather than TrieCorpus.each_depth_level.
    # Trie metadata (parent, depth, child_count, next-token counts) is loaded
    # lazily one depth at a time; only the LRU-bounded window stays in RAM.
    class LeveledTrieWalkTrainer
      getter reader : LeveledTrieReader
      getter loss_fn : WeightedNextTokenLoss
      getter observed_count : Int32
      property entropy_lambda : Float64 = 0.0

      def initialize(@reader : LeveledTrieReader, @loss_fn = WeightedNextTokenLoss.new)
        @observed_count = 0
        @reader.depth_file_count.times do |d|
          next if d == 0
          @reader.nodes_at_depth(d).each do |rec|
            @observed_count += 1 unless rec.depth == 0
          end
        end
      end

      # Maximum nodes processed in one forward→backward→update pass within a depth.
      # Caps peak block_state memory: 10k nodes × 2L × ~4.2 KB = ~84 MB regardless
      # of trie breadth. Raise if you have more RAM; lower if you OOM.
      DEPTH_CHUNK_SIZE = 10_000

      def train_epoch(model : MiniGPT) : {Float64, Int32}
        epoch_started = Time.instant if MicroGPT::PerfTrace.enabled?
        seq_len   = model.config.seq_len
        head_dims = model.blocks.first.attn.head_dims
        n_layers  = model.config.n_layers

        total_loss    = 0.0
        nodes_trained = 0

        # No-op store: forward_caches covers all backward KV lookups, and
        # prev_caches covers all forward parent lookups, so the store is never
        # read in the CPU-backend interleaved design. Skipping writes saves
        # O(total_nodes × kv_row_size) RAM — the main Phase B memory win.
        kv_store = NodeKVStore.new(no_op: true)

        node_ancestor_ids = {} of Int32 => Array(Int32)
        node_positions    = {} of Int32 => Int32
        node_root_child   = {} of Int32 => Int32

        # Root (id 0) always has an empty ancestor list
        node_ancestor_ids[0] = [] of Int32

        prev_caches : Hash(Int32, Array(AGPT::LayerKVCache))? = nil

        @reader.depth_file_count.times do |depth|
          next if depth == 0

          # Collect previous depth's NodeResult Mats (block_states, logits, etc.)
          # before allocating the next depth. Without this, GC pressure from
          # ~50k unfreed Mats per depth accumulates to several GiB at peak breadth.
          GC.collect

          depth_started = Time.instant if MicroGPT::PerfTrace.enabled?

          raw_nodes = @reader.nodes_at_depth(depth)
          eligible = Array(BatchedDepthForward::NodeProxy).new

          raw_nodes.each do |rec|
            parent_id = rec.parent_id
            next unless node_ancestor_ids.has_key?(parent_id)
            next if @reader.depth_of(parent_id) >= seq_len
            eligible << BatchedDepthForward::NodeProxy.new(rec.id, rec.token, rec.depth)
          end
          next if eligible.empty?

          eligible.each do |node|
            node_positions[node.id] = depth - 1
            if depth == 1
              node_root_child[node.id] = node.id
            else
              node_root_child[node.id] = node_root_child[@reader.parent_id(node.id)]
            end
          end

          # Accumulate KV caches across chunks — all chunks at this depth share
          # prev_caches (from depth d-1) and contribute to this_caches (for depth d+1).
          accumulated_caches = {} of Int32 => Array(AGPT::LayerKVCache)

          # Process eligible nodes in memory-bounded chunks. Each chunk does a
          # full forward→loss→backward→update cycle so block_states are freed
          # before the next chunk is allocated.
          eligible.each_slice(DEPTH_CHUNK_SIZE) do |chunk|
            forward_started = Time.instant if MicroGPT::PerfTrace.enabled?
            results, chunk_caches = BatchedDepthForward.forward_depth(
              chunk, node_ancestor_ids, node_positions, kv_store, model, @reader, prev_caches
            )
            # Merge chunk caches into the accumulator for this depth.
            chunk_caches.each { |nid, caches| accumulated_caches[nid] = caches }
            MicroGPT::PerfTrace.observe_max("agpt.forward_stage_bytes", Mat.allocated_bytes)
            MicroGPT::PerfTrace.add_time("agpt.epoch.forward", Time.instant - forward_started.not_nil!) if forward_started

            loss_started = Time.instant if MicroGPT::PerfTrace.enabled?
            loss_grads  = {} of Int32 => Mat
            result_map  = {} of Int32 => BatchedDepthForward::NodeResult
            MicroGPT::PerfTrace.with_scope("agpt.loss") do
              n_results  = results.size
              vocab_size = model.config.vocab_size

              logits_batched = Mat.new(n_results, vocab_size)
              n_results.times do |i|
                vocab_size.times { |j| logits_batched[i, j] = results[i].logits[0, j] }
              end
              probs_batched = MicroGPT.backend.softmax_rows(logits_batched)
              all_probs = probs_batched.data

              log_vocab = Math.log(vocab_size.to_f64)
              lambda    = @entropy_lambda
              results.each_with_index do |result, i|
                result_map[result.node_id] = result
                counts_arr = @reader.counts_of(result.node_id)
                unless counts_arr.empty?
                  total   = counts_arr.sum(0) { |t| t[1] }
                  total_f = total.to_f64

                  entropy = 0.0
                  if lambda > 0.0 && counts_arr.size > 1
                    counts_arr.each do |_tok, count|
                      q = count / total_f
                      entropy -= q * Math.log(q) if q > 0.0
                    end
                  end
                  weight = (lambda > 0.0) ? 1.0 + lambda * (entropy / log_vocab) : 1.0

                  loss_value  = 0.0
                  prob_offset = i * vocab_size
                  counts_arr.each do |token_id, count|
                    loss_value -= count * Math.log(all_probs[prob_offset + token_id] + 1e-10)
                  end
                  loss_value /= total
                  loss_value *= weight

                  grad       = Mat.new(1, vocab_size)
                  weight_f32 = weight.to_f32
                  vocab_size.times { |j| grad[0, j] = all_probs[prob_offset + j] * weight_f32 }
                  counts_arr.each do |token_id, count|
                    grad[0, token_id] -= (count.to_f32 / total) * weight_f32
                  end

                  loss_grads[result.node_id] = grad
                  total_loss    += loss_value
                  nodes_trained += 1
                end
              end
            end
            MicroGPT::PerfTrace.add_time("agpt.epoch.loss", Time.instant - loss_started.not_nil!) if loss_started

            partition_started = Time.instant if MicroGPT::PerfTrace.enabled?
            subtries = {} of Int32 => Array(BatchedDepthForward::NodeResult)
            if ENV["AGPT_DEPTH_BATCHED"]? == "1"
              all_at_depth = [] of BatchedDepthForward::NodeResult
              chunk.each { |node| all_at_depth << result_map[node.id] }
              subtries[-1] = all_at_depth
            else
              chunk.each do |node|
                root_id = node_root_child[node.id]
                (subtries[root_id] ||= [] of BatchedDepthForward::NodeResult) << result_map[node.id]
              end
            end
            if partition_started
              MicroGPT::PerfTrace.add_time("agpt.epoch.partition", Time.instant - partition_started.not_nil!)
              MicroGPT::PerfTrace.increment("agpt.epoch.subtries", subtries.size.to_i64)
            end

            subtries.each do |_root_id, subtrie_results|
              backward_started = Time.instant if MicroGPT::PerfTrace.enabled?
              MicroGPT::PerfTrace.with_scope("agpt.zero_gradients") do
                zero_gradients(model)
              end
              grad_accums = {} of Int32 => NodeGradAccum

              subtrie_grads = subtrie_results.map do |result|
                if d_logits = loss_grads.delete(result.node_id)
                  d_logits
                else
                  Mat.new(1, model.config.vocab_size)
                end
              end

              BatchedDepthBackward.backward_depth(
                subtrie_results, subtrie_grads, grad_accums, kv_store, model, @reader, chunk_caches
              )
              # Free BlockStepState Mats now that backward is done with them.
              subtrie_results.each { |r| r.block_states.each(&.free!) }
              MicroGPT::PerfTrace.observe_max("agpt.backward_stage_bytes", Mat.allocated_bytes)
              MicroGPT::PerfTrace.add_time("agpt.epoch.backward", Time.instant - backward_started.not_nil!) if backward_started

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
                MicroGPT::PerfTrace.add_time("agpt.epoch.update", Time.instant - update_started.not_nil!) if update_started
              end
            end
          end  # end each_slice chunk

          # Replace prev_caches with the full accumulated caches for this depth,
          # freeing the old ones explicitly.
          this_caches = accumulated_caches
          if old = prev_caches
            old.each_value { |layer_caches| layer_caches.each(&.free!) }
          end
          prev_caches = this_caches

          MicroGPT::PerfTrace.add_time("agpt.epoch.depth_total", Time.instant - depth_started.not_nil!) if depth_started
        end

        mean_loss = nodes_trained > 0 ? total_loss / nodes_trained : 0.0
        MicroGPT::PerfTrace.add_time("agpt.epoch.total", Time.instant - epoch_started.not_nil!) if epoch_started
        {mean_loss, nodes_trained}
      end

      private def scale_gradients(model : MiniGPT, scale : Float64)
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
