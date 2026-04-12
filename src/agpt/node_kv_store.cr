module MicroGPT
  module AGPT
    # Compact per-node storage of K/V contributions for memory-efficient
    # backward reconstruction.
    #
    # During forward, each node produces one K and one V row per layer per head
    # (post-RoPE). We store these ~1 KB per node instead of the full ~9 KB
    # NodeForwardState. During backward, we reconstruct the full KV cache for
    # any node by walking up the trie's parent chain and assembling ancestor
    # entries.
    class NodeKVStore
      # Flat storage: indexed by node_id.
      # Each entry: Array of {k_row, v_row} per layer, per head within layer.
      # k_row, v_row are [1, head_dim] Mat.
      getter entries : Hash(Int32, Array(Array({Mat, Mat})))

      def initialize
        @entries = {} of Int32 => Array(Array({Mat, Mat}))
      end

      # Store node's K/V contribution extracted from the KV cache after extend.
      def store(node_id : Int32, kv_cache : ModelKVCache, head_dims : Array(Int32))
        position = kv_cache.len - 1
        layers = Array(Array({Mat, Mat})).new(kv_cache.layer_caches.size)

        kv_cache.layer_caches.each do |layer_cache|
          heads = Array({Mat, Mat}).new(head_dims.size)
          head_dims.each_with_index do |hd, hi|
            k_row = Mat.new(1, hd)
            v_row = Mat.new(1, hd)
            hd.times do |j|
              k_row[0, j] = layer_cache.k_parts[hi][position, j]
              v_row[0, j] = layer_cache.v_parts[hi][position, j]
            end
            heads << {k_row, v_row}
          end
          layers << heads
        end

        @entries[node_id] = layers
      end

      # Reconstruct the KV cache containing all ANCESTOR entries (not the node
      # itself). The caller will run forward_token to extend by one, regenerating
      # the node's own K/V and capturing BlockStepState.
      def reconstruct_parent_cache(
        node_id : Int32,
        corpus : TrieCorpus,
        n_layers : Int32,
        head_dims : Array(Int32),
        seq_len : Int32
      ) : ModelKVCache
        # Collect ancestor chain: root → ... → parent (exclude node itself)
        chain = [] of Int32
        current = corpus.parent_id(node_id)
        while current != -1
          chain << current
          current = corpus.parent_id(current)
        end
        chain.reverse!

        # Skip root (id 0) — it has no K/V entry (root is the empty prefix)
        cache = ModelKVCache.new(n_layers, head_dims, seq_len)
        (1...chain.size).each do |i|
          ancestor_id = chain[i]
          entry = @entries[ancestor_id]
          n_layers.times do |li|
            k_parts = Array(Mat).new(head_dims.size)
            v_parts = Array(Mat).new(head_dims.size)
            head_dims.each_with_index do |_hd, hi|
              k_row, v_row = entry[li][hi]
              k_parts << k_row
              v_parts << v_row
            end
            cache.layer_caches[li].extend(k_parts, v_parts)
          end
        end

        cache
      end

      # Store K/V for a single layer (used by batched forward which processes
      # one layer at a time). Initializes the entry if needed.
      def store_layer(node_id : Int32, layer : Int32, k_parts : Array(Mat), v_parts : Array(Mat))
        entry = @entries[node_id]? || begin
          e = Array(Array({Mat, Mat})).new
          @entries[node_id] = e
          e
        end
        # Grow the layers array if needed
        while entry.size <= layer
          entry << [] of {Mat, Mat}
        end
        heads = Array({Mat, Mat}).new(k_parts.size)
        k_parts.each_with_index do |k, hi|
          k_copy = Mat.new(k.rows, k.cols)
          v_copy = Mat.new(v_parts[hi].rows, v_parts[hi].cols)
          k.rows.times { |r| k.cols.times { |c| k_copy[r, c] = k[r, c] } }
          v_parts[hi].rows.times { |r| v_parts[hi].cols.times { |c| v_copy[r, c] = v_parts[hi][r, c] } }
          heads << {k_copy, v_copy}
        end
        entry[layer] = heads
      end

      # Reconstruct a single-layer KV cache for a node's prefix (for per-node
      # attention within the batched forward). Returns a LayerKVCache.
      def reconstruct_layer_cache(
        node_id : Int32,
        corpus : TrieCorpus,
        layer : Int32,
        head_dims : Array(Int32),
        seq_len : Int32
      ) : LayerKVCache
        chain = [] of Int32
        current = corpus.parent_id(node_id)
        while current != -1
          chain << current
          current = corpus.parent_id(current)
        end
        chain.reverse!

        cache = LayerKVCache.new(head_dims, seq_len)
        (1...chain.size).each do |i|
          ancestor_id = chain[i]
          entry = @entries[ancestor_id]
          k_parts = Array(Mat).new(head_dims.size)
          v_parts = Array(Mat).new(head_dims.size)
          head_dims.each_with_index do |_hd, hi|
            k_row, v_row = entry[layer][hi]
            k_parts << k_row
            v_parts << v_row
          end
          cache.extend(k_parts, v_parts)
        end
        cache
      end

      def clear
        @entries.clear
      end

      def size : Int32
        @entries.size
      end
    end
  end
end
