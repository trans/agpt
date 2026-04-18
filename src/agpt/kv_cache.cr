module MicroGPT
  module AGPT
    # Per-layer KV cache for incremental attention.
    #
    # Pre-allocates to max_len rows; `len` tracks how many positions are filled.
    # Supports extend (append one position) and truncate (backtrack to parent).
    class LayerKVCache
      getter k_parts : Array(Mat)   # per head: [max_len, head_dim]
      getter v_parts : Array(Mat)   # per head: [max_len, head_dim]
      getter head_dims : Array(Int32)
      property len : Int32 = 0

      def initialize(@head_dims : Array(Int32), max_len : Int32)
        @k_parts = @head_dims.map { |dim| Mat.new(max_len, dim) }
        @v_parts = @head_dims.map { |dim| Mat.new(max_len, dim) }
      end

      def extend(new_k : Array(Mat), new_v : Array(Mat))
        @head_dims.size.times do |i|
          dim = @head_dims[i]
          dim.times do |j|
            @k_parts[i][@len, j] = new_k[i][0, j]
            @v_parts[i][@len, j] = new_v[i][0, j]
          end
        end
        @len += 1
      end

      # Return K/V slices up to current length for head i.
      def k_slice(head : Int32) : Mat
        slice_rows(@k_parts[head], @len)
      end

      def v_slice(head : Int32) : Mat
        slice_rows(@v_parts[head], @len)
      end

      def truncate(new_len : Int32)
        @len = new_len
      end

      def deep_clone : LayerKVCache
        # Allocate to @len + 1 (room for one more extend), NOT the full pre-allocated
        # max_len. This keeps cache Mats proportional to actual depth rather than
        # seq_len, which is critical at large breadth (e.g. 50k nodes × 128 rows = 6 GB).
        max_len = [@len + 1, 1].max
        copy = LayerKVCache.new(@head_dims, max_len)
        @head_dims.size.times do |i|
          dim = @head_dims[i]
          @len.times do |row|
            dim.times do |j|
              copy.k_parts[i][row, j] = @k_parts[i][row, j]
              copy.v_parts[i][row, j] = @v_parts[i][row, j]
            end
          end
        end
        copy.len = @len
        copy
      end

      # Explicitly release backing Mat memory without waiting for GC.
      def free!
        @k_parts.each(&.free!)
        @v_parts.each(&.free!)
      end

      private def slice_rows(m : Mat, n : Int32) : Mat
        result = Mat.new(n, m.cols)
        n.times do |r|
          m.cols.times { |c| result[r, c] = m[r, c] }
        end
        result
      end
    end

    # Full model KV cache — one LayerKVCache per transformer block.
    class ModelKVCache
      getter layer_caches : Array(LayerKVCache)

      def initialize(n_layers : Int32, head_dims : Array(Int32), max_len : Int32)
        @layer_caches = Array.new(n_layers) { LayerKVCache.new(head_dims, max_len) }
      end

      def len : Int32
        @layer_caches.first.len
      end

      def truncate(new_len : Int32)
        @layer_caches.each &.truncate(new_len)
      end

      def deep_clone : ModelKVCache
        clone = ModelKVCache.new(0, [] of Int32, 0)
        clone.layer_caches.clear
        @layer_caches.each { |lc| clone.layer_caches << lc.deep_clone }
        clone
      end
    end
  end
end
