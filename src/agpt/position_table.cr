# Per-substring position distribution table.
#
# For each substring (indexed by substring_id from SubstringCatalog), stores
# a sparse histogram of the W-window positions where the substring occurs.
# The position bin is `corpus_position % W` where `corpus_position` is the
# start of the substring's occurrence in the corpus.
#
# On-disk format (binary, little-endian):
#   magic           u32   "APOS" (0x534F5041 little-endian)
#   regime          u8    0=aligned, 1=sliding
#   window_size     u16   W
#   reserved        u8    0  (padding to align to 4 bytes)
#   substring_count u32   number of substrings indexed (= catalog size)
#   total_bins      u64   total nonzero (substring_id, pos_bin) pairs
#   pos_offsets     i32[substring_count + 1]  start offsets into pos_bins
#   pos_bins        PosBin[total_bins]
#       PosBin { pos: u16; count: u32 }
#
# Lookup: bins for substring_id are pos_bins[pos_offsets[id] .. pos_offsets[id+1]].
#
# Storage estimate on Gutenberg 5M with W=64, sliding regime:
#   ~7M substrings × ~10 nonzero bins/substring × 6 bytes/bin ≈ 420 MB,
#   plus ~28 MB of pos_offsets.

module MicroGPT::AGPT
  class PositionTable
    MAGIC = "APOS"

    enum Regime : UInt8
      Aligned = 0
      Sliding = 1
    end

    struct PosBin
      getter pos : UInt16
      getter count : UInt32

      def initialize(@pos, @count)
      end
    end

    getter window_size : Int32
    getter regime : Regime
    getter substring_count : Int32

    # Sparse storage: per-substring (pos, count) pairs in one flat array.
    @pos_offsets : Slice(Int32)  # [substring_count + 1]
    @pos_bins : Slice(PosBin)    # [total_bins]

    def initialize(@window_size, @regime, @substring_count, @pos_offsets, @pos_bins)
    end

    # Builder: in-memory accumulator that gets compacted into the on-disk
    # sparse layout via #compact_to.
    class Builder
      getter window_size : Int32
      getter regime : Regime

      # Per-substring sparse histogram: substring_id => {pos_bin => count}
      @counts : Array(Hash(Int32, UInt32))

      def initialize(@window_size, @regime, substring_count : Int32)
        @counts = Array(Hash(Int32, UInt32)).new(substring_count) do
          Hash(Int32, UInt32).new
        end
      end

      def increment(substring_id : Int32, corpus_position : Int32) : Nil
        # Grow the per-substring array on demand
        while @counts.size <= substring_id
          @counts << Hash(Int32, UInt32).new
        end
        bin = corpus_position % @window_size
        @counts[substring_id][bin] = (@counts[substring_id][bin]? || 0_u32) + 1
      end

      def substring_count : Int32
        @counts.size
      end

      # Compact to PositionTable. Bins are sorted by position within each
      # substring for deterministic output and binary-search friendliness.
      def build : PositionTable
        sc = @counts.size
        offsets = Slice(Int32).new(sc + 1, 0)

        # Pass 1: compute offsets via prefix-sum of bin counts.
        total = 0
        @counts.each_with_index do |h, i|
          offsets[i] = total
          total += h.size
        end
        offsets[sc] = total

        # Pass 2: fill pos_bins.
        bins = Slice(PosBin).new(total) { PosBin.new(0_u16, 0_u32) }
        @counts.each_with_index do |h, i|
          sorted = h.to_a.sort_by(&.[0])
          sorted.each_with_index do |(pos, count), j|
            bins[offsets[i] + j] = PosBin.new(pos.to_u16, count)
          end
        end

        PositionTable.new(@window_size, @regime, sc, offsets, bins)
      end
    end

    def total_bins : Int32
      @pos_bins.size
    end

    # Returns a slice over the bins for this substring_id.
    def bins(substring_id : Int32) : Slice(PosBin)
      start = @pos_offsets[substring_id]
      stop = @pos_offsets[substring_id + 1]
      @pos_bins[start, stop - start]
    end

    # Sum of all counts for this substring (the substring's total occurrences).
    def total_count(substring_id : Int32) : UInt64
      sum = 0_u64
      bins(substring_id).each { |b| sum += b.count }
      sum
    end

    # Expected position: mean of position weighted by count.
    def expected_pos(substring_id : Int32) : Float32
      sum_pc = 0_u64
      sum_c = 0_u64
      bins(substring_id).each do |b|
        sum_pc += b.pos.to_u64 * b.count.to_u64
        sum_c += b.count.to_u64
      end
      return 0.0_f32 if sum_c == 0
      (sum_pc.to_f / sum_c.to_f).to_f32
    end

    def write_to(io : IO)
      io.write(MAGIC.to_slice)
      io.write_byte(@regime.value.to_u8)
      io.write_bytes(@window_size.to_u16, IO::ByteFormat::LittleEndian)
      io.write_byte(0_u8)  # reserved padding
      io.write_bytes(@substring_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(@pos_bins.size.to_u64, IO::ByteFormat::LittleEndian)

      # pos_offsets
      @pos_offsets.each do |o|
        io.write_bytes(o, IO::ByteFormat::LittleEndian)
      end
      # pos_bins (PosBin: u16 pos + u32 count = 6 bytes each)
      @pos_bins.each do |b|
        io.write_bytes(b.pos, IO::ByteFormat::LittleEndian)
        io.write_bytes(b.count, IO::ByteFormat::LittleEndian)
      end
    end

    def self.read_from(io : IO) : PositionTable
      magic = Bytes.new(4)
      io.read_fully(magic)
      raise "Bad magic in position table: expected #{MAGIC}, got #{String.new(magic)}" \
        unless magic == MAGIC.to_slice
      regime = Regime.new(io.read_byte.not_nil!)
      window_size = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian).to_i32
      io.read_byte  # reserved
      substring_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      total_bins = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian).to_i

      offsets = Slice(Int32).new(substring_count + 1) do
        io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      end
      bins = Slice(PosBin).new(total_bins) do
        pos = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian)
        count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
        PosBin.new(pos, count)
      end

      new(window_size, regime, substring_count, offsets, bins)
    end
  end
end
