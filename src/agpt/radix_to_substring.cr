# Per-trie lookup from radix_id → substring_id (canonical ID from
# SubstringCatalog). One file per trie; loader detects which one to use
# based on which trie the consumer is processing.
#
# On-disk format (binary, little-endian):
#   magic       u32   "PRTS" for prefix-side, "SRTS" for suffix-side
#   radix_count u32   number of radix nodes (must match the trie)
#   ids         i32[radix_count]  substring_id for each radix_id
#
# Storage: 4 bytes/node × ~7M nodes ≈ 28 MB per trie on Gutenberg.

module MicroGPT::AGPT
  class RadixToSubstring
    PREFIX_MAGIC = "PRTS"
    SUFFIX_MAGIC = "SRTS"

    enum Side : UInt8
      Prefix = 0
      Suffix = 1
    end

    getter side : Side
    getter ids : Slice(Int32)

    def initialize(@side, @ids)
    end

    def radix_count : Int32
      @ids.size
    end

    def substring_id_for(radix_id : Int32) : Int32
      @ids[radix_id]
    end

    def write_to(io : IO)
      magic = (@side == Side::Prefix ? PREFIX_MAGIC : SUFFIX_MAGIC).to_slice
      io.write(magic)
      io.write_bytes(@ids.size.to_u32, IO::ByteFormat::LittleEndian)
      @ids.each { |id| io.write_bytes(id, IO::ByteFormat::LittleEndian) }
    end

    def self.read_from(io : IO) : RadixToSubstring
      magic = Bytes.new(4)
      io.read_fully(magic)
      side = case String.new(magic)
             when PREFIX_MAGIC then Side::Prefix
             when SUFFIX_MAGIC then Side::Suffix
             else raise "Bad magic in radix-to-substring file: #{String.new(magic)}"
             end
      count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i
      ids = Slice(Int32).new(count) do
        io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      end
      new(side, ids)
    end
  end
end
