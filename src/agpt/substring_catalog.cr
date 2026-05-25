# Substring catalog: assigns a canonical, dense, sequential ID to every
# unique substring encountered. Used to unify per-substring data across
# the prefix and suffix tries — both tries' radix_ids translate to the
# same substring_id when they represent the same substring.
#
# Substrings are stored in forward (left-to-right corpus order) form.
# When indexing a suffix trie, callers must reverse the suffix-trie's
# token sequence before passing it in.
#
# In-memory representation: substrings are tuples of token IDs
# (Array(Int32)), keyed by content (Crystal hashes Array by element).
#
# On-disk format:
#   magic       u32   "ASUB" (0x42555341 little-endian)
#   count       u32   number of substrings
#   for each id 0..count-1:
#     length    u8    substring length in tokens (1..255)
#     tokens    u8[length]  token IDs (vocab_size must be ≤ 256)
#
# Storage estimate on Gutenberg 5M with d=16: roughly
#   7M substrings × (1 + avg_length 8) ≈ 63 MB

module MicroGPT::AGPT
  class SubstringCatalog
    MAGIC = "ASUB"

    @by_tokens : Hash(Array(Int32), Int32)
    @by_id : Array(Array(Int32))

    def initialize
      @by_tokens = {} of Array(Int32) => Int32
      @by_id = [] of Array(Int32)
    end

    # Returns the substring_id for these tokens, assigning a new ID if
    # this is the first time the sequence has been seen.
    def get_or_assign(tokens : Array(Int32)) : Int32
      if id = @by_tokens[tokens]?
        id
      else
        id = @by_id.size
        # Store a defensive copy so the caller can mutate freely.
        copy = tokens.dup
        @by_tokens[copy] = id
        @by_id << copy
        id
      end
    end

    # Returns the substring_id for these tokens, or nil if not in catalog.
    def lookup(tokens : Array(Int32)) : Int32?
      @by_tokens[tokens]?
    end

    # Returns the token sequence for a substring_id.
    def tokens_for(id : Int32) : Array(Int32)
      @by_id[id]
    end

    def size : Int32
      @by_id.size
    end

    def each_with_id(&block : Array(Int32), Int32 ->)
      @by_id.each_with_index do |tokens, id|
        yield tokens, id
      end
    end

    # Write the catalog to an IO. Token IDs must all fit in u8 (vocab_size ≤ 256).
    def write_to(io : IO)
      io.write(MAGIC.to_slice)
      io.write_bytes(@by_id.size.to_i32, IO::ByteFormat::LittleEndian)
      @by_id.each do |tokens|
        raise "Substring too long for u8 length field: #{tokens.size}" if tokens.size > 255
        io.write_byte(tokens.size.to_u8)
        tokens.each do |t|
          raise "Token id #{t} does not fit in u8 (vocab_size > 256)" if t < 0 || t > 255
          io.write_byte(t.to_u8)
        end
      end
    end

    # Read a catalog from an IO.
    def self.read_from(io : IO) : SubstringCatalog
      magic = Bytes.new(4)
      io.read_fully(magic)
      raise "Bad magic in substring catalog: expected #{MAGIC}, got #{String.new(magic)}" \
        unless magic == MAGIC.to_slice
      count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      catalog = new
      count.times do
        len = io.read_byte.not_nil!.to_i
        tokens = Array(Int32).new(len) { io.read_byte.not_nil!.to_i }
        catalog.get_or_assign(tokens)
      end
      catalog
    end
  end
end
