module MicroGPT
  module AGPT
    # Reader for the radix-compressed trie index written by StreamingRadixBuilder.
    #
    # Mirrors LeveledTrieReader's LRU-per-depth pattern. Each depth file is a
    # group of radix nodes whose edges END at that character depth.
    class RadixTrieReader
      MAGIC   = 0x52445841_u32  # 'RDXA'
      # Version 3 (2026-05-17): edge tokens and counts_tok stored as int16
      # instead of int32. Cuts edge_tokens disk/RAM by 2×. Reader supports
      # both v2 (int32 tokens) and v3 (int16 tokens) for backward compat
      # with already-built tries.
      VERSION = 3_i32
      VERSION_INT16_TOKENS = 3_i32  # threshold version for narrow token storage

      struct LoadedRecord
        getter id : Int32
        getter parent_id : Int32
        getter first_char_depth : Int32
        getter edge_tokens : Array(Int32)
        getter edge_mass : Int32
        getter counts : Array({Int32, Int32})

        def initialize(@id, @parent_id, @first_char_depth, @edge_tokens, @edge_mass, @counts)
        end

        def edge_len : Int32
          @edge_tokens.size
        end

        def endpoint_depth : Int32
          @first_char_depth + @edge_tokens.size - 1
        end
      end

      class LoadedDepth
        getter depth : Int32
        getter records : Array(LoadedRecord)
        getter id_to_local : Hash(Int32, Int32)

        def initialize(@depth, @records, @id_to_local)
        end
      end

      getter dir : String
      getter radix_count : Int32
      getter depth_file_count : Int32
      getter total_edge_chars : Int64
      getter corpus_token_count : Int32
      getter vocab_size : Int32
      getter corpus_hash : UInt64
      getter tokenizer_tag : String

      @loaded : Hash(Int32, LoadedDepth)
      @lru : Array(Int32)
      @max_cached : Int32
      @file_version : Int32  # actual version of files on disk; controls int16 vs int32 token reads

      def initialize(@dir : String, @max_cached : Int32 = 3)
        meta_path = File.join(@dir, "meta.bin")
        raise "meta.bin missing in #{@dir}" unless File.exists?(meta_path)

        @radix_count = 0
        @depth_file_count = 0
        @total_edge_chars = 0_i64
        @corpus_token_count = 0
        @vocab_size = 0
        @corpus_hash = 0_u64
        @tokenizer_tag = "unknown"
        @file_version = 0

        File.open(meta_path, "rb") do |io|
          magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
          raise "bad radix magic in #{meta_path}" unless magic == MAGIC
          version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          # Accept v2 (int32 tokens) and v3 (int16 tokens). Versions older
          # than 2 or future-newer-than-current are not understood.
          unless version == 2 || version == 3
            raise "unsupported radix version #{version} (supported: 2, 3)"
          end
          @file_version = version
          @radix_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          @depth_file_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          @total_edge_chars = io.read_bytes(Int64, IO::ByteFormat::LittleEndian)
          @corpus_token_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          @vocab_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          @corpus_hash = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
          tlen = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          tbytes = Bytes.new(tlen)
          io.read_fully(tbytes)
          @tokenizer_tag = String.new(tbytes)
        end

        @loaded = {} of Int32 => LoadedDepth
        @lru = [] of Int32
      end

      def fault(d : Int32) : LoadedDepth
        if cached = @loaded[d]?
          @lru.delete(d)
          @lru << d
          return cached
        end

        records = [] of LoadedRecord
        id_to_local = {} of Int32 => Int32

        path = File.join(@dir, "radix_depth_#{"%03d" % d}.bin")
        if !File.exists?(path)
          # Empty depth file — return empty LoadedDepth
          loaded = LoadedDepth.new(d, records, id_to_local)
          @loaded[d] = loaded
          @lru << d
          return loaded
        end

        File.open(path, "rb") do |io|
          magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
          raise "bad radix depth magic in #{path}" unless magic == MAGIC
          stored_depth = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          raise "depth mismatch in #{path}: #{stored_depth} vs #{d}" unless stored_depth == d
          n = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          # In v3, edge tokens and counts_tok are int16 on disk. In v2, all int32.
          # We promote both to Int32 in the in-memory LoadedRecord for API stability.
          narrow_tokens = (@file_version >= VERSION_INT16_TOKENS)
          n.times do |i|
            rid = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            parent = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            fcd = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            edge_len = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            edge = Array(Int32).new(edge_len)
            if narrow_tokens
              edge_len.times { edge << io.read_bytes(Int16, IO::ByteFormat::LittleEndian).to_i32 }
            else
              edge_len.times { edge << io.read_bytes(Int32, IO::ByteFormat::LittleEndian) }
            end
            edge_mass = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            ec = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            entries = Array({Int32, Int32}).new(ec)
            ec.times do
              if narrow_tokens
                tok = io.read_bytes(Int16, IO::ByteFormat::LittleEndian).to_i32
              else
                tok = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              end
              cnt = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              entries << {tok, cnt}
            end
            records << LoadedRecord.new(rid, parent, fcd, edge, edge_mass, entries)
            id_to_local[rid] = i
          end
        end

        loaded = LoadedDepth.new(d, records, id_to_local)
        @loaded[d] = loaded
        @lru << d

        while @lru.size > @max_cached
          oldest = @lru.shift
          @loaded.delete(oldest)
        end

        loaded
      end

      def nodes_at_endpoint_depth(d : Int32) : Array(LoadedRecord)
        fault(d).records
      end

      # Walk all radix records in endpoint-depth order.
      def each(&block : LoadedRecord ->)
        @depth_file_count.times do |d|
          nodes_at_endpoint_depth(d).each { |r| yield r }
        end
      end
    end
  end
end
