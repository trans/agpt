module MicroGPT
  module AGPT
    # Lazy per-depth reader for the save_by_depth file format.
    #
    # Loads depth files on demand into a bounded LRU cache. Only meta.bin and
    # the id→depth lookup are always resident. Each faulted depth stores the
    # NodeRecord array and next-token counts for that depth's nodes.
    #
    # Intended for training where access pattern is BFS (current depth +
    # parent depth). Default cache holds 3 depths.
    class LeveledTrieReader
      include TrieAccessor

      MAGIC = 0x4C475041_u32  # 'LGPA' (matches TrieCorpus::LEVELED_MAGIC)

      # Per-node record as loaded from disk for one depth.
      struct LoadedRecord
        getter id : Int32
        getter parent_id : Int32
        getter token : Int32
        getter depth : Int32
        getter child_count : Int32
        getter first_child : Int32

        def initialize(@id, @parent_id, @token, @depth, @child_count, @first_child)
        end
      end

      # Cached depth chunk: records for nodes at this depth + optional counts.
      class LoadedDepth
        getter depth : Int32
        getter records : Array(LoadedRecord)
        getter id_to_local : Hash(Int32, Int32)   # global_id → index into records
        getter counts : Hash(Int32, Array({Int32, Int32}))   # global_id → entries

        def initialize(@depth, @records, @id_to_local, @counts)
        end
      end

      getter dir : String
      getter node_count : Int32
      getter max_depth : Int32?
      getter max_starts : Int32?
      getter start_offset : Int32
      getter starts_used : Int32
      getter index_metadata : TrieCorpus::IndexMetadata
      getter depth_file_count : Int32
      getter id_to_depth : Array(UInt8)    # 1 byte per node, always resident

      @loaded : Hash(Int32, LoadedDepth)
      @lru : Array(Int32)
      @max_cached : Int32

      def initialize(@dir : String, @max_cached : Int32 = 3)
        meta_path = File.join(@dir, "meta.bin")
        raise "meta.bin missing in #{@dir}" unless File.exists?(meta_path)

        node_count = 0
        depth_file_count = 0
        @node_count = 0
        @depth_file_count = 0
        @max_depth = nil
        @max_starts = nil
        @start_offset = 0
        @starts_used = 0
        @index_metadata = TrieCorpus::IndexMetadata.new(0, 0, 0_u64, "unknown")

        File.open(meta_path, "rb") do |io|
          magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
          raise "bad leveled magic in #{meta_path}" unless magic == MAGIC
          version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          raise "unsupported leveled version #{version}" unless version == TrieCorpus::LEVELED_VERSION
          mdx = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          mst = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          @start_offset = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          @starts_used = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          @node_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          @depth_file_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          ctc = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          vs = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          ch = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
          tlen = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          tbytes = Bytes.new(tlen)
          io.read_fully(tbytes)
          @max_depth = mdx < 0 ? nil : mdx
          @max_starts = mst < 0 ? nil : mst
          @index_metadata = TrieCorpus::IndexMetadata.new(
            corpus_token_count: ctc,
            vocab_size: vs,
            corpus_hash: ch,
            tokenizer_tag: String.new(tbytes)
          )
        end

        @loaded = {} of Int32 => LoadedDepth
        @lru = [] of Int32

        # Build id_to_depth by scanning depth files' headers + id columns.
        # Cheap: one pass over fixed-size records, reads only id + depth.
        @id_to_depth = Array(UInt8).new(@node_count, 0_u8)
        build_id_to_depth
      end

      private def build_id_to_depth
        @depth_file_count.times do |d|
          path = File.join(@dir, "depth_#{"%03d" % d}.bin")
          next unless File.exists?(path)
          File.open(path, "rb") do |io|
            magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
            raise "bad depth magic in #{path}" unless magic == MAGIC
            stored_depth = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            raise "depth mismatch in #{path}: #{stored_depth} vs #{d}" unless stored_depth == d
            n = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            n.times do
              id = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              @id_to_depth[id] = d.to_u8
              # skip remaining 5 Int32 fields (parent, token, depth,
              # child_count, first_child) + variable-length counts section.
              io.read_bytes(Int32, IO::ByteFormat::LittleEndian)   # parent
              io.read_bytes(Int32, IO::ByteFormat::LittleEndian)   # token
              io.read_bytes(Int32, IO::ByteFormat::LittleEndian)   # depth
              io.read_bytes(Int32, IO::ByteFormat::LittleEndian)   # child_count
              io.read_bytes(Int32, IO::ByteFormat::LittleEndian)   # first_child
              entry_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              entry_count.times do
                io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
                io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              end
            end
          end
        end
      end

      # Fault in depth d's records if not already cached. Update LRU.
      # Evict oldest if cache exceeds @max_cached.
      def fault(d : Int32) : LoadedDepth
        if cached = @loaded[d]?
          @lru.delete(d)
          @lru << d
          return cached
        end

        records = [] of LoadedRecord
        id_to_local = {} of Int32 => Int32
        counts = {} of Int32 => Array({Int32, Int32})

        path = File.join(@dir, "depth_#{"%03d" % d}.bin")
        File.open(path, "rb") do |io|
          magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
          raise "bad depth magic" unless magic == MAGIC
          stored_depth = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          raise "depth mismatch" unless stored_depth == d
          n = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          n.times do |i|
            id = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            parent = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            token = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            depth = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            child_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            first_child = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            entry_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            entries = [] of {Int32, Int32}
            entry_count.times do
              tok = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              cnt = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              entries << {tok, cnt}
            end
            records << LoadedRecord.new(id, parent, token, depth, child_count, first_child)
            id_to_local[id] = i
            counts[id] = entries unless entries.empty?
          end
        end

        loaded = LoadedDepth.new(d, records, id_to_local, counts)
        @loaded[d] = loaded
        @lru << d

        while @lru.size > @max_cached
          oldest = @lru.shift
          @loaded.delete(oldest)
        end

        loaded
      end

      # Accessors — all route through fault() and look up by global id.
      def record(id : Int32) : LoadedRecord
        d = @id_to_depth[id].to_i
        chunk = fault(d)
        idx = chunk.id_to_local[id]
        chunk.records[idx]
      end

      def parent_id(id : Int32) : Int32
        record(id).parent_id
      end

      def token_id_of(id : Int32) : Int32
        record(id).token
      end

      def depth_of(id : Int32) : Int32
        @id_to_depth[id].to_i
      end

      def counts_of(id : Int32) : Array({Int32, Int32})
        d = depth_of(id)
        chunk = fault(d)
        chunk.counts[id]? || ([] of {Int32, Int32})
      end

      def child_count_of(id : Int32) : Int32
        record(id).child_count
      end

      def nodes_at_depth(d : Int32) : Array(LoadedRecord)
        fault(d).records
      end
    end
  end
end
