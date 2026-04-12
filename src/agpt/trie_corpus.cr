module MicroGPT
  module AGPT
    # A corpus represented as a weighted prefix trie.
    #
    # Storage is columnar (struct-of-arrays): all per-node fields live in
    # parallel arrays on this object, indexed by a dense Int32 node id.
    # `TrieNode` is a thin value-type handle that forwards into these arrays.
    #
    # Children use a packed CSR-style storage: each parent's children live in a
    # contiguous, token-sorted slice of `@child_storage`. Insertion is O(k)
    # amortized (arises only during one-time ingestion); lookup is O(log k)
    # via binary search.
    #
    # Observation counts are stored in a sidecar hash keyed by node id. Only
    # observed nodes populate an entry, so most nodes cost zero extra bytes.
    #
    # MVP note:
    # This is intended for correctness-first replay-prefix training, not yet for
    # the full shared-prefix DAG execution described in the AGPT core note.
    class TrieCorpus
      MAGIC = 0x54475041_u32
      VERSION = 2_i32

      record IndexMetadata,
        corpus_token_count : Int32,
        vocab_size : Int32,
        corpus_hash : UInt64,
        tokenizer_tag : String do
      end

      record PrefixExample, prefix_tokens : Array(Int32), next_token_counts : Hash(Int32, Int32) do
        def total_count : Int32
          next_token_counts.values.sum(0)
        end
      end

      record ShapeStats,
        node_count : Int32,
        root_children : Int32,
        leaves : Int32,
        unary_nodes : Int32,
        branching_nodes : Int32,
        max_children : Int32,
        peak_width : Int32,
        peak_width_depth : Int32,
        avg_children_per_internal : Float64 do
      end

      # Columnar per-node storage, indexed by node id.
      @parents : Array(Int32)       # parent id, -1 for root
      @tokens : Array(Int32)        # incoming token_id, -1 for root
      @depths : Array(Int32)
      @first_child : Array(Int32)   # starting index into @child_storage, -1 if no children
      @child_count : Array(Int32)   # number of children owned by this node

      # Packed children storage: entries are (token, child_id), token-sorted
      # within each parent's contiguous slice. Slices are appended in parent-id
      # order during ingestion (the row for a node is created immediately after
      # the node itself is allocated, so no gaps exist between slices).
      @child_storage : Array({Int32, Int32})

      # Sidecar counts: only observed nodes get entries.
      @counts : Hash(Int32, Array({Int32, Int32}))

      property node_count : Int32
      getter max_depth : Int32?
      getter max_starts : Int32?
      getter start_offset : Int32
      property starts_used : Int32 = 0
      property index_metadata : IndexMetadata?

      def initialize(
        @max_depth : Int32? = nil,
        @max_starts : Int32? = nil,
        @start_offset : Int32 = 0,
        @progress_interval : Int32 = 0
      )
        @parents = [] of Int32
        @tokens = [] of Int32
        @depths = [] of Int32
        @first_child = [] of Int32
        @child_count = [] of Int32
        @child_storage = [] of {Int32, Int32}
        @counts = {} of Int32 => Array({Int32, Int32})
        @node_count = 0
        allocate_node(parent: -1, token: -1, depth: 0) # root, id 0
      end

      def root : TrieNode
        TrieNode.new(self, 0)
      end

      def self.from_token_ids(
        token_ids : Array(Int32),
        max_depth : Int32? = nil,
        max_starts : Int32? = nil,
        start_offset : Int32 = 0,
        progress_interval : Int32 = 0,
        vocab_size : Int32? = nil,
        corpus_hash : UInt64? = nil,
        tokenizer_tag : String = "unknown"
      ) : TrieCorpus
        corpus = new(max_depth, max_starts, start_offset, progress_interval)
        corpus.index_metadata = IndexMetadata.new(
          corpus_token_count: token_ids.size,
          vocab_size: vocab_size || 0,
          corpus_hash: corpus_hash || token_hash(token_ids),
          tokenizer_tag: tokenizer_tag
        )
        corpus.ingest(token_ids)
        corpus
      end

      def self.token_hash(token_ids : Array(Int32)) : UInt64
        hash = 1469598103934665603_u64
        prime = 1099511628211_u64
        token_ids.each do |token_id|
          value = token_id.to_u32.to_u64
          4.times do
            hash ^= value & 0xFF_u64
            hash &*= prime
            value >>= 8
          end
        end
        hash ^= token_ids.size.to_u64
        hash &*= prime
        hash
      end

      def self.load(path : String) : TrieCorpus
        File.open(path, "rb") do |io|
          magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
          raise "Invalid AGPT index file: bad magic" unless magic == MAGIC

          version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          raise "Unsupported AGPT index version: #{version}" unless version == VERSION

          max_depth_raw = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          max_starts_raw = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          start_offset = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          starts_used = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          node_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          corpus_token_count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          vocab_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          corpus_hash = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
          tokenizer_tag_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          tokenizer_bytes = Bytes.new(tokenizer_tag_size)
          io.read_fully(tokenizer_bytes)
          tokenizer_tag = String.new(tokenizer_bytes)

          max_depth = max_depth_raw < 0 ? nil : max_depth_raw
          max_starts = max_starts_raw < 0 ? nil : max_starts_raw
          corpus = new(max_depth, max_starts, start_offset)
          corpus.starts_used = starts_used
          corpus.index_metadata = IndexMetadata.new(
            corpus_token_count: corpus_token_count,
            vocab_size: vocab_size,
            corpus_hash: corpus_hash,
            tokenizer_tag: tokenizer_tag
          )

          node_count.times do |index|
            id = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            parent_id = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            token_id = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            depth = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            counts_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)

            if index == 0
              raise "Invalid AGPT index file: root node id must be 0" unless id == 0
              raise "Invalid AGPT index file: root parent must be -1" unless parent_id == -1
              raise "Invalid AGPT index file: root token must be -1" unless token_id == -1
              raise "Invalid AGPT index file: root depth must be 0" unless depth == 0
            else
              raise "Invalid AGPT index file: negative parent id for node #{id}" if parent_id < 0
              raise "Invalid AGPT index file: out-of-order id" unless id == corpus.node_count
              corpus.load_child(parent_id, token_id, depth)
            end

            counts = Array(TrieNode::CountEntry).new(counts_size)
            counts_size.times do
              next_token = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              counts << {next_token, count}
            end
            corpus.replace_counts(id, counts) unless counts.empty?
          end

          raise "Invalid AGPT index file: node count mismatch" unless corpus.node_count == node_count
          corpus
        end
      end

      def ingest(token_ids : Array(Int32))
        return if token_ids.size < 2

        limit = @max_depth || Int32::MAX
        total_starts = token_ids.size - 1
        starts = distributed_starts(total_starts)
        @starts_used = starts.size
        started_at = Time.instant

        starts.each_with_index do |start, i|
          path_id = 0 # root
          depth = 0
          index = start

          while index < total_starts && depth < limit
            tok = token_ids[index]
            next_tok = token_ids[index + 1]
            path_id = insert_prefix_step(path_id, tok)
            observe(path_id, next_tok)
            index += 1
            depth += 1
          end

          if @progress_interval > 0 && ((i + 1) % @progress_interval == 0 || i + 1 == starts.size)
            elapsed = Time.instant - started_at
            STDERR.puts "[agpt build] starts=#{i + 1}/#{starts.size} nodes=#{@node_count} elapsed=#{elapsed.total_seconds.round(2)}s"
          end
        end
      end

      def save(path : String)
        metadata = @index_metadata || IndexMetadata.new(
          corpus_token_count: 0,
          vocab_size: 0,
          corpus_hash: 0_u64,
          tokenizer_tag: "unknown"
        )

        File.open(path, "wb") do |io|
          io.write_bytes(MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(VERSION, IO::ByteFormat::LittleEndian)
          io.write_bytes((@max_depth || -1).to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes((@max_starts || -1).to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(@start_offset.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(@starts_used.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(@node_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(metadata.corpus_token_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(metadata.vocab_size.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(metadata.corpus_hash, IO::ByteFormat::LittleEndian)
          tokenizer_bytes = metadata.tokenizer_tag.to_slice
          io.write_bytes(tokenizer_bytes.size.to_i32, IO::ByteFormat::LittleEndian)
          io.write(tokenizer_bytes)

          @node_count.times do |id|
            io.write_bytes(id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(@parents[id].to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(@tokens[id].to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(@depths[id].to_i32, IO::ByteFormat::LittleEndian)
            entries = @counts[id]? || ([] of {Int32, Int32})
            io.write_bytes(entries.size.to_i32, IO::ByteFormat::LittleEndian)
            entries.each do |token_id, count|
              io.write_bytes(token_id.to_i32, IO::ByteFormat::LittleEndian)
              io.write_bytes(count.to_i32, IO::ByteFormat::LittleEndian)
            end
          end
        end
      end

      def validate_metadata!(
        *,
        corpus_token_count : Int32,
        vocab_size : Int32,
        corpus_hash : UInt64,
        tokenizer_tag : String,
        max_depth : Int32?
      )
        metadata = @index_metadata || raise "AGPT index metadata missing"

        if metadata.corpus_token_count != corpus_token_count
          raise "AGPT index corpus token count mismatch: index=#{metadata.corpus_token_count} current=#{corpus_token_count}"
        end
        if metadata.vocab_size != vocab_size
          raise "AGPT index vocab mismatch: index=#{metadata.vocab_size} current=#{vocab_size}"
        end
        if metadata.corpus_hash != corpus_hash
          raise "AGPT index corpus hash mismatch"
        end
        if metadata.tokenizer_tag != tokenizer_tag
          raise "AGPT index tokenizer mismatch: index=#{metadata.tokenizer_tag} current=#{tokenizer_tag}"
        end
        if @max_depth != max_depth
          raise "AGPT index max depth mismatch: index=#{@max_depth || "full"} current=#{max_depth || "full"}"
        end
      end

      def each_example(& : PrefixExample ->)
        each_observed_node do |node|
          yield PrefixExample.new(prefix_for(node), node.next_token_counts_hash)
        end
      end

      def example_count : Int32
        count = 0
        each_observed_node { count += 1 }
        count
      end

      def shape_stats : ShapeStats
        node_count = 0
        leaves = 0
        unary_nodes = 0
        branching_nodes = 0
        max_children = 0
        depth_counts = Hash(Int32, Int32).new(0)

        @node_count.times do |id|
          node_count += 1
          child_count = @child_count[id]
          depth_counts[@depths[id]] += 1
          leaves += 1 if child_count == 0
          unary_nodes += 1 if child_count == 1
          branching_nodes += 1 if child_count > 1
          max_children = child_count if child_count > max_children
        end

        internal_nodes = node_count - leaves
        total_children = node_count - 1
        avg_children_per_internal =
          if internal_nodes > 0
            total_children.to_f / internal_nodes
          else
            0.0
          end

        peak_width_depth = 0
        peak_width = 0
        depth_counts.each do |depth, count|
          next unless count > peak_width
          peak_width = count
          peak_width_depth = depth
        end

        ShapeStats.new(
          node_count: node_count,
          root_children: @child_count[0],
          leaves: leaves,
          unary_nodes: unary_nodes,
          branching_nodes: branching_nodes,
          max_children: max_children,
          peak_width: peak_width,
          peak_width_depth: peak_width_depth,
          avg_children_per_internal: avg_children_per_internal
        )
      end

      def each_depth_level(& : {Int32, Array(TrieNode)} ->)
        current_level = [root]
        depth = 0
        until current_level.empty?
          yield({depth, current_level})
          next_level = [] of TrieNode
          current_level.each do |node|
            each_child_of(node.id) do |_, child_id|
              next_level << TrieNode.new(self, child_id)
            end
          end
          current_level = next_level
          depth += 1
        end
      end

      def max_trie_depth : Int32
        max_d = 0
        @node_count.times do |id|
          d = @depths[id]
          max_d = d if d > max_d
        end
        max_d
      end

      def each_observed_node(& : TrieNode ->)
        @node_count.times do |id|
          next unless @counts.has_key?(id)
          yield TrieNode.new(self, id)
        end
      end

      def prefix_for(node : TrieNode) : Array(Int32)
        prefix = Array(Int32).new(node.depth)
        current_id = node.id
        while current_id != -1
          tok = @tokens[current_id]
          break if tok == -1
          prefix << tok
          current_id = @parents[current_id]
        end
        prefix.reverse!
        prefix
      end

      # ---------------- Columnar accessors (internal API used by TrieNode) ----

      def parent_id(id : Int32) : Int32
        @parents[id]
      end

      def token_id_of(id : Int32) : Int32
        @tokens[id]
      end

      def depth_of(id : Int32) : Int32
        @depths[id]
      end

      def children_of(id : Int32) : Array({Int32, Int32})
        start = @first_child[id]
        count = @child_count[id]
        return [] of {Int32, Int32} if count == 0
        result = Array({Int32, Int32}).new(count)
        count.times { |i| result << @child_storage[start + i] }
        result
      end

      def counts_of(id : Int32) : Array({Int32, Int32})
        entries = @counts[id]?
        return [] of {Int32, Int32} if entries.nil?
        entries.dup
      end

      # Binary search over parent's sorted child slice. Returns child id, or -1.
      def find_child(parent_id : Int32, token : Int32) : Int32
        start = @first_child[parent_id]
        count = @child_count[parent_id]
        return -1 if count == 0

        lo = 0
        hi = count - 1
        while lo <= hi
          mid = (lo + hi) >> 1
          entry_tok, entry_cid = @child_storage[start + mid]
          if entry_tok == token
            return entry_cid
          elsif entry_tok < token
            lo = mid + 1
          else
            hi = mid - 1
          end
        end
        -1
      end

      def ensure_child_id(parent_id : Int32, token : Int32, next_id : Int32) : Int32
        existing = find_child(parent_id, token)
        return existing unless existing == -1

        child_id = allocate_node(parent: parent_id, token: token, depth: @depths[parent_id] + 1)
        insert_child_entry(parent_id, token, child_id)
        child_id
      end

      def observe(node_id : Int32, next_token : Int32)
        entries = @counts[node_id] ||= [] of {Int32, Int32}
        entries.each_with_index do |(tok, count), index|
          next unless tok == next_token
          entries[index] = {tok, count + 1}
          return
        end
        entries << {next_token, 1}
      end

      def replace_counts(node_id : Int32, entries : Array({Int32, Int32}))
        if entries.empty?
          @counts.delete(node_id)
        else
          @counts[node_id] = entries.dup
        end
      end

      # Used by `TrieCorpus.load` to re-create a child node from the on-disk
      # stream. `id` is implicit — the next id to be allocated.
      def load_child(parent : Int32, token : Int32, depth : Int32) : Int32
        child_id = allocate_node(parent: parent, token: token, depth: depth)
        insert_child_entry(parent, token, child_id)
        child_id
      end

      # ---------------- Internal helpers --------------------------------------

      private def allocate_node(parent : Int32, token : Int32, depth : Int32) : Int32
        id = @node_count
        @parents << parent
        @tokens << token
        @depths << depth
        @first_child << -1
        @child_count << 0
        @node_count += 1
        id
      end

      # Insert (token, child_id) into the parent's sorted child slice.
      #
      # If the parent has never had a child before, start a new slice at the
      # tail of @child_storage. If the parent already owns a contiguous slice
      # that ends at the current tail of @child_storage, extend it in place.
      # Otherwise (the general case after many interleaved inserts) we shift
      # the slice to the tail and re-link. This is O(k) amortized which is
      # dominated by single-pass ingestion cost.
      private def insert_child_entry(parent_id : Int32, token : Int32, child_id : Int32)
        start = @first_child[parent_id]
        count = @child_count[parent_id]

        if count == 0
          # Brand new slice: put it at the end of @child_storage.
          @first_child[parent_id] = @child_storage.size
          @child_storage << {token, child_id}
          @child_count[parent_id] = 1
          return
        end

        end_index = start + count
        if end_index == @child_storage.size
          # Slice is already at the tail: sorted-insert without relocating.
          insert_pos = start + sorted_insert_pos(start, count, token)
          @child_storage.insert(insert_pos, {token, child_id})
          @child_count[parent_id] = count + 1
          return
        end

        # General case: relocate slice to tail, insert, update first_child.
        # (Reachable only after another parent appended children between this
        # parent's prior inserts. Cost is O(k) per such move, but each node
        # relocates at most a handful of times during ingestion because child
        # insertion order follows the BFS trie-walk.)
        new_start = @child_storage.size
        count.times { |i| @child_storage << @child_storage[start + i] }
        insert_pos = new_start + sorted_insert_pos(new_start, count, token)
        @child_storage.insert(insert_pos, {token, child_id})
        @first_child[parent_id] = new_start
        @child_count[parent_id] = count + 1
      end

      private def sorted_insert_pos(start : Int32, count : Int32, token : Int32) : Int32
        lo = 0
        hi = count
        while lo < hi
          mid = (lo + hi) >> 1
          entry_tok, _ = @child_storage[start + mid]
          if entry_tok < token
            lo = mid + 1
          else
            hi = mid
          end
        end
        lo
      end

      private def each_child_of(parent_id : Int32, & : Int32, Int32 ->)
        start = @first_child[parent_id]
        count = @child_count[parent_id]
        count.times do |i|
          tok, cid = @child_storage[start + i]
          yield tok, cid
        end
      end

      private def insert_prefix_step(path_id : Int32, token : Int32) : Int32
        existing = find_child(path_id, token)
        return existing unless existing == -1
        ensure_child_id(path_id, token, @node_count)
      end

      private def distributed_starts(total_starts : Int32) : Array(Int32)
        return [] of Int32 if total_starts <= 0

        count = @max_starts ? Math.min(total_starts, @max_starts.not_nil!) : total_starts
        return [] of Int32 if count <= 0
        return (0...total_starts).to_a if count == total_starts

        offset = @start_offset % total_starts
        offset += total_starts if offset < 0

        starts = Array(Int32).new(count)
        count.times do |i|
          base = ((i.to_i64 * total_starts) // count).to_i
          starts << ((base + offset) % total_starts)
        end
        starts
      end
    end
  end
end
