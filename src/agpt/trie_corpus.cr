module MicroGPT
  module AGPT
    # A corpus represented as a weighted prefix trie.
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

      getter root : TrieNode
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
        @root = TrieNode.new(id: 0)
        @node_count = 1
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

          nodes_by_id = Array(TrieNode?).new(node_count, nil)

          node_count.times do |index|
            id = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            parent_id = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            token_id = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            depth = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
            counts_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)

            node =
              if index == 0
                raise "Invalid AGPT index file: root node id must be 0" unless id == 0
                raise "Invalid AGPT index file: root parent must be -1" unless parent_id == -1
                raise "Invalid AGPT index file: root token must be -1" unless token_id == -1
                raise "Invalid AGPT index file: root depth must be 0" unless depth == 0
                corpus.root
              else
                raise "Invalid AGPT index file: negative parent id for node #{id}" if parent_id < 0
                parent = nodes_by_id[parent_id].not_nil!
                parent.ensure_child(token_id, id)
              end

            counts = Array(TrieNode::CountEntry).new(counts_size)
            counts_size.times do
              next_token = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              count = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
              counts << {next_token, count}
            end
            node.replace_next_token_counts(counts)
            nodes_by_id[id] = node
          end

          corpus.node_count = node_count
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
          path = @root
          depth = 0
          index = start

          while index < total_starts && depth < limit
            tok = token_ids[index]
            next_tok = token_ids[index + 1]
            path = insert_prefix_step(path, tok)
            path.observe(next_tok)
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
        nodes = Array(TrieNode?).new(@node_count, nil)
        each_node do |node|
          nodes[node.id] = node
        end

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

          nodes.each do |node|
            actual = node.not_nil!
            io.write_bytes(actual.id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes((actual.parent.try(&.id) || -1).to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes((actual.token_id || -1).to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(actual.depth.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(actual.next_token_counts.size.to_i32, IO::ByteFormat::LittleEndian)
            actual.next_token_counts.each do |token_id, count|
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

        each_node do |node|
          node_count += 1
          child_count = node.children.size
          depth_counts[node.depth] += 1
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
          root_children: @root.children.size,
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
        current_level = [@root]
        depth = 0
        until current_level.empty?
          yield({depth, current_level})
          next_level = [] of TrieNode
          current_level.each do |node|
            node.children.each { |_, child| next_level << child }
          end
          current_level = next_level
          depth += 1
        end
      end

      def max_trie_depth : Int32
        max_d = 0
        each_node { |n| max_d = n.depth if n.depth > max_d }
        max_d
      end

      def each_observed_node(& : TrieNode ->)
        each_node do |node|
          next if node.next_token_counts.empty?
          yield node
        end
      end

      def prefix_for(node : TrieNode) : Array(Int32)
        prefix = Array(Int32).new(node.depth)
        current = node
        while token = current.token_id
          prefix << token
          parent = current.parent
          break unless parent
          current = parent
        end
        prefix.reverse!
        prefix
      end

      private def insert_prefix_step(path : TrieNode, token : Int32) : TrieNode
        child = path.child_for(token)
        return child if child

        new_node = path.ensure_child(token, @node_count)
        @node_count += 1
        new_node
      end

      private def each_node(& : TrieNode ->)
        stack = [@root]
        until stack.empty?
          node = stack.pop
          yield node
          node.children.each do |_, child|
            stack << child
          end
        end
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
