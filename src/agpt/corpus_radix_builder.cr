module MicroGPT
  module AGPT
    # Builds a radix-compressed trie directly from a corpus, processing one
    # root-character subtree at a time. Bypasses the leveled-trie intermediate
    # used by StreamingRadixBuilder, so peak memory is bounded by ONE subtree's
    # working set (~1/vocab_size of the full trie).
    #
    # Designed for corpora that don't fit in memory under the leveled-then-radix
    # pipeline (e.g. 5M+ chars at d=32 OOMs on a 16 GB box via that path).
    #
    # Output is byte-compatible with StreamingRadixBuilder's depth-flat format
    # (radix_depth_NNN.bin + meta.bin), so all existing consumers
    # (synth_wrap_corpus, agpt_train, radix-verify, trie-profile) read it
    # unchanged. Optionally also emits per-subtree files (manifest.bin +
    # subtrees/radix_subtree_NNNNNN.bin) when @per_subtree is true.
    class CorpusRadixBuilder
      RADIX_MAGIC   = 0x52445841_u32  # 'RDXA' — same as StreamingRadixBuilder
      RADIX_VERSION = 2_i32

      # CharTrieNode: mutable in-memory node for one subtree's character trie.
      # Children are keyed by token id. count = total D-gram inserts that pass
      # through this node (= the prefix's frequency at this depth).
      private class CharTrieNode
        property children : Hash(Int32, CharTrieNode)
        property count : Int32
        def initialize
          @children = {} of Int32 => CharTrieNode
          @count = 0
        end
      end

      # DepthStream / SubtreeStream: same role as StreamingRadixBuilder's
      # streaming writers. Headers are placeholder-then-patch so we can write
      # records as they're produced without buffering the full output.
      private class DepthStream
        property io : File
        property count : Int32
        def initialize(@io : File)
          @count = 0
        end
      end

      private class SubtreeStream
        property io : File
        property count : Int32
        property total_edge_chars : Int64
        property max_ep : Int32
        def initialize(@io : File)
          @count = 0
          @total_edge_chars = 0_i64
          @max_ep = 0
        end
      end

      def initialize(
        @corpus_tokens : Array(Int32),
        @vocab_size : Int32,
        @max_depth : Int32,
        @out_dir : String,
        @corpus_hash : UInt64,
        @tokenizer_tag : String,
        @progress : Bool = true,
        @per_subtree : Bool = false,
        @subtree_level : Int32 = 1,
        @prune_min_mass : Int32 = 1,
        @prune_min_depth : Int32 = 4
      )
        unless @subtree_level == 1
          raise "CorpusRadixBuilder currently only supports subtree_level=1 (unigram)"
        end
      end

      def build : NamedTuple(radix_count: Int32, total_edge_chars: Int64, max_endpoint_depth: Int32)
        Dir.mkdir_p(@out_dir)
        Dir.mkdir_p(File.join(@out_dir, "subtrees")) if @per_subtree

        build_started = Time.instant

        # Index the corpus once: positions[c] = sorted list of i such that
        # corpus_tokens[i] == c. Then a subtree's D-gram set is just D-gram(i)
        # for each i in positions[c]. ~vocab_size lists, total size = corpus
        # length.
        positions = Array.new(@vocab_size) { [] of Int32 }
        @corpus_tokens.each_with_index do |tok, i|
          positions[tok] << i
        end

        depth_streams   = {} of Int32 => DepthStream
        subtree_streams = {} of Int32 => SubtreeStream

        next_radix_id = 1_i32   # 0 reserved for the virtual root
        radix_count = 1         # count the virtual root
        total_edge_chars = 0_i64
        max_endpoint_depth = 0

        # Process each root character's subtree in turn. Each iteration builds
        # a small char-trie of the D-grams starting with that char, walks it
        # to emit radix records, then drops the trie to free memory.
        @vocab_size.times do |root_char|
          starts = positions[root_char]
          next if starts.empty?

          char_root = build_char_trie(starts, root_char)

          # Walk the subtree, emitting radix records. The subtree root
          # represents the depth-1 character (root_char itself); its parent in
          # the radix tree is the virtual root (radix_id = 0).
          subtree_id = next_radix_id
          before_count = radix_count

          rc = emit_subtree(
            char_root,
            head_token: root_char,
            parent_radix_id: 0,
            first_char_depth: 1,
            subtree_id: subtree_id,
            depth_streams: depth_streams,
            subtree_streams: subtree_streams,
            next_radix_id: next_radix_id,
            total_edge_chars: total_edge_chars,
            max_endpoint_depth: max_endpoint_depth,
          )
          next_radix_id      = rc[:next_radix_id]
          total_edge_chars   = rc[:total_edge_chars]
          max_endpoint_depth = rc[:max_endpoint_depth]
          radix_count       += rc[:emitted]

          if @progress
            elapsed = (Time.instant - build_started).total_seconds
            STDERR.puts "[radix-corpus] subtree char=#{root_char} (#{starts.size} positions): emitted #{rc[:emitted]} radix nodes  (cum total #{radix_count}, max_ep_depth=#{max_endpoint_depth}, #{elapsed.round(1)}s)"
          end

          # Drop the char trie. Crystal GC will collect it before the next
          # subtree's working set is allocated.
          char_root = nil
        end

        # Patch per-depth file headers and close.
        depth_streams.each do |_, stream|
          stream.io.seek(8)   # past magic(4) + depth(4)
          stream.io.write_bytes(stream.count.to_i32, IO::ByteFormat::LittleEndian)
          stream.io.close
        end
        # Mirror StreamingRadixBuilder: write empty depth files for any depth
        # 0..max_endpoint_depth that wasn't opened. Keeps depth_file_count
        # consistent for downstream readers.
        (0..max_endpoint_depth).each do |d|
          next if depth_streams.has_key?(d)
          path = File.join(@out_dir, "radix_depth_#{"%03d" % d}.bin")
          File.open(path, "wb") do |io|
            io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
            io.write_bytes(d.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(0_i32, IO::ByteFormat::LittleEndian)
          end
        end

        # Patch per-subtree headers and write manifest.
        if @per_subtree
          manifest = [] of {Int32, Int32, Int64, Int32}
          subtree_streams.each do |key, stream|
            stream.io.seek(12)  # past magic(4) + version(4) + subtree_key(4)
            stream.io.write_bytes(stream.count.to_i32, IO::ByteFormat::LittleEndian)
            stream.io.write_bytes(stream.total_edge_chars, IO::ByteFormat::LittleEndian)
            stream.io.write_bytes(stream.max_ep.to_i32, IO::ByteFormat::LittleEndian)
            stream.io.close
            manifest << {key, stream.count, stream.total_edge_chars, stream.max_ep}
          end
          manifest.sort_by! { |m| m[0] }
          write_manifest(manifest)
        end

        write_meta(
          radix_count,
          max_endpoint_depth + 1,
          total_edge_chars,
          @corpus_tokens.size,
          @vocab_size,
          @corpus_hash,
          @tokenizer_tag,
        )

        elapsed = (Time.instant - build_started).total_seconds
        STDERR.puts "[radix-corpus] done: #{radix_count} radix nodes, #{total_edge_chars} total edge chars, #{max_endpoint_depth + 1} endpoint depths, #{elapsed.round(2)}s"

        {radix_count: radix_count, total_edge_chars: total_edge_chars, max_endpoint_depth: max_endpoint_depth}
      end

      # Build a character-level trie of the D-grams whose first char is
      # root_char, by inserting each D-gram one position at a time. The root
      # of the returned trie represents the depth-1 character (root_char).
      private def build_char_trie(starts : Array(Int32), root_char : Int32) : CharTrieNode
        root = CharTrieNode.new
        starts.each do |i|
          current = root
          current.count += 1
          # Insert chars at depths 2..max_depth+1 (i.e. corpus positions
          # i+1..i+max_depth). The depth-(max_depth+1) layer exists only to
          # populate endpoint_counts at the depth-max_depth radix nodes —
          # we never emit a radix record at depth > max_depth. This matches
          # the leveled builder's behavior of recording the (d+1)th char
          # distribution at every depth-d node, including d = max_depth.
          last = i + @max_depth + 1
          last = @corpus_tokens.size if last > @corpus_tokens.size
          j = i + 1
          while j < last
            tok = @corpus_tokens[j]
            child = current.children[tok]?
            if child.nil?
              child = CharTrieNode.new
              current.children[tok] = child
            end
            current = child
            current.count += 1
            j += 1
          end
        end
        root
      end

      # Walk the char trie and emit radix records. `node` is the head of an
      # incoming edge — the unary chain extends from `node` until we hit a
      # branching point or the depth cap. `head_token` is the token at the
      # node's position (the key in the parent's children map that pointed
      # to it; for the subtree root, it's the subtree's root_char).
      #
      # Returns updated counters in a NamedTuple.
      private def emit_subtree(
        node : CharTrieNode,
        head_token : Int32,
        parent_radix_id : Int32,
        first_char_depth : Int32,
        subtree_id : Int32,
        depth_streams : Hash(Int32, DepthStream),
        subtree_streams : Hash(Int32, SubtreeStream),
        next_radix_id : Int32,
        total_edge_chars : Int64,
        max_endpoint_depth : Int32,
      )
        emitted = 0
        # Iterative work-list to avoid deep recursion (paths can be ~max_depth long).
        # Each entry: (node, head_token, parent_radix_id, first_char_depth)
        work = Deque({CharTrieNode, Int32, Int32, Int32}).new
        work << {node, head_token, parent_radix_id, first_char_depth}

        while !work.empty?
          n, head_tok, p_id, fcd = work.shift

          # Walk the unary chain from n down through children-of-size-1.
          current = n
          current_depth = fcd
          edge_tokens = [head_tok]

          # mass at the head of the edge = count at this node
          edge_mass = current.count

          # Extend through unary chain
          while current.children.size == 1 && current_depth < @max_depth
            tok, child = current.children.first
            current = child
            current_depth += 1
            edge_tokens << tok
          end

          # `current` is the endpoint. Build endpoint_counts from its children.
          endpoint_counts = current.children.map { |tok, child| {tok, child.count} }

          # End-of-corpus tail: empty endpoint counts means there's no next-
          # token observation at this prefix (we hit the corpus end). No
          # training signal — drop. Matches StreamingRadixBuilder semantics.
          if endpoint_counts.empty?
            next
          end

          # Frequency pruning (matches StreamingRadixBuilder semantics).
          if edge_mass < @prune_min_mass && fcd >= @prune_min_depth
            next
          end

          radix_id = next_radix_id
          next_radix_id += 1
          emitted += 1
          total_edge_chars += edge_tokens.size
          endpoint_depth = current_depth
          max_endpoint_depth = endpoint_depth if endpoint_depth > max_endpoint_depth

          # Per-depth output stream.
          dstream = depth_streams[endpoint_depth]?
          if dstream.nil?
            path = File.join(@out_dir, "radix_depth_#{"%03d" % endpoint_depth}.bin")
            io = File.open(path, "wb")
            io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
            io.write_bytes(endpoint_depth.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(0_i32, IO::ByteFormat::LittleEndian)
            dstream = DepthStream.new(io)
            depth_streams[endpoint_depth] = dstream
          end
          write_record(dstream.io, radix_id, p_id, fcd, edge_tokens, edge_mass, endpoint_counts)
          dstream.count += 1

          if @per_subtree
            sstream = subtree_streams[subtree_id]?
            if sstream.nil?
              dir = File.join(@out_dir, "subtrees")
              spath = File.join(dir, "radix_subtree_#{"%06d" % subtree_id}.bin")
              sio = File.open(spath, "wb")
              sio.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
              sio.write_bytes(RADIX_VERSION, IO::ByteFormat::LittleEndian)
              sio.write_bytes(subtree_id.to_i32, IO::ByteFormat::LittleEndian)
              sio.write_bytes(0_i32, IO::ByteFormat::LittleEndian)
              sio.write_bytes(0_i64, IO::ByteFormat::LittleEndian)
              sio.write_bytes(0_i32, IO::ByteFormat::LittleEndian)
              sstream = SubtreeStream.new(sio)
              subtree_streams[subtree_id] = sstream
            end
            write_record(sstream.io, radix_id, p_id, fcd, edge_tokens, edge_mass, endpoint_counts)
            sstream.count += 1
            sstream.total_edge_chars += edge_tokens.size.to_i64
            sstream.max_ep = endpoint_depth if endpoint_depth > sstream.max_ep
          end

          # Queue each branching child of the endpoint as a new edge head.
          # We only get here with ≥2 children OR depth-cap reached. The
          # depth-cap case has no children to queue. Each queued child
          # carries its head token (the key it sits under in current.children).
          if endpoint_counts.size >= 2
            current.children.each do |child_tok, child|
              work << {child, child_tok, radix_id, current_depth + 1}
            end
          end
        end

        {next_radix_id: next_radix_id, total_edge_chars: total_edge_chars, max_endpoint_depth: max_endpoint_depth, emitted: emitted}
      end

      private def write_record(
        io : IO,
        radix_id : Int32,
        parent_radix_id : Int32,
        first_char_depth : Int32,
        edge : Array(Int32),
        edge_mass : Int32,
        entries : Array({Int32, Int32}),
      )
        io.write_bytes(radix_id.to_i32, IO::ByteFormat::LittleEndian)
        io.write_bytes(parent_radix_id.to_i32, IO::ByteFormat::LittleEndian)
        io.write_bytes(first_char_depth.to_i32, IO::ByteFormat::LittleEndian)
        io.write_bytes(edge.size.to_i32, IO::ByteFormat::LittleEndian)
        edge.each { |tok| io.write_bytes(tok.to_i32, IO::ByteFormat::LittleEndian) }
        io.write_bytes(edge_mass.to_i32, IO::ByteFormat::LittleEndian)
        io.write_bytes(entries.size.to_i32, IO::ByteFormat::LittleEndian)
        entries.each do |(token_id, count)|
          io.write_bytes(token_id.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(count.to_i32, IO::ByteFormat::LittleEndian)
        end
      end

      private def write_manifest(manifest : Array({Int32, Int32, Int64, Int32}))
        path = File.join(@out_dir, "manifest.bin")
        File.open(path, "wb") do |io|
          io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(RADIX_VERSION, IO::ByteFormat::LittleEndian)
          io.write_bytes(manifest.size.to_i32, IO::ByteFormat::LittleEndian)
          manifest.each do |(rc, n, chars, max_ep)|
            io.write_bytes(rc.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(n.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(chars.to_i64, IO::ByteFormat::LittleEndian)
            io.write_bytes(max_ep.to_i32, IO::ByteFormat::LittleEndian)
          end
        end
      end

      private def write_meta(
        radix_count : Int32,
        depth_file_count : Int32,
        total_edge_chars : Int64,
        corpus_token_count : Int32,
        vocab_size : Int32,
        corpus_hash : UInt64,
        tokenizer_tag : String,
      )
        path = File.join(@out_dir, "meta.bin")
        File.open(path, "wb") do |io|
          io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(RADIX_VERSION, IO::ByteFormat::LittleEndian)
          io.write_bytes(radix_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(depth_file_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(total_edge_chars.to_i64, IO::ByteFormat::LittleEndian)
          io.write_bytes(corpus_token_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(vocab_size.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(corpus_hash, IO::ByteFormat::LittleEndian)
          tag_bytes = tokenizer_tag.to_slice
          io.write_bytes(tag_bytes.size.to_i32, IO::ByteFormat::LittleEndian)
          io.write tag_bytes
        end
      end
    end
  end
end
