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

      # Compact char-trie for one subtree, struct-of-arrays. Each node is
      # 4 × Int32 (token, count, first_child id, next_sibling id) = 16 bytes
      # raw — vs ~150 B for a class+Hash node, a 10× memory reduction. Children
      # are stored as a singly-linked list per parent: first_child[parent]
      # points to head, next_sibling[child] chains the rest. Sentinel = -1.
      #
      # All operations are O(branching) at the parent. For typical char tries
      # at deep depths (mostly mass-1 unary), branching is 1, so lookups are
      # effectively O(1).
      private class CompactCharTrie
        getter tokens       : Array(Int32)
        getter counts       : Array(Int32)
        getter first_child  : Array(Int32)
        getter next_sibling : Array(Int32)

        def initialize
          @tokens       = [] of Int32
          @counts       = [] of Int32
          @first_child  = [] of Int32
          @next_sibling = [] of Int32
          # Allocate root at id=0. Token gets set by the caller (it represents
          # the subtree's depth-1 char).
          add_node(-1)
        end

        def size : Int32
          @tokens.size
        end

        def add_node(token : Int32) : Int32
          id = @tokens.size
          @tokens       << token
          @counts       << 0
          @first_child  << -1
          @next_sibling << -1
          id
        end

        # Find an existing child of `parent` matching `token`, or create one.
        # Returns the child's node id.
        def get_or_add_child(parent : Int32, token : Int32) : Int32
          cid = @first_child.unsafe_fetch(parent)
          while cid != -1
            if @tokens.unsafe_fetch(cid) == token
              return cid
            end
            cid = @next_sibling.unsafe_fetch(cid)
          end
          new_id = add_node(token)
          @next_sibling[new_id] = @first_child.unsafe_fetch(parent)
          @first_child[parent] = new_id
          new_id
        end

        # Returns the single child's id if `parent` has exactly one child,
        # otherwise -1. Cheap O(1) test (walks at most two list nodes).
        def single_child(parent : Int32) : Int32
          cid = @first_child.unsafe_fetch(parent)
          return -1 if cid == -1
          return -1 if @next_sibling.unsafe_fetch(cid) != -1
          cid
        end

        # Yield (token, child_id, count) for every child of `parent`.
        def each_child(parent : Int32, & : Int32, Int32, Int32 -> _) : Nil
          cid = @first_child.unsafe_fetch(parent)
          while cid != -1
            yield @tokens.unsafe_fetch(cid), cid, @counts.unsafe_fetch(cid)
            cid = @next_sibling.unsafe_fetch(cid)
          end
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
            root_node: 0,
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

          # The char_root reference goes out of scope at the next loop
          # iteration when it's rebound. Crystal GC reclaims the previous
          # subtree's compact trie before the next one's allocations grow.
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
      private def build_char_trie(starts : Array(Int32), root_char : Int32) : CompactCharTrie
        trie = CompactCharTrie.new
        # Root node id 0 = the subtree's depth-1 node (representing root_char).
        # Set its token explicitly (the constructor stub'd it as -1).
        trie.tokens[0] = root_char
        starts.each do |i|
          current_id = 0
          trie.counts[current_id] += 1
          # Insert chars at depths 2..max_depth+1 (positions i+1..i+max_depth).
          # See the StreamingRadixBuilder-equivalence note: the depth-
          # (max_depth+1) layer exists only to populate endpoint_counts at
          # the depth-max_depth radix nodes; no radix record is emitted at
          # depth > max_depth.
          last = i + @max_depth + 1
          last = @corpus_tokens.size if last > @corpus_tokens.size
          j = i + 1
          while j < last
            tok = @corpus_tokens.unsafe_fetch(j)
            current_id = trie.get_or_add_child(current_id, tok)
            trie.counts[current_id] += 1
            j += 1
          end
        end
        trie
      end

      # Walk the compact char trie and emit radix records. `root_node` is
      # the trie's node id for the head of an incoming edge. The unary chain
      # extends from `root_node` until we hit a branching point or the depth
      # cap. `head_token` is the token at the node's position (for the
      # subtree root it's the subtree's root_char; for descendants it's the
      # parent's children-link token).
      #
      # Returns updated counters in a NamedTuple.
      private def emit_subtree(
        trie : CompactCharTrie,
        root_node : Int32,
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
        # Iterative work-list. Each entry: (node_id, head_token, parent_radix_id, first_char_depth).
        work = Deque({Int32, Int32, Int32, Int32}).new
        work << {root_node, head_token, parent_radix_id, first_char_depth}

        while !work.empty?
          n_id, head_tok, p_id, fcd = work.shift

          current_id = n_id
          current_depth = fcd
          edge_tokens = [head_tok]

          edge_mass = trie.counts.unsafe_fetch(current_id)

          # Extend through unary chain
          while current_depth < @max_depth
            sc = trie.single_child(current_id)
            break if sc == -1
            current_id = sc
            current_depth += 1
            edge_tokens << trie.tokens.unsafe_fetch(current_id)
          end

          # Build endpoint_counts from this node's children.
          endpoint_counts = [] of {Int32, Int32}
          trie.each_child(current_id) do |tok, _cid, count|
            endpoint_counts << {tok, count}
          end

          # End-of-corpus tail: empty endpoint counts means we hit corpus end
          # at this prefix. No training signal. Drop.
          if endpoint_counts.empty?
            next
          end

          # Frequency pruning.
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
          if endpoint_counts.size >= 2
            trie.each_child(current_id) do |child_tok, child_id, _count|
              work << {child_id, child_tok, radix_id, current_depth + 1}
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
