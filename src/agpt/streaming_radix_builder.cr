module MicroGPT
  module AGPT
    # Builds a radix-compressed trie index from an existing leveled trie index.
    #
    # A radix node is a branching endpoint (or leaf) that owns an incoming edge
    # of L characters. Runs of unary nodes (nodes with entry_count==1 in the
    # leveled trie) are collapsed into a single edge.
    #
    # In a leveled trie node N:
    #   entry_count == 0 : no next-token observations (end-of-corpus leaf) → drop
    #   entry_count == 1 : unary, exactly one child → extend the edge
    #   entry_count >= 2 : branching → emit radix node
    #
    # Binary format (radix_depth_NNN.bin where NNN is ENDPOINT character depth):
    #   magic (u32 LEVELED_MAGIC)
    #   depth (i32)
    #   record_count (i32)
    #   per record:
    #     radix_id         (i32)
    #     parent_radix_id  (i32)
    #     first_char_depth (i32)
    #     edge_len         (i32) — L
    #     edge_tokens[L]   (i32 × L)
    #     entry_count      (i32)
    #     entries[]        : (token_id i32, count i32)
    class StreamingRadixBuilder
      RADIX_MAGIC   = 0x52445841_u32  # 'RDXA'
      # v2 adds edge_mass (sum of counts at the FIRST original-trie node in the edge),
      # used for corpus-mass-weighted training. Mass is preserved along pure unary
      # chains (no branching, no truncation), so the head count equals the true
      # prefix frequency — avoids truncation-reduced endpoint counts.
      RADIX_VERSION = 2_i32

      # per_subtree: when true, emit one file per subtree (radix_subtree_NNNNNN.bin)
      # + a manifest.bin listing them. Enables per-subtree loading for memory scaling
      # at large depths where a global KV cache would exceed available memory.
      #
      # subtree_level: depth of the key prefix that defines a subtree.
      #   1 = one subtree per first-character (unigram, ≤65 subtrees) — default.
      #   2 = one subtree per (first, second) character pair (bigram, ≤65² subtrees) —
      #       attacks the "one dominant root-child" problem at deep tries (e.g. space
      #       is 30% of Shakespeare corpus, its d=32 subtree alone is 4 GB KV). Bigram
      #       splits that single subtree by the second character, bringing peak down to
      #       ~1 GB and unlocking d=48+.
      #
      # KNOWN APPROXIMATION (bigram): when a root-child has edge_len==1 AND has
      # descendants, those descendants land in bigram subtrees that do NOT contain
      # the root-child (it lives in its own "(first,255)" singleton group). During
      # training, each descendant's ancestor chain is truncated — the first char's
      # KV is not available in the subtree's local KV cache, so attention at
      # depth ≥ 2 proceeds without that context. In practice this mostly affects
      # the high-frequency root-children (space, 'e', 't') whose edges are usually
      # exactly 1 char. The token embedding still encodes first-char identity, so
      # loss is partial. If quality regresses vs unigram at matched compute, the
      # fix is to duplicate the root-child as a context-only node (entry_count=0)
      # in each of its bigram subtree files so the ancestor chain resolves locally.
      # prune_min_mass: drop radix edges whose head-of-edge prefix count is below
      # this threshold. Mass=1 paths (paths that appear exactly once in the corpus)
      # contribute essentially zero gradient signal in count-weighted loss, but
      # consume KV-cache memory proportionally to their char positions. Pruning
      # these is nearly free in loss quality and dramatic for memory at deep d
      # — at d=128 typically 90%+ of paths have mass=1. Default 1 (keep everything).
      #
      # prune_min_depth: never prune at depths shallower than this (preserve
      # vocabulary coverage). Default 4.
      def initialize(
        @reader : LeveledTrieReader,
        @out_dir : String,
        @progress : Bool = true,
        @per_subtree : Bool = false,
        @subtree_level : Int32 = 1,
        @prune_min_mass : Int32 = 1,
        @prune_min_depth : Int32 = 4
      )
      end

      # DepthStream: open output file for a given endpoint depth, track
      # record count, fix header at close. We write records as they come
      # (streaming) instead of buffering in memory — critical for d≥64 where
      # the in-memory buffer would exceed available RAM.
      private class DepthStream
        property io : File
        property count : Int32
        def initialize(@io : File)
          @count = 0
        end
      end

      # SubtreeStream: per-subtree output file. Tracks record_count,
      # total_edge_chars, max_endpoint_depth for fix-up at close.
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

      def build : NamedTuple(radix_count: Int32, total_edge_chars: Int64, max_endpoint_depth: Int32)
        Dir.mkdir_p(@out_dir)
        Dir.mkdir_p(File.join(@out_dir, "subtrees")) if @per_subtree

        build_started = Time.instant

        # parent-id → children index, one hash per depth. At d=64 the full
        # set of 64 per-depth indexes doesn't fit in RAM on a 16 GB machine.
        # Cap to an LRU of MAX_CACHED_INDEX depths.
        children_by_depth = {} of Int32 => Hash(Int32, Array(LeveledTrieReader::LoadedRecord))
        idx_lru = Deque(Int32).new

        # Frontier: edge-start points to explore.
        # Each entry: (parent_radix_id, starting_original_id, starting_char_depth)
        # starting_original_id is the PARENT node in the original trie; we iterate
        # its children_at_{starting_char_depth}.
        frontier = Deque({Int32, Int32, Int32}).new
        frontier << {0, 0, 1}   # from virtual root, children at depth 1

        # Streaming writers: one open File per endpoint depth (and per subtree
        # when --per-subtree is on). Opened lazily when the first record for
        # that depth/subtree arrives. Headers have placeholder counts that get
        # patched at close time.
        depth_streams   = {} of Int32 => DepthStream
        subtree_streams = {} of Int32 => SubtreeStream

        # subtree_of[radix_id] = integer key identifying the subtree this node belongs to.
        #   subtree_level = 1 (unigram): key = the depth-1 radix ancestor's radix_id.
        #   subtree_level = 2 (bigram):  key = first_char * 256 + second_char, where 255 is
        #                                the "no second char yet" sentinel for root-children
        #                                with edge_len=1 and no descendants.
        subtree_of = {} of Int32 => Int32
        next_radix_id = 1_i32  # 0 reserved for virtual root
        radix_count = 1        # count the virtual root
        total_edge_chars = 0_i64
        max_endpoint_depth = 0

        while !frontier.empty?
          parent_radix_id, start_original_id, start_char_depth = frontier.shift
          next if start_char_depth >= @reader.depth_file_count

          children_idx = children_index_for(children_by_depth, idx_lru, start_char_depth)
          children = children_idx[start_original_id]?
          next if children.nil?

          children.each do |child|
            edge = [child.token]
            current = child
            current_depth = start_char_depth
            # Edge mass = sum of counts at the FIRST node of the edge. In a pure
            # unary chain, mass is preserved through every intermediate position,
            # so the head count is the true prefix frequency (not a truncation-
            # reduced endpoint count).
            head_counts = @reader.counts_of(child.id)
            edge_mass = head_counts.sum(0) { |t| t[1] }

            # Extend while unary
            loop do
              cnts = @reader.counts_of(current.id)
              if cnts.size == 1 && current_depth + 1 < @reader.depth_file_count
                # Unary: find the single child
                next_children_idx = children_index_for(children_by_depth, idx_lru, current_depth + 1)
                next_children = next_children_idx[current.id]?
                break if next_children.nil? || next_children.size != 1
                current = next_children[0]
                current_depth += 1
                edge << current.token
              else
                break
              end
            end

            # At this point, `current` is the endpoint. Its counts decide action.
            endpoint_counts = @reader.counts_of(current.id)
            if endpoint_counts.empty?
              # End-of-corpus leaf: drop entirely (no training signal, no descendants).
              next
            end

            # Frequency pruning.
            #
            # Intuition for what this actually prunes: in the trie's interior,
            # mass=1 paths are always unary-all-the-way (integer counts summing
            # to 1 allow only size=0 or size=1 continuations), so radix already
            # absorbs them and they drop at EOF. The mass=1 emissions we see
            # past this point come from the *max-depth boundary* of the leveled
            # trie — depth-D nodes whose counts point to a depth-(D+1) successor
            # that was never stored because we capped the trie.
            #
            # Dropping those boundary tails past prune_min_depth saves KV memory
            # dramatically at deep d (at Shakespeare d=32, ~23× reduction) in
            # exchange for real PPL cost on small corpora (~3 at d=32). The
            # trade flips at scale: on a 1B-token corpus the long tail of
            # mass=1 paths is essentially noise, and pruning them is near-free.
            if edge_mass < @prune_min_mass && start_char_depth >= @prune_min_depth
              next
            end

            # Emit radix node
            radix_id = next_radix_id
            next_radix_id += 1
            radix_count += 1
            total_edge_chars += edge.size

            endpoint_depth = current_depth
            if endpoint_depth > max_endpoint_depth
              max_endpoint_depth = endpoint_depth
            end

            # Determine the subtree key for this new radix node.
            #   subtree_level = 1 (unigram): key = depth-1 radix ancestor's radix_id.
            #   subtree_level = 2 (bigram):  key = first_char * 256 + second_char.
            subtree_key : Int32
            if @subtree_level == 2
              if parent_radix_id == 0
                # This IS a root-child. First char = edge[0]; second char = edge[1] if
                # the edge is ≥2 chars (or 255 sentinel until a descendant defines it).
                first_c  = edge[0]
                second_c = edge.size >= 2 ? edge[1] : 255
                subtree_key = (first_c.to_i32 << 8) | second_c.to_i32
              else
                parent_key = subtree_of[parent_radix_id]
                parent_first  = (parent_key >> 8) & 0xff
                parent_second = parent_key & 0xff
                if parent_second != 255
                  # Bigram already determined up the chain — inherit.
                  subtree_key = parent_key
                else
                  # Parent was a root-child with edge_len=1 and hadn't seen a second
                  # char yet. This node's edge[0] is the second character of the path.
                  subtree_key = (parent_first.to_i32 << 8) | edge[0].to_i32
                end
              end
            else
              # Unigram: the depth-1 ancestor is the key.
              subtree_key = parent_radix_id == 0 ? radix_id : subtree_of[parent_radix_id]
            end
            subtree_of[radix_id] = subtree_key

            # Stream this record straight to disk instead of buffering.
            # Opens the per-depth (and per-subtree) file lazily on first write,
            # with a placeholder record_count that we patch at close time.
            dstream = depth_streams[endpoint_depth]?
            if dstream.nil?
              path = File.join(@out_dir, "radix_depth_#{"%03d" % endpoint_depth}.bin")
              io = File.open(path, "wb")
              io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
              io.write_bytes(endpoint_depth.to_i32, IO::ByteFormat::LittleEndian)
              io.write_bytes(0_i32, IO::ByteFormat::LittleEndian)  # placeholder for record_count
              dstream = DepthStream.new(io)
              depth_streams[endpoint_depth] = dstream
            end
            write_record(dstream.io, radix_id, parent_radix_id, start_char_depth, edge, edge_mass, endpoint_counts)
            dstream.count += 1

            if @per_subtree
              sstream = subtree_streams[subtree_key]?
              if sstream.nil?
                dir = File.join(@out_dir, "subtrees")
                spath = File.join(dir, "radix_subtree_#{"%06d" % subtree_key}.bin")
                sio = File.open(spath, "wb")
                sio.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
                sio.write_bytes(RADIX_VERSION, IO::ByteFormat::LittleEndian)
                sio.write_bytes(subtree_key.to_i32, IO::ByteFormat::LittleEndian)
                sio.write_bytes(0_i32, IO::ByteFormat::LittleEndian)   # placeholder record_count
                sio.write_bytes(0_i64, IO::ByteFormat::LittleEndian)   # placeholder total_edge_chars
                sio.write_bytes(0_i32, IO::ByteFormat::LittleEndian)   # placeholder max_endpoint_depth
                sstream = SubtreeStream.new(sio)
                subtree_streams[subtree_key] = sstream
              end
              write_record(sstream.io, radix_id, parent_radix_id, start_char_depth, edge, edge_mass, endpoint_counts)
              sstream.count += 1
              sstream.total_edge_chars += edge.size.to_i64
              sstream.max_ep = endpoint_depth if endpoint_depth > sstream.max_ep
            end

            # Descend: queue each branching child as a new edge-start
            if endpoint_counts.size >= 2
              frontier << {radix_id, current.id, current_depth + 1}
            end
          end

          if @progress && radix_count % 10_000 == 0
            elapsed = (Time.instant - build_started).total_seconds
            STDERR.puts "[radix] #{radix_count} radix nodes, frontier=#{frontier.size}, max_ep_depth=#{max_endpoint_depth}, cached_idx=#{children_by_depth.size}  (#{elapsed.round(1)}s)"
          end
        end

        # Close per-depth streams, patching record_count header in each.
        depth_streams.each do |_, stream|
          stream.io.seek(8)    # after magic(4) + depth(4)
          stream.io.write_bytes(stream.count.to_i32, IO::ByteFormat::LittleEndian)
          stream.io.close
        end
        # Parity with non-streaming builder: the original code wrote one file
        # per depth in 0..max_endpoint_depth, even if the depth had no records
        # (e.g. depth 0 where only the virtual root lives). Write empty files
        # for any depth that didn't open a stream.
        (0..max_endpoint_depth).each do |d|
          next if depth_streams.has_key?(d)
          path = File.join(@out_dir, "radix_depth_#{"%03d" % d}.bin")
          File.open(path, "wb") do |io|
            io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
            io.write_bytes(d.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(0_i32, IO::ByteFormat::LittleEndian)  # record_count = 0
          end
        end

        # Close per-subtree streams and build manifest from their stats.
        if @per_subtree
          manifest = [] of {Int32, Int32, Int64, Int32}
          subtree_streams.each do |rc, stream|
            # Header: magic(4) + version(4) + root_child(4) = offset 12
            # Then record_count(4) + total_edge_chars(8) + max_endpoint_depth(4).
            stream.io.seek(12)
            stream.io.write_bytes(stream.count.to_i32, IO::ByteFormat::LittleEndian)
            stream.io.write_bytes(stream.total_edge_chars, IO::ByteFormat::LittleEndian)
            stream.io.write_bytes(stream.max_ep.to_i32, IO::ByteFormat::LittleEndian)
            stream.io.close
            manifest << {rc, stream.count, stream.total_edge_chars, stream.max_ep}
          end
          manifest.sort_by! { |m| m[0] }
          write_manifest(manifest)
          STDERR.puts "[radix] per-subtree: #{manifest.size} subtree files written"
        end

        write_meta(
          radix_count,
          max_endpoint_depth + 1,
          total_edge_chars,
          @reader.index_metadata.corpus_token_count,
          @reader.index_metadata.vocab_size,
          @reader.index_metadata.corpus_hash,
          @reader.index_metadata.tokenizer_tag
        )

        elapsed = (Time.instant - build_started).total_seconds
        STDERR.puts "[radix] done: #{radix_count} radix nodes, #{total_edge_chars} total edge chars, #{max_endpoint_depth + 1} endpoint depths, #{elapsed.round(2)}s"

        {radix_count: radix_count, total_edge_chars: total_edge_chars, max_endpoint_depth: max_endpoint_depth}
      end

      # Max depths to keep in children_by_depth simultaneously. The builder's
      # BFS walks the same depth sequence for each root-child's unary chain
      # extension (depths 1..N), so the cache needs to be big enough to hold
      # a root-child's working set without thrashing. Anything ≥ the
      # characteristic extension length works. At d=8 extensions go 1..8;
      # at d=64 they go 1..~32 typical. 64 covers both cases while capping
      # RAM at roughly 64 × (per-depth index size).
      MAX_CACHED_INDEX = 64

      private def children_index_for(
        cache : Hash(Int32, Hash(Int32, Array(LeveledTrieReader::LoadedRecord))),
        lru : Deque(Int32),
        depth : Int32
      ) : Hash(Int32, Array(LeveledTrieReader::LoadedRecord))
        if existing = cache[depth]?
          # Move to MRU position
          lru.delete(depth)
          lru << depth
          return existing
        end
        # Build the index
        idx = {} of Int32 => Array(LeveledTrieReader::LoadedRecord)
        @reader.nodes_at_depth(depth).each do |rec|
          arr = idx[rec.parent_id]?
          if arr
            arr << rec
          else
            idx[rec.parent_id] = [rec]
          end
        end
        cache[depth] = idx
        lru << depth
        # Evict oldest while over capacity
        while lru.size > MAX_CACHED_INDEX
          evict = lru.shift
          cache.delete(evict)
        end
        idx
      end

      # Emit one record (same wire format for per-depth and per-subtree files).
      # Called while streaming; the file's header record_count is patched at
      # close time by the builder.
      private def write_record(
        io : IO,
        radix_id : Int32,
        parent_radix_id : Int32,
        first_char_depth : Int32,
        edge : Array(Int32),
        edge_mass : Int32,
        entries : Array({Int32, Int32})
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

      # Manifest: ordered list of subtree files.
      #   magic, version, n_subtrees
      #   per entry: root_child_id (i32), n_nodes (i32), total_edge_chars (i64), max_endpoint_depth (i32)
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
        tokenizer_tag : String
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
