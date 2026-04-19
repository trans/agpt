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
      def initialize(
        @reader : LeveledTrieReader,
        @out_dir : String,
        @progress : Bool = true,
        @per_subtree : Bool = false,
        @subtree_level : Int32 = 1
      )
      end

      def build : NamedTuple(radix_count: Int32, total_edge_chars: Int64, max_endpoint_depth: Int32)
        Dir.mkdir_p(@out_dir)

        build_started = Time.instant

        # Build per-depth children indexes on demand: depth d+1 → parent_id → [records]
        # We lazily populate and keep only the currently-needed ones (small footprint).
        children_by_depth = {} of Int32 => Hash(Int32, Array(LeveledTrieReader::LoadedRecord))

        # Frontier: edge-start points to explore.
        # Each entry: (parent_radix_id, starting_original_id, starting_char_depth)
        # starting_original_id is the PARENT node in the original trie; we iterate
        # its children_at_{starting_char_depth}.
        frontier = Deque({Int32, Int32, Int32}).new
        frontier << {0, 0, 1}   # from virtual root, children at depth 1

        # Record tuple: {radix_id, parent_radix_id, first_char_depth, edge_tokens, edge_mass, endpoint_counts}
        radix_records_by_endpoint = {} of Int32 => Array({Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})})
        radix_records_by_subtree  = {} of Int32 => Array({Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})})
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

          children_idx = children_index_for(children_by_depth, start_char_depth)
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
                next_children_idx = children_index_for(children_by_depth, current_depth + 1)
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

            record = {radix_id, parent_radix_id, start_char_depth, edge, edge_mass, endpoint_counts}

            list = radix_records_by_endpoint[endpoint_depth]?
            if list.nil?
              list = [] of {Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})}
              radix_records_by_endpoint[endpoint_depth] = list
            end
            list << record

            if @per_subtree
              slist = radix_records_by_subtree[subtree_key]?
              if slist.nil?
                slist = [] of {Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})}
                radix_records_by_subtree[subtree_key] = slist
              end
              slist << record
            end

            # Descend: queue each branching child as a new edge-start
            if endpoint_counts.size >= 2
              frontier << {radix_id, current.id, current_depth + 1}
            end
          end

          # Keep children indexes resident for the full build — they are small
          # (~40 bytes/record) and eviction thrashes with the reader's own LRU.

          if @progress && radix_count % 10_000 == 0
            elapsed = (Time.instant - build_started).total_seconds
            STDERR.puts "[radix] #{radix_count} radix nodes, frontier=#{frontier.size}, max_ep_depth=#{max_endpoint_depth}  (#{elapsed.round(1)}s)"
          end
        end

        # Write per-endpoint-depth files
        (0..max_endpoint_depth).each do |d|
          records = radix_records_by_endpoint[d]? || ([] of {Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})})
          write_depth_file(d, records)
        end

        # Per-subtree output mode: one file per root-child, plus a manifest. This
        # is for scalable training at large depths — the trainer can load ONE
        # subtree at a time and scope its KV cache to just that subtree's
        # character positions, instead of needing a global KV cache that grows
        # with total corpus positions.
        if @per_subtree
          Dir.mkdir_p(File.join(@out_dir, "subtrees"))
          manifest = [] of {Int32, Int32, Int64, Int32}   # {root_child_id, n_nodes, total_edge_chars, max_endpoint_depth}
          radix_records_by_subtree.each do |rc, recs|
            # Sort by endpoint depth for BFS-order loading.
            recs.sort_by! { |r| r[2] + r[3].size - 1 }
            st_edge_chars = 0_i64
            st_max_ep = 0
            recs.each do |r|
              st_edge_chars += r[3].size.to_i64
              ep = r[2] + r[3].size - 1
              st_max_ep = ep if ep > st_max_ep
            end
            write_subtree_file(rc, recs, st_max_ep)
            manifest << {rc, recs.size, st_edge_chars, st_max_ep}
          end
          # Sort manifest by root_child id for deterministic order
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

      private def children_index_for(
        cache : Hash(Int32, Hash(Int32, Array(LeveledTrieReader::LoadedRecord))),
        depth : Int32
      ) : Hash(Int32, Array(LeveledTrieReader::LoadedRecord))
        if existing = cache[depth]?
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
        idx
      end

      private def write_depth_file(
        d : Int32,
        records : Array({Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})})
      )
        path = File.join(@out_dir, "radix_depth_#{"%03d" % d}.bin")
        File.open(path, "wb") do |io|
          io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(d.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(records.size.to_i32, IO::ByteFormat::LittleEndian)
          records.each do |(radix_id, parent_radix_id, first_char_depth, edge, edge_mass, entries)|
            io.write_bytes(radix_id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(parent_radix_id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(first_char_depth.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(edge.size.to_i32, IO::ByteFormat::LittleEndian)
            edge.each { |tok| io.write_bytes(tok.to_i32, IO::ByteFormat::LittleEndian) }
            io.write_bytes(edge_mass.to_i32, IO::ByteFormat::LittleEndian) # v2: prefix mass
            io.write_bytes(entries.size.to_i32, IO::ByteFormat::LittleEndian)
            entries.each do |(token_id, count)|
              io.write_bytes(token_id.to_i32, IO::ByteFormat::LittleEndian)
              io.write_bytes(count.to_i32, IO::ByteFormat::LittleEndian)
            end
          end
        end
      end

      # Per-subtree file: one self-contained file per root-child subtree.
      # Header:
      #   magic (u32 RADIX_MAGIC)
      #   version (i32) — matches radix format version
      #   root_child_id (i32)
      #   record_count (i32)
      #   total_edge_chars (i64)
      #   max_endpoint_depth (i32)
      # Records: same layout as per-endpoint-depth files (radix_id, parent, fcd, edge_len, edge_tokens[], edge_mass, entry_count, entries[]).
      private def write_subtree_file(
        root_child_id : Int32,
        records : Array({Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})}),
        max_endpoint_depth : Int32
      )
        dir = File.join(@out_dir, "subtrees")
        path = File.join(dir, "radix_subtree_#{"%06d" % root_child_id}.bin")
        total_edge_chars = 0_i64
        records.each { |r| total_edge_chars += r[3].size.to_i64 }
        File.open(path, "wb") do |io|
          io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(RADIX_VERSION, IO::ByteFormat::LittleEndian)
          io.write_bytes(root_child_id.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(records.size.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(total_edge_chars.to_i64, IO::ByteFormat::LittleEndian)
          io.write_bytes(max_endpoint_depth.to_i32, IO::ByteFormat::LittleEndian)
          records.each do |(radix_id, parent_radix_id, first_char_depth, edge, edge_mass, entries)|
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
