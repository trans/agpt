require "./radix_trie_reader"

module MicroGPT
  module AGPT
    # CorpusTrieWalker — shared substrate for sliding-window walks of a
    # corpus through a radix-compressed trie.
    #
    # Storage is compact slice-based instead of hashmaps so we can handle
    # 7M+ node tries (Gutenberg-scale) without OOMing a 15 GB laptop.
    #
    # On a 7M-node trie the footprint is roughly:
    #   parent_ids       : Int32 × N            ≈ 28 MB
    #   endpoint_depths  : Int32 × N            ≈ 28 MB
    #   edge_masses      : Int32 × N            ≈ 28 MB
    #   edge_token_offs  : Int32 × (N+1)        ≈ 28 MB
    #   edge_tokens_flat : Int32 × total_chars  ≈ 140 MB
    #   child_index      : UInt64 → Int32 hash  ≈ 170 MB
    #                                           total ≈ 420 MB
    #
    # Public API:
    #   - `new(reader, corpus_tokens)`           — load all records into slices
    #   - `walk { |radix_id, start_pos, end_pos| ... }` — outer walk callback
    #   - `d_max : Int32`                         — derived from trie
    #   - `fall_off_count : Int64`                — set after walk
    #   - `no_root_child_count : Int64`           — set after walk
    #   - `radix_count : Int32`                   — total node count
    #   - per-id accessors: `parent_id_of`, `endpoint_depth_of`, `edge_mass_of`,
    #     `edge_tokens_of` (returns a Slice(Int32) view)
    #   - `child_of(parent_id, first_token) : Int32?`
    #   - `record(id) : LoadedRecord`             — constructs on demand (slow)
    class CorpusTrieWalker
      getter d_max : Int32
      getter radix_count : Int32
      getter fall_off_count : Int64
      getter no_root_child_count : Int64

      @parent_ids : Slice(Int32)
      @endpoint_depths : Slice(Int32)
      @edge_masses : Slice(Int32)
      @edge_token_offs : Slice(Int32)
      @edge_tokens_flat : Slice(Int32)

      # (parent_id << 32) | first_token  →  child_id
      @child_index : Hash(UInt64, Int32)

      @corpus_tokens : Array(Int32)

      def initialize(reader : RadixTrieReader, @corpus_tokens : Array(Int32))
        @radix_count = reader.radix_count
        @fall_off_count = 0_i64
        @no_root_child_count = 0_i64

        n = @radix_count
        total_chars = reader.total_edge_chars

        @parent_ids = Slice(Int32).new(n, 0)
        @endpoint_depths = Slice(Int32).new(n, 0)
        @edge_masses = Slice(Int32).new(n, 0)
        @edge_token_offs = Slice(Int32).new(n + 1, 0)
        @edge_tokens_flat = Slice(Int32).new(total_chars.to_i32, 0)

        # Pre-size hash to avoid rehashing for 7M+ entries.
        @child_index = Hash(UInt64, Int32).new(initial_capacity: n)

        # Pass 1: populate scalar fields per id, record edge_len per id,
        # populate child_index. We can't lay out edge_tokens_flat yet
        # because reader.each yields in depth-file order, not id order,
        # so flat offsets must come from a prefix-sum of edge_len[id]
        # after all records are seen.
        edge_lens = Slice(Int32).new(n, 0)
        d_max = 0
        reader.each do |r|
          rid = r.id
          @parent_ids[rid] = r.parent_id
          ep = r.endpoint_depth
          @endpoint_depths[rid] = ep
          @edge_masses[rid] = r.edge_mass
          edge = r.edge_tokens
          edge_lens[rid] = edge.size
          d_max = ep if ep > d_max
          first_tok = edge[0]
          key = (r.parent_id.to_u64 << 32) | (first_tok.to_u32.to_u64)
          @child_index[key] = rid
        end

        # Prefix-sum edge_lens → edge_token_offs.
        acc = 0
        n.times do |i|
          @edge_token_offs[i] = acc
          acc += edge_lens[i]
        end
        @edge_token_offs[n] = acc

        # Pass 2: copy edge tokens into the flat array at their id-keyed offsets.
        reader.each do |r|
          rid = r.id
          off = @edge_token_offs[rid]
          r.edge_tokens.each_with_index do |t, i|
            @edge_tokens_flat[off + i] = t
          end
        end

        @d_max = d_max
      end

      def parent_id_of(id : Int32) : Int32
        @parent_ids[id]
      end

      def endpoint_depth_of(id : Int32) : Int32
        @endpoint_depths[id]
      end

      def edge_mass_of(id : Int32) : Int32
        @edge_masses[id]
      end

      # Returns a Slice view into the flat edge-token store. Length is
      # `edge_token_offs[id+1] - edge_token_offs[id]`. Zero-copy.
      def edge_tokens_of(id : Int32) : Slice(Int32)
        off = @edge_token_offs[id]
        len = @edge_token_offs[id + 1] - off
        @edge_tokens_flat[off, len]
      end

      def edge_len_of(id : Int32) : Int32
        @edge_token_offs[id + 1] - @edge_token_offs[id]
      end

      def child_of(parent_id : Int32, first_token : Int32) : Int32?
        key = (parent_id.to_u64 << 32) | (first_token.to_u32.to_u64)
        @child_index[key]?
      end

      # Construct a LoadedRecord on demand. Slightly slower than direct
      # slice access; intended for low-frequency callers that want the old
      # struct shape. `counts` is not stored by the walker (would double
      # memory) and is returned empty — callers that need it should query
      # the underlying RadixTrieReader directly.
      def record(id : Int32) : RadixTrieReader::LoadedRecord
        edge = edge_tokens_of(id).to_a
        first_char_depth = @endpoint_depths[id] - edge.size + 1
        RadixTrieReader::LoadedRecord.new(
          id, @parent_ids[id], first_char_depth, edge, @edge_masses[id],
          [] of {Int32, Int32}
        )
      end

      # Iterate every radix id in [0, radix_count).
      def each_id(&block : Int32 ->)
        @radix_count.times { |i| yield i }
      end

      # Bytes resident in the walker's compact storage (excluding the
      # child-index hash, which Crystal doesn't expose a byte-size for).
      def approximate_bytes : Int64
        b = 0_i64
        b += @parent_ids.bytesize.to_i64
        b += @endpoint_depths.bytesize.to_i64
        b += @edge_masses.bytesize.to_i64
        b += @edge_token_offs.bytesize.to_i64
        b += @edge_tokens_flat.bytesize.to_i64
        b
      end

      # Walk the corpus from every starting position s in [0, start_count),
      # following the trie up to d_max characters. Invokes the block for
      # every node landing with:
      #   (radix_id, start_corpus_pos, terminal_corpus_pos)
      #
      # `start_count` lets callers append wrap-around lookahead tokens for
      # matching while still counting contributions only from original starts.
      #
      # Updates `fall_off_count` and `no_root_child_count` as a side
      # effect. Block-form to let callers stream contributions without
      # an intermediate buffer.
      def walk(start_count : Int32? = nil, &block : Int32, Int32, Int32 ->)
        n_corpus = @corpus_tokens.size
        n_starts = start_count || n_corpus
        n_starts = n_corpus if n_starts > n_corpus
        dmax = @d_max
        fall_off = 0_i64
        no_root = 0_i64
        s = 0
        while s < n_starts
          parent_id = 0
          pos_off = 0
          while pos_off < dmax && (s + pos_off) < n_corpus
            next_char = @corpus_tokens[s + pos_off]
            kid = child_of(parent_id, next_char)
            if kid.nil?
              no_root += 1 if parent_id == 0
              break
            end
            kid_off = @edge_token_offs[kid]
            kid_end = @edge_token_offs[kid + 1]
            edge_len = kid_end - kid_off
            remaining_d = dmax - pos_off
            remaining_n = n_corpus - (s + pos_off)
            max_can = edge_len
            max_can = remaining_d if remaining_d < max_can
            max_can = remaining_n if remaining_n < max_can

            match_len = 0
            i = 0
            base = s + pos_off
            while i < max_can
              if @edge_tokens_flat[kid_off + i] == @corpus_tokens[base + i]
                match_len += 1
                i += 1
              else
                break
              end
            end

            if match_len > 0
              terminal_pos = s + pos_off + match_len - 1
              yield kid, s, terminal_pos
            end

            pos_off += match_len
            if match_len < edge_len
              fall_off += 1
              break
            end
            parent_id = kid
          end
          s += 1
        end
        @fall_off_count = fall_off
        @no_root_child_count = no_root
      end
    end
  end
end
