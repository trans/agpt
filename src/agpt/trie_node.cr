module MicroGPT
  module AGPT
    # Thin value-type façade over columnar storage in `TrieCorpus`.
    #
    # A `TrieNode` no longer owns its data — it is simply an `(corpus, id)`
    # handle. All getters forward into the parallel arrays on `TrieCorpus`.
    #
    # `next_token_counts[x]` records how often token `x` follows that prefix.
    struct TrieNode
      alias ChildEntry = {Int32, TrieNode}
      alias CountEntry = {Int32, Int32}

      getter id : Int32

      def initialize(@corpus : TrieCorpus, @id : Int32)
      end

      def parent : TrieNode?
        pid = @corpus.parent_id(@id)
        pid == -1 ? nil : TrieNode.new(@corpus, pid)
      end

      def token_id : Int32?
        tok = @corpus.token_id_of(@id)
        tok == -1 ? nil : tok
      end

      def depth : Int32
        @corpus.depth_of(@id)
      end

      def children : Array(ChildEntry)
        result = Array(ChildEntry).new
        @corpus.children_of(@id).each do |(tok, cid)|
          result << {tok, TrieNode.new(@corpus, cid)}
        end
        result
      end

      def next_token_counts : Array(CountEntry)
        @corpus.counts_of(@id)
      end

      def total_outgoing_mass : Int32
        @corpus.counts_of(@id).sum(0) { |(_, count)| count }
      end

      def terminal? : Bool
        @corpus.children_of(@id).empty? && @corpus.counts_of(@id).empty?
      end

      def observe(next_token : Int32)
        @corpus.observe(@id, next_token)
      end

      def child_for(token : Int32) : TrieNode?
        cid = @corpus.find_child(@id, token)
        cid == -1 ? nil : TrieNode.new(@corpus, cid)
      end

      def ensure_child(token : Int32, next_id : Int32) : TrieNode
        cid = @corpus.ensure_child_id(@id, token, next_id)
        TrieNode.new(@corpus, cid)
      end

      def next_token_counts_hash : Hash(Int32, Int32)
        h = {} of Int32 => Int32
        @corpus.counts_of(@id).each { |(tok, count)| h[tok] = count }
        h
      end

      def replace_next_token_counts(entries : Array(CountEntry))
        @corpus.replace_counts(@id, entries)
      end

      def ==(other : TrieNode) : Bool
        @id == other.id && @corpus.same?(other.@corpus)
      end

      def_hash @id
    end
  end
end
