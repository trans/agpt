module MicroGPT
  module AGPT
    # Minimal read-only corpus interface used by the hot forward/backward
    # paths (and KV store reconstruction). Both in-memory `TrieCorpus` and
    # the lazy `LeveledTrieReader` include this module, so callers that
    # type a parameter as `TrieAccessor` accept either backing store.
    #
    # Intentionally narrow: only the methods that truly need to be polymorphic
    # between in-memory and disk-paged access. Trainer-level iteration
    # (each_depth_level, etc.) is not here yet — it will be added when the
    # trainer migrates to optional lazy mode.
    module TrieAccessor
      abstract def parent_id(id : Int32) : Int32
      abstract def token_id_of(id : Int32) : Int32
      abstract def depth_of(id : Int32) : Int32
      # Next-token counts for the node: Array of {token_id, count}.
      # Empty when the node has no observed continuations (leaf or root).
      abstract def counts_of(id : Int32) : Array({Int32, Int32})
      # Number of direct children. Used to decide whether siblings share a
      # parent cache (>1 → deep_clone needed in cpu_per_node_attention).
      abstract def child_count_of(id : Int32) : Int32
    end
  end
end
