require "microgpt"
require "./agpt/trie_node"
require "./agpt/trie_accessor"
require "./agpt/trie_corpus"
require "./agpt/leveled_trie_reader"
require "./agpt/weighted_loss"
require "./agpt/trainer"
require "./agpt/kv_cache"
require "./agpt/node_state"
require "./agpt/incremental_forward"
require "./agpt/incremental_backward"
require "./agpt/node_kv_store"
require "./agpt/batched_depth_forward"
require "./agpt/batched_depth_backward"
require "./agpt/trie_walk_trainer"
require "./agpt/leveled_trie_walk_trainer"
require "./agpt/streaming_leveled_builder"
require "./agpt/streaming_radix_builder"
require "./agpt/radix_trie_reader"

module MicroGPT
  module AGPT
    VERSION = "0.1.0-dev"

    # Identifies how a corpus was tokenized when building a trie. Embedded
    # into trie metadata so a loader can detect tokenizer drift.
    TOKENIZER_TAG = "char.sorted_unique.v1"
  end
end
