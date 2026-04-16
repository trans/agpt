require "./spec_helper"
require "file_utils"

# Round-trip test for the per-depth trie serialization format (Phase A of
# disk-paged trainer). Builds a small trie, writes it via save_by_depth,
# reads it back with load_by_depth, and verifies every per-node field and
# observation count matches.

private def sample_tokens : Array(Int32)
  [0, 1, 2, 3, 0, 1, 4, 5, 0, 2, 6, 7, 1, 2, 3, 4] of Int32
end

private def build_sample_corpus : MicroGPT::AGPT::TrieCorpus
  MicroGPT::AGPT::TrieCorpus.from_token_ids(
    sample_tokens, max_depth: 5, max_starts: 6, vocab_size: 8
  )
end

describe "AGPT leveled trie save/load" do
  it "round-trips columnar fields and next-token counts" do
    original = build_sample_corpus

    tmpdir = File.join(Dir.tempdir, "agpt_leveled_#{Random.rand(UInt64)}")
    begin
      original.save_by_depth(tmpdir)
      reloaded = MicroGPT::AGPT::TrieCorpus.load_by_depth(tmpdir)

      reloaded.node_count.should eq original.node_count
      reloaded.max_depth.should eq original.max_depth
      reloaded.starts_used.should eq original.starts_used

      original.node_count.times do |id|
        reloaded.parent_id(id).should eq original.parent_id(id)
        reloaded.token_id_of(id).should eq original.token_id_of(id)
        reloaded.depth_of(id).should eq original.depth_of(id)

        node_orig = original.node_for_id(id)
        node_reload = reloaded.node_for_id(id)
        node_reload.children.size.should eq node_orig.children.size

        orig_counts = node_orig.next_token_counts_hash
        reload_counts = node_reload.next_token_counts_hash
        reload_counts.size.should eq orig_counts.size
        orig_counts.each do |tok, cnt|
          reload_counts[tok].should eq cnt
        end
      end

      # Verify children traversal produces identical (token, id) pairs
      original.node_count.times do |id|
        orig_children = original.children_of(id)
        reload_children = reloaded.children_of(id)
        reload_children.should eq orig_children
      end

      # Verify each_depth_level reconstructs the same structure
      orig_by_depth = {} of Int32 => Array(Int32)
      original.each_depth_level do |depth, nodes|
        orig_by_depth[depth] = nodes.map(&.id).sort
      end
      reload_by_depth = {} of Int32 => Array(Int32)
      reloaded.each_depth_level do |depth, nodes|
        reload_by_depth[depth] = nodes.map(&.id).sort
      end
      reload_by_depth.should eq orig_by_depth
    ensure
      FileUtils.rm_rf(tmpdir) if Dir.exists?(tmpdir)
    end
  end
end
