require "./spec_helper"

# Reference spec for unary chain compression.
#
# The current AGPT forward processes each trie node individually at its depth
# level. Chain compression will process each unary chain as a single window-style
# batched forward. For correctness, the new path must produce per-node outputs
# (logits, final_x, K/V per layer) that match the current path to float
# tolerance on the CPU backend.
#
# This spec serves as the target contract for the chain forward implementation:
# it runs the current path over a deterministic small trie, records each
# observed node's outputs, and asserts any new implementation reproduces them.

private def fill_mat!(m : MicroGPT::Mat, scale : Float32, offset : Int32 = 0)
  m.rows.times do |r|
    m.cols.times do |c|
      value = (((offset + r * m.cols + c) % 37) + 1).to_f32 * scale
      m[r, c] = value
    end
  end
end

private def make_deterministic_model : MicroGPT::MiniGPT
  config = MicroGPT::Config.new
  config.d_model = 8
  config.n_heads = 2
  config.n_layers = 1
  config.d_ff = 16
  config.vocab_size = 8
  config.seq_len = 8
  config.learning_rate = 1e-3

  model = MicroGPT::MiniGPT.new(config, [4, 4])
  fill_mat!(model.embedding.token_emb, 0.01_f32, 1)

  block = model.blocks.first
  fill_mat!(block.ln1.gamma, 0.02_f32, 2)
  fill_mat!(block.ln1.beta, 0.01_f32, 3)
  fill_mat!(block.ln2.gamma, 0.02_f32, 4)
  fill_mat!(block.ln2.beta, 0.01_f32, 5)

  fill_mat!(block.attn.wq.w, 0.01_f32, 6)
  fill_mat!(block.attn.wq.b, 0.01_f32, 7)
  fill_mat!(block.attn.wk.w, 0.01_f32, 8)
  fill_mat!(block.attn.wk.b, 0.01_f32, 9)
  fill_mat!(block.attn.wv.w, 0.01_f32, 10)
  fill_mat!(block.attn.wv.b, 0.01_f32, 11)
  fill_mat!(block.attn.wo.w, 0.01_f32, 12)
  fill_mat!(block.attn.wo.b, 0.01_f32, 13)

  fill_mat!(block.ff.l1.w, 0.01_f32, 14)
  fill_mat!(block.ff.l1.b, 0.01_f32, 15)
  fill_mat!(block.ff.l2.w, 0.01_f32, 16)
  fill_mat!(block.ff.l2.b, 0.01_f32, 17)

  fill_mat!(model.final_norm.gamma, 0.02_f32, 18)
  fill_mat!(model.final_norm.beta, 0.01_f32, 19)
  fill_mat!(model.output.proj.w, 0.01_f32, 20)
  fill_mat!(model.output.proj.b, 0.01_f32, 21)

  model
end

# Node fingerprint for equivalence checking.
# We snapshot:
#   - logits (exposed per-node prediction surface)
#   - final_x (post-block hidden state, pre-final-norm)
#   - attn K/V per layer per head (what gets stored in NodeKVStore)
private record NodeSnapshot,
  logits : Array(Float32),
  final_x : Array(Float32),
  k_rows : Array(Array(Float32)),  # per-head, post-RoPE K row
  v_rows : Array(Array(Float32))   # per-head, post-RoPE V row

# Run the current depth-based forward over a small trie and snapshot per-node
# outputs keyed by node_id.
private def reference_snapshot(model : MicroGPT::MiniGPT) : Hash(Int32, NodeSnapshot)
  token_ids = [0, 1, 2, 3, 4, 5, 6, 7] of Int32
  corpus = MicroGPT::AGPT::TrieCorpus.from_token_ids(
    token_ids, max_depth: 6, max_starts: 1, vocab_size: 8
  )
  kv_store = MicroGPT::AGPT::NodeKVStore.new
  node_ancestor_ids = {corpus.root.id => [] of Int32}
  node_positions = {} of Int32 => Int32
  prev_caches = nil.as(Hash(Int32, Array(MicroGPT::AGPT::LayerKVCache))?)
  snapshots = {} of Int32 => NodeSnapshot
  head_dims = model.blocks.first.attn.head_dims
  n_layers = model.config.n_layers
  n_heads = head_dims.size

  corpus.each_depth_level do |depth, nodes|
    next if depth == 0
    eligible = [] of MicroGPT::AGPT::TrieNode
    nodes.each do |node|
      parent = node.parent.not_nil!
      next unless node_ancestor_ids.has_key?(parent.id)
      next if parent.depth >= model.config.seq_len
      eligible << node
    end
    next if eligible.empty?

    eligible.each do |node|
      node_positions[node.id] = depth - 1
    end

    results, this_caches = MicroGPT::AGPT::BatchedDepthForward.forward_depth(
      eligible, node_ancestor_ids, node_positions, kv_store, model, corpus, prev_caches
    )
    prev_caches = this_caches

    results.each do |result|
      logits = Array(Float32).new(result.logits.cols) { |j| result.logits[0, j] }
      final_x = Array(Float32).new(result.final_x.cols) { |j| result.final_x[0, j] }
      entry = kv_store.entries[result.node_id]
      k_rows = Array(Array(Float32)).new(n_layers * n_heads)
      v_rows = Array(Array(Float32)).new(n_layers * n_heads)
      n_layers.times do |li|
        n_heads.times do |hi|
          k, v = entry[li][hi]
          hd = head_dims[hi]
          k_rows << Array(Float32).new(hd) { |j| k[0, j] }
          v_rows << Array(Float32).new(hd) { |j| v[0, j] }
        end
      end
      snapshots[result.node_id] = NodeSnapshot.new(
        logits: logits, final_x: final_x, k_rows: k_rows, v_rows: v_rows
      )
    end
  end

  snapshots
end

private def assert_close(actual : Array(Float32), expected : Array(Float32), tol : Float32, label : String)
  actual.size.should eq expected.size
  actual.size.times do |i|
    diff = (actual[i] - expected[i]).abs
    if diff > tol
      fail "#{label}[#{i}]: actual=#{actual[i]} expected=#{expected[i]} diff=#{diff} > tol=#{tol}"
    end
  end
end

describe "AGPT unary chain compression reference" do
  it "produces self-consistent per-node snapshots from the current path" do
    MicroGPT.use_crystal!
    model = make_deterministic_model
    snap_a = reference_snapshot(model)
    snap_b = reference_snapshot(model)

    snap_a.size.should be > 0
    snap_a.size.should eq snap_b.size

    snap_a.each do |node_id, a|
      b = snap_b[node_id]
      assert_close(a.logits, b.logits, 1e-6_f32, "logits node=#{node_id}")
      assert_close(a.final_x, b.final_x, 1e-6_f32, "final_x node=#{node_id}")
      a.k_rows.each_with_index { |row, li| assert_close(row, b.k_rows[li], 1e-6_f32, "k node=#{node_id} head=#{li}") }
      a.v_rows.each_with_index { |row, li| assert_close(row, b.v_rows[li], 1e-6_f32, "v node=#{node_id} head=#{li}") }
    end
  end

  # When chain-compressed forward is implemented, uncomment this and point it at
  # the new path. The contract: same per-node outputs to float32 tolerance.
  pending "chain-compressed forward reproduces reference snapshot" do
    # MicroGPT.use_crystal!
    # model = make_deterministic_model
    # reference = reference_snapshot(model)
    # compressed = chain_compressed_snapshot(model)
    #
    # reference.size.should eq compressed.size
    # reference.each do |node_id, expected|
    #   actual = compressed[node_id]
    #   assert_close(actual.logits, expected.logits, 1e-4_f32, "logits node=#{node_id}")
    #   assert_close(actual.final_x, expected.final_x, 1e-5_f32, "final_x node=#{node_id}")
    #   expected.k_rows.each_with_index { |row, li|
    #     assert_close(actual.k_rows[li], row, 1e-5_f32, "k node=#{node_id} head=#{li}")
    #   }
    #   expected.v_rows.each_with_index { |row, li|
    #     assert_close(actual.v_rows[li], row, 1e-5_f32, "v node=#{node_id} head=#{li}")
    #   }
    # end
  end
end
