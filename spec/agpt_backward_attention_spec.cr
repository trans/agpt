require "./spec_helper"

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

private def build_results_by_depth(model : MicroGPT::MiniGPT) : Hash(Int32, {MicroGPT::AGPT::BatchedDepthForward::NodeResult, Hash(Int32, Array(MicroGPT::AGPT::LayerKVCache))})
  token_ids = [0, 1, 2, 3, 4, 5, 6, 7] of Int32
  corpus = MicroGPT::AGPT::TrieCorpus.from_token_ids(token_ids, max_depth: 6, max_starts: 1, vocab_size: 8)
  kv_store = MicroGPT::AGPT::NodeKVStore.new
  node_ancestor_ids = {corpus.root.id => [] of Int32}
  node_positions = {} of Int32 => Int32
  node_root_child = {} of Int32 => Int32
  prev_caches = nil.as(Hash(Int32, Array(MicroGPT::AGPT::LayerKVCache))?)
  results_by_depth = {} of Int32 => {MicroGPT::AGPT::BatchedDepthForward::NodeResult, Hash(Int32, Array(MicroGPT::AGPT::LayerKVCache))}

  corpus.each_depth_level do |depth, nodes|
    next if depth == 0

    eligible = [] of MicroGPT::AGPT::BatchedDepthForward::NodeProxy
    nodes.each do |node|
      parent_id = corpus.parent_id(node.id)
      next unless node_ancestor_ids.has_key?(parent_id)
      next if corpus.depth_of(parent_id) >= model.config.seq_len
      eligible << MicroGPT::AGPT::BatchedDepthForward::NodeProxy.new(node.id, node.token_id.not_nil!, node.depth)
    end
    next if eligible.empty?

    eligible.each do |node|
      node_positions[node.id] = depth - 1
      if depth == 1
        node_root_child[node.id] = node.id
      else
        node_root_child[node.id] = node_root_child[corpus.parent_id(node.id)]
      end
    end

    results, this_caches = MicroGPT::AGPT::BatchedDepthForward.forward_depth(
      eligible, node_ancestor_ids, node_positions, kv_store, model, corpus, prev_caches
    )
    prev_caches = this_caches
    cache_snapshot = {} of Int32 => Array(MicroGPT::AGPT::LayerKVCache)
    this_caches.each do |node_id, caches|
      cache_snapshot[node_id] = caches.map(&.deep_clone)
    end
    results_by_depth[depth] = {results.first, cache_snapshot}
  end

  results_by_depth
end

private def build_branching_depth_batch(model : MicroGPT::MiniGPT) : {Array(MicroGPT::AGPT::BatchedDepthForward::NodeResult), Hash(Int32, Array(MicroGPT::AGPT::LayerKVCache))}
  token_ids = [0, 1, 2, 7, 0, 1, 3, 7, 0, 4, 2, 7, 0, 4, 3, 7] of Int32
  corpus = MicroGPT::AGPT::TrieCorpus.from_token_ids(token_ids, max_depth: 4, vocab_size: 8)
  kv_store = MicroGPT::AGPT::NodeKVStore.new
  node_ancestor_ids = {corpus.root.id => [] of Int32}
  node_positions = {} of Int32 => Int32
  prev_caches = nil.as(Hash(Int32, Array(MicroGPT::AGPT::LayerKVCache))?)

  corpus.each_depth_level do |depth, nodes|
    next if depth == 0

    eligible = [] of MicroGPT::AGPT::BatchedDepthForward::NodeProxy
    nodes.each do |node|
      parent_id = corpus.parent_id(node.id)
      next unless node_ancestor_ids.has_key?(parent_id)
      next if corpus.depth_of(parent_id) >= model.config.seq_len
      eligible << MicroGPT::AGPT::BatchedDepthForward::NodeProxy.new(node.id, node.token_id.not_nil!, node.depth)
    end
    next if eligible.empty?

    eligible.each do |node|
      node_positions[node.id] = depth - 1
    end

    results, this_caches = MicroGPT::AGPT::BatchedDepthForward.forward_depth(
      eligible, node_ancestor_ids, node_positions, kv_store, model, corpus, prev_caches
    )
    prev_caches = this_caches
    if results.size > 1
      cache_snapshot = {} of Int32 => Array(MicroGPT::AGPT::LayerKVCache)
      this_caches.each do |node_id, caches|
        cache_snapshot[node_id] = caches.map(&.deep_clone)
      end
      return {results, cache_snapshot}
    end
  end

  raise "expected a branching depth batch"
end

private def head_slice(m : MicroGPT::Mat, head_dims : Array(Int32), head : Int32) : Array(Float32)
  offset = 0
  head.times { |i| offset += head_dims[i] }
  Array(Float32).new(head_dims[head]) do |i|
    m[0, offset + i]
  end
end

private def row_from_values(values : Array(Float32)) : MicroGPT::Mat
  m = MicroGPT::Mat.new(1, values.size)
  values.each_with_index { |v, i| m[0, i] = v }
  m
end

private def concat_head_values(parts : Array(Array(Float32))) : MicroGPT::Mat
  total = parts.sum(0) { |part| part.size }
  m = MicroGPT::Mat.new(1, total)
  offset = 0
  parts.each do |part|
    part.each_with_index { |v, i| m[0, offset + i] = v }
    offset += part.size
  end
  m
end

private def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float64
  raise "size mismatch #{a.size} != #{b.size}" unless a.size == b.size
  max = 0.0_f64
  a.size.times do |i|
    diff = (a[i] - b[i]).abs.to_f64
    max = diff if diff > max
  end
  max
end

private def max_abs_diff(a : MicroGPT::Mat, b : MicroGPT::Mat) : Float64
  raise "shape mismatch #{a.rows}x#{a.cols} != #{b.rows}x#{b.cols}" unless a.rows == b.rows && a.cols == b.cols
  max = 0.0_f64
  a.rows.times do |r|
    a.cols.times do |c|
      diff = (a[r, c] - b[r, c]).abs.to_f64
      max = diff if diff > max
    end
  end
  max
end

private def softmax_backward_row_reference(s : MicroGPT::Mat, ds : MicroGPT::Mat) : MicroGPT::Mat
  cols = s.cols
  dot = 0.0_f64
  cols.times { |j| dot += ds[0, j] * s[0, j] }
  result = MicroGPT::Mat.new(1, cols)
  cols.times { |j| result[0, j] = (s[0, j] * (ds[0, j] - dot)).to_f32 }
  result
end

private def reference_attention_output(attn_weights : MicroGPT::Mat, layer_cache : MicroGPT::AGPT::LayerKVCache, head : Int32) : Array(Float32)
  old = attn_weights * layer_cache.v_slice(head)
  old.raw_data.dup
end

private def direct_attention_output(attn_weights : MicroGPT::Mat, layer_cache : MicroGPT::AGPT::LayerKVCache, head : Int32, hd : Int32) : Array(Float32)
  values = Array(Float32).new(hd, 0.0_f32)
  weights = attn_weights.raw_data
  v_data = layer_cache.v_parts[head].raw_data
  layer_cache.len.times do |pos|
    base = pos * hd
    hd.times do |j|
      values[j] += weights[pos] * v_data[base + j]
    end
  end
  values
end

private def reference_attention_backward_head(
  position : Int32,
  ancestor_ids : Array(Int32),
  layer : Int32,
  head : Int32,
  head_dims : Array(Int32),
  n_layers : Int32,
  d_out_h : MicroGPT::Mat,
  attn_weights : MicroGPT::Mat,
  q_part : MicroGPT::Mat,
  layer_cache : MicroGPT::AGPT::LayerKVCache,
  accum : MicroGPT::AGPT::NodeGradAccum,
  grad_accums : Hash(Int32, MicroGPT::AGPT::NodeGradAccum),
  rope : MicroGPT::RoPE
) : {Array(Float32), Array(Float32), Array(Float32)}
  hd = head_dims[head]
  prefix_len = layer_cache.len
  k_h = layer_cache.k_slice(head)
  v_h = layer_cache.v_slice(head)

  dv_full = attn_weights.t * d_out_h
  d_weights = d_out_h * v_h.t
  scale = (1.0 / Math.sqrt(hd.to_f64)).to_f32
  d_scores = softmax_backward_row_reference(attn_weights, d_weights)
  d_scores.scale!(scale)
  dq_h = d_scores * k_h
  dk_full = d_scores.t * q_part

  last = prefix_len - 1
  hd.times do |j|
    dk_full[last, j] += accum.dk[layer][head][0, j]
    dv_full[last, j] += accum.dv[layer][head][0, j]
  end

  dk_current = MicroGPT::Mat.new(1, hd)
  dv_current = MicroGPT::Mat.new(1, hd)
  hd.times do |j|
    dk_current[0, j] = dk_full[last, j]
    dv_current[0, j] = dv_full[last, j]
  end

  if prefix_len > 1
    (prefix_len - 1).times do |pos|
      ancestor_id = ancestor_ids[pos]
      acc = grad_accums[ancestor_id]? || begin
        a = MicroGPT::AGPT::NodeGradAccum.new(n_layers, head_dims)
        grad_accums[ancestor_id] = a
        a
      end
      row_dk = MicroGPT::Mat.new(1, hd)
      row_dv = MicroGPT::Mat.new(1, hd)
      hd.times do |j|
        row_dk[0, j] = dk_full[pos, j]
        row_dv[0, j] = dv_full[pos, j]
      end
      acc.add_dk(layer, head, row_dk)
      acc.add_dv(layer, head, row_dv)
    end
  end

  MicroGPT::AGPT::IncrementalForward.apply_inverse_rope_at!(dq_h, rope, position)
  MicroGPT::AGPT::IncrementalForward.apply_inverse_rope_at!(dk_current, rope, position)

  {dq_h.raw_data.dup, dk_current.raw_data.dup, dv_current.raw_data.dup}
end

private def seed_current_accum(head_dims : Array(Int32), prefix_len : Int32) : MicroGPT::AGPT::NodeGradAccum
  accum = MicroGPT::AGPT::NodeGradAccum.new(1, head_dims)
  head_dims.each_with_index do |hd, hi|
    hd.times do |j|
      accum.dk[0][hi][0, j] = (0.001_f32 * (prefix_len + hi + j + 1)).to_f32
      accum.dv[0][hi][0, j] = (0.002_f32 * (prefix_len + hi + j + 1)).to_f32
    end
  end
  accum
end

private def seed_ancestor_accums(ancestor_ids : Array(Int32), head_dims : Array(Int32)) : Hash(Int32, MicroGPT::AGPT::NodeGradAccum)
  accums = {} of Int32 => MicroGPT::AGPT::NodeGradAccum
  ancestor_ids.each_with_index do |ancestor_id, idx|
    next unless idx.even?
    accums[ancestor_id] = MicroGPT::AGPT::NodeGradAccum.new(1, head_dims)
  end
  accums
end

describe "AGPT backward attention optimization" do
  it "matches the old backward-attention math across prefix lengths" do
    tolerance = 1e-5
    model = make_deterministic_model
    block = model.blocks.first
    head_dims = block.attn.head_dims
    results_by_depth = build_results_by_depth(model)

    [1, 2, 3, 4].each do |prefix_len|
      result, caches = results_by_depth[prefix_len]
      layer_cache = caches[result.node_id][0]
      bs = result.block_states[0]

      d_attn_proj = MicroGPT::Mat.new(1, model.config.d_model)
      model.config.d_model.times do |j|
        d_attn_proj[0, j] = (0.01_f32 * (prefix_len + j + 1)).to_f32
      end
      d_concat = d_attn_proj * block.attn.wo.w.t
      d_head_outs = block.attn.split_cols(d_concat, head_dims)

      current_old = seed_current_accum(head_dims, prefix_len)
      current_new = seed_current_accum(head_dims, prefix_len)
      grads_old = seed_ancestor_accums(result.ancestor_ids, head_dims)
      grads_new = seed_ancestor_accums(result.ancestor_ids, head_dims)

      old_dq_parts = [] of Array(Float32)
      old_dk_parts = [] of Array(Float32)
      old_dv_parts = [] of Array(Float32)
      new_dq_parts = [] of Array(Float32)
      new_dk_parts = [] of Array(Float32)
      new_dv_parts = [] of Array(Float32)

      head_dims.each_with_index do |hd, hi|
        output_old = reference_attention_output(bs.attn_weights[hi], layer_cache, hi)
        output_new = direct_attention_output(bs.attn_weights[hi], layer_cache, hi, hd)
        loss_old = output_old.zip(d_head_outs[hi].raw_data).sum(0.0_f64) { |(o, g)| o * g }
        loss_new = output_new.zip(d_head_outs[hi].raw_data).sum(0.0_f64) { |(o, g)| o * g }
        saved_output = head_slice(bs.wo_input, head_dims, hi)

        max_abs_diff(output_old, output_new).should be <= tolerance
        max_abs_diff(output_old, saved_output).should be <= tolerance
        (loss_old - loss_new).abs.should be <= tolerance

        old_dq, old_dk, old_dv = reference_attention_backward_head(
          position: result.position,
          ancestor_ids: result.ancestor_ids,
          layer: 0,
          head: hi,
          head_dims: head_dims,
          n_layers: 1,
          d_out_h: row_from_values(d_head_outs[hi].raw_data.dup),
          attn_weights: bs.attn_weights[hi],
          q_part: bs.q_parts[hi],
          layer_cache: layer_cache,
          accum: current_old,
          grad_accums: grads_old,
          rope: block.attn.ropes[hi]
        )
        scratch_dw = Array(Float32).new(prefix_len, 0.0_f32)
        scratch_ds = Array(Float32).new(prefix_len, 0.0_f32)
        scratch_rp = Array(Float32).new(hd, 0.0_f32)
        new_dq = Array(Float32).new(hd, 0.0_f32)
        new_dk = Array(Float32).new(hd, 0.0_f32)
        new_dv = Array(Float32).new(hd, 0.0_f32)
        MicroGPT::AGPT::BatchedDepthBackward.optimized_attention_backward_head(
          position: result.position,
          ancestor_ids: result.ancestor_ids,
          layer: 0,
          head: hi,
          head_dims: head_dims,
          n_layers: 1,
          d_out_data: d_head_outs[hi].raw_data,
          d_out_base: 0,
          attn_weights: bs.attn_weights[hi],
          q_part: bs.q_parts[hi],
          layer_cache: layer_cache,
          accum: current_new,
          grad_accums: grads_new,
          rope: block.attn.ropes[hi],
          scratch_d_weights: scratch_dw,
          scratch_d_scores: scratch_ds,
          scratch_rope: scratch_rp,
          dq_all_data: new_dq,
          dk_current_all_data: new_dk,
          dv_current_all_data: new_dv,
          out_offset: 0
        )

        max_abs_diff(old_dq, new_dq).should be <= tolerance
        max_abs_diff(old_dk, new_dk).should be <= tolerance
        max_abs_diff(old_dv, new_dv).should be <= tolerance

        old_dq_parts << old_dq
        old_dk_parts << old_dk
        old_dv_parts << old_dv
        new_dq_parts << new_dq
        new_dk_parts << new_dk
        new_dv_parts << new_dv
      end

      old_dq_all = concat_head_values(old_dq_parts)
      old_dk_all = concat_head_values(old_dk_parts)
      old_dv_all = concat_head_values(old_dv_parts)
      new_dq_all = concat_head_values(new_dq_parts)
      new_dk_all = concat_head_values(new_dk_parts)
      new_dv_all = concat_head_values(new_dv_parts)

      max_abs_diff(old_dq_all, new_dq_all).should be <= tolerance
      max_abs_diff(old_dk_all, new_dk_all).should be <= tolerance
      max_abs_diff(old_dv_all, new_dv_all).should be <= tolerance

      dwq_old = bs.ln1_out.t * old_dq_all
      dwk_old = bs.ln1_out.t * old_dk_all
      dwv_old = bs.ln1_out.t * old_dv_all
      dwq_new = bs.ln1_out.t * new_dq_all
      dwk_new = bs.ln1_out.t * new_dk_all
      dwv_new = bs.ln1_out.t * new_dv_all

      max_abs_diff(dwq_old, dwq_new).should be <= tolerance
      max_abs_diff(dwk_old, dwk_new).should be <= tolerance
      max_abs_diff(dwv_old, dwv_new).should be <= tolerance

      result.ancestor_ids.each do |ancestor_id|
        next unless old_acc = grads_old[ancestor_id]?
        new_acc = grads_new[ancestor_id]?
        new_acc.should_not be_nil
        head_dims.each_with_index do |_hd, hi|
          max_abs_diff(old_acc.dk[0][hi], new_acc.not_nil!.dk[0][hi]).should be <= tolerance
          max_abs_diff(old_acc.dv[0][hi], new_acc.not_nil!.dv[0][hi]).should be <= tolerance
        end
      end
    end
  end

  it "matches the old attention loop across a branching depth batch" do
    tolerance = 1e-5
    model = make_deterministic_model
    block = model.blocks.first
    head_dims = block.attn.head_dims
    results, caches = build_branching_depth_batch(model)

    d_attn_proj = MicroGPT::Mat.new(results.size, model.config.d_model)
    results.size.times do |i|
      model.config.d_model.times do |j|
        d_attn_proj[i, j] = (0.005_f32 * (i * model.config.d_model + j + 1)).to_f32
      end
    end
    d_concat = d_attn_proj * block.attn.wo.w.t
    d_head_outs = block.attn.split_cols(d_concat, head_dims)

    grads_old = {} of Int32 => MicroGPT::AGPT::NodeGradAccum
    grads_new = {} of Int32 => MicroGPT::AGPT::NodeGradAccum
    dq_old = MicroGPT::Mat.new(results.size, model.config.d_model)
    dk_old = MicroGPT::Mat.new(results.size, model.config.d_model)
    dv_old = MicroGPT::Mat.new(results.size, model.config.d_model)
    dq_new = MicroGPT::Mat.new(results.size, model.config.d_model)
    dk_new = MicroGPT::Mat.new(results.size, model.config.d_model)
    dv_new = MicroGPT::Mat.new(results.size, model.config.d_model)

    results.each_with_index do |result, i|
      layer_cache = caches[result.node_id][0]
      bs = result.block_states[0]
      current_old = seed_current_accum(head_dims, layer_cache.len)
      current_new = seed_current_accum(head_dims, layer_cache.len)
      col_offset = 0

      head_dims.each_with_index do |hd, hi|
        old_dq, old_dk, old_dv = reference_attention_backward_head(
          position: result.position,
          ancestor_ids: result.ancestor_ids,
          layer: 0,
          head: hi,
          head_dims: head_dims,
          n_layers: 1,
          d_out_h: row_from_values(d_head_outs[hi].raw_data[i * hd, hd]),
          attn_weights: bs.attn_weights[hi],
          q_part: bs.q_parts[hi],
          layer_cache: layer_cache,
          accum: current_old,
          grad_accums: grads_old,
          rope: block.attn.ropes[hi]
        )
        scratch_dw2 = Array(Float32).new(layer_cache.len, 0.0_f32)
        scratch_ds2 = Array(Float32).new(layer_cache.len, 0.0_f32)
        scratch_rp2 = Array(Float32).new(hd, 0.0_f32)
        new_dq = Array(Float32).new(hd, 0.0_f32)
        new_dk = Array(Float32).new(hd, 0.0_f32)
        new_dv = Array(Float32).new(hd, 0.0_f32)
        MicroGPT::AGPT::BatchedDepthBackward.optimized_attention_backward_head(
          position: result.position,
          ancestor_ids: result.ancestor_ids,
          layer: 0,
          head: hi,
          head_dims: head_dims,
          n_layers: 1,
          d_out_data: d_head_outs[hi].raw_data,
          d_out_base: i * hd,
          attn_weights: bs.attn_weights[hi],
          q_part: bs.q_parts[hi],
          layer_cache: layer_cache,
          accum: current_new,
          grad_accums: grads_new,
          rope: block.attn.ropes[hi],
          scratch_d_weights: scratch_dw2,
          scratch_d_scores: scratch_ds2,
          scratch_rope: scratch_rp2,
          dq_all_data: new_dq,
          dk_current_all_data: new_dk,
          dv_current_all_data: new_dv,
          out_offset: 0
        )

        max_abs_diff(old_dq, new_dq).should be <= tolerance
        max_abs_diff(old_dk, new_dk).should be <= tolerance
        max_abs_diff(old_dv, new_dv).should be <= tolerance

        hd.times do |j|
          dq_old[i, col_offset + j] = old_dq[j]
          dk_old[i, col_offset + j] = old_dk[j]
          dv_old[i, col_offset + j] = old_dv[j]
          dq_new[i, col_offset + j] = new_dq[j]
          dk_new[i, col_offset + j] = new_dk[j]
          dv_new[i, col_offset + j] = new_dv[j]
        end
        col_offset += hd
      end
    end

    max_abs_diff(dq_old, dq_new).should be <= tolerance
    max_abs_diff(dk_old, dk_new).should be <= tolerance
    max_abs_diff(dv_old, dv_new).should be <= tolerance

    grads_old.keys.sort.each do |node_id|
      new_acc = grads_new[node_id]?
      new_acc.should_not be_nil
      head_dims.each_with_index do |_hd, hi|
        max_abs_diff(grads_old[node_id].dk[0][hi], new_acc.not_nil!.dk[0][hi]).should be <= tolerance
        max_abs_diff(grads_old[node_id].dv[0][hi], new_acc.not_nil!.dv[0][hi]).should be <= tolerance
      end
    end
  end
end
