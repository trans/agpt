# Pluggable routers for cooperative expert ensembles
#
# All routers: input_ids + stream → per-position expert weights [seq_len, n_routed]
# Backward: d_router_probs → updates internal params, returns d_stream contribution
#
# Available routers:
#   GlobalRouter  — mean(stream) → W · softmax (original, single weight for whole sequence)
#   ContextRouter — "follow the vector": ctx[t] = ctx[t-1] + emb[t], per-position routing
#   GatedRouter   — gated context: ctx[t] = gate(t)*ctx[t-1] + emb[t], learned reset

module MicroGPT

abstract class Router
  # Number of routed experts
  getter n_routed : Int32

  # Exploration rate (ε-greedy)
  property epsilon : Float64 = 0.2

  def initialize(@n_routed : Int32)
  end

  # Forward: compute per-position blended router probs [seq_len, n_routed]
  # Also caches state needed for backward.
  abstract def forward(input_ids : Array(Int32), stream : Mat) : Mat

  # Backward: given d_router_probs [seq_len, n_routed], compute gradients
  # and return d_stream contribution from router. Updates own params.
  abstract def backward(d_router_probs : Mat, lr : Float64) : Mat

  # Weight matrices for serialization (WeightStore)
  abstract def weight_mats : Array(Mat)

  # Adam state matrices (2 per weight mat: m, v)
  abstract def adam_mats : Array(Mat)

  # Gradient matrices (same order as weight_mats)
  abstract def grad_mats : Array(Mat)

  # Parameter count
  abstract def param_count : Int64

  # Human-readable description
  abstract def describe : String

  # Apply ε-greedy blending to raw softmax probs
  protected def apply_epsilon(raw_probs : Mat) : Mat
    eps = @epsilon.to_f32
    uniform = 1.0_f32 / @n_routed
    blended = Mat.new(raw_probs.rows, raw_probs.cols)
    (raw_probs.rows * raw_probs.cols).times do |idx|
      blended.raw_data[idx] = (1.0_f32 - eps) * raw_probs.data[idx] + eps * uniform
    end
    blended
  end
end

# =============================================================================
# GlobalRouter — original router: mean(stream) → linear → softmax
# Single routing weight for the whole sequence.
# =============================================================================

class GlobalRouter < Router
  getter w_router : Linear

  @s_mean : Mat?
  @router_probs_raw : Mat?
  @stream_rows : Int32 = 0
  @stream_cols : Int32 = 0

  def initialize(@n_routed : Int32, stream_dim : Int32)
    super(@n_routed)
    @w_router = Linear.new(stream_dim, @n_routed)
    @w_router.w.zero!
    @w_router.b.zero!
  end

  def forward(input_ids : Array(Int32), stream : Mat) : Mat
    seq_len = stream.rows
    stream_dim = stream.cols
    @stream_rows = seq_len
    @stream_cols = stream_dim

    # Mean pool stream over sequence
    s_mean = Mat.new(1, stream_dim)
    stream_dim.times do |j|
      sum = 0.0_f32
      seq_len.times { |r| sum += stream[r, j] }
      s_mean[0, j] = sum / seq_len
    end
    @s_mean = s_mean

    router_out = @w_router.forward(s_mean)  # [1, nr]
    router_probs_raw = MicroGPT.backend.softmax_rows(router_out)
    @router_probs_raw = router_probs_raw

    blended = apply_epsilon(router_probs_raw)

    # Broadcast [1, nr] → [seq_len, nr] for uniform interface
    result = Mat.new(seq_len, @n_routed)
    seq_len.times do |r|
      @n_routed.times { |k| result[r, k] = blended[0, k] }
    end
    result
  end

  def backward(d_router_probs : Mat, lr : Float64) : Mat
    seq_len = d_router_probs.rows
    router_probs_raw = @router_probs_raw.not_nil!

    # Sum d_router_probs across positions (since forward broadcast 1→seq_len)
    d_blended = Mat.new(1, @n_routed)
    seq_len.times do |r|
      @n_routed.times { |k| d_blended[0, k] += d_router_probs[r, k] }
    end

    # ε-greedy backward: d_raw = (1-ε) * d_blended
    eps = @epsilon.to_f32
    d_raw = Mat.new(1, @n_routed)
    @n_routed.times { |k| d_raw[0, k] = (1.0_f32 - eps) * d_blended[0, k] }

    d_router_out = MicroGPT.backend.softmax_backward(router_probs_raw, d_raw)
    d_s_mean = @w_router.backward(d_router_out)

    # Broadcast d_s_mean → d_stream (mean backward: divide by seq_len)
    stream_dim = @stream_cols
    d_stream = Mat.new(seq_len, stream_dim)
    inv_seq = 1.0_f32 / seq_len
    d_s_mean_d = d_s_mean.data
    d_stream_d = d_stream.raw_data
    seq_len.times do |r|
      stream_dim.times do |c|
        d_stream_d[r * stream_dim + c] = d_s_mean_d[c] * inv_seq
      end
    end

    @w_router.update(lr)
    d_stream
  end

  def weight_mats : Array(Mat)
    [@w_router.w, @w_router.b]
  end

  def adam_mats : Array(Mat)
    [@w_router.adam_w.m, @w_router.adam_w.v,
     @w_router.adam_b.m, @w_router.adam_b.v]
  end

  def grad_mats : Array(Mat)
    [@w_router.dw, @w_router.db]
  end

  def param_count : Int64
    (@w_router.w.data.size + @w_router.b.data.size).to_i64
  end

  def describe : String
    "global"
  end
end

# =============================================================================
# ContextRouter — "follow the vector" per-position routing
# ctx[t] = ctx[t-1] + router_emb[token[t]]
# Router uses ctx[t] per position: W_router · ctx[t] → softmax
# =============================================================================

class ContextRouter < Router
  getter router_emb : Mat         # [vocab_size, stream_dim]
  getter d_router_emb : Mat
  getter adam_emb : AdamParam
  getter w_router : Linear

  @ctx : Mat?                     # [seq_len, stream_dim] — cached context vectors
  @router_probs_raw : Mat?
  @last_ids : Array(Int32)?
  @stream_dim : Int32

  def initialize(@n_routed : Int32, @stream_dim : Int32, vocab_size : Int32)
    super(@n_routed)
    scale = Math.sqrt(1.0 / @stream_dim)
    @router_emb = Mat.randn(vocab_size, @stream_dim, scale)
    @d_router_emb = Mat.new(vocab_size, @stream_dim)
    @adam_emb = AdamParam.new(vocab_size, @stream_dim)
    @w_router = Linear.new(@stream_dim, @n_routed)
    @w_router.w.zero!
    @w_router.b.zero!
  end

  def forward(input_ids : Array(Int32), stream : Mat) : Mat
    seq_len = input_ids.size
    @last_ids = input_ids

    # Build context vectors: ctx[t] = ctx[t-1] + router_emb[token[t]]
    ctx = Mat.new(seq_len, @stream_dim)
    seq_len.times do |t|
      tok = input_ids[t]
      @stream_dim.times do |j|
        prev = t > 0 ? ctx[t - 1, j] : 0.0_f32
        ctx[t, j] = prev + @router_emb[tok, j]
      end
    end
    @ctx = ctx

    # Per-position routing: W_router · ctx[t] → softmax
    router_out = @w_router.forward(ctx)  # [seq_len, nr]
    router_probs_raw = MicroGPT.backend.softmax_rows(router_out)
    @router_probs_raw = router_probs_raw

    apply_epsilon(router_probs_raw)
  end

  def backward(d_router_probs : Mat, lr : Float64) : Mat
    seq_len = d_router_probs.rows
    router_probs_raw = @router_probs_raw.not_nil!
    ctx = @ctx.not_nil!
    ids = @last_ids.not_nil!

    # ε-greedy backward
    eps = @epsilon.to_f32
    d_raw = Mat.new(seq_len, @n_routed)
    (seq_len * @n_routed).times do |idx|
      d_raw.raw_data[idx] = (1.0_f32 - eps) * d_router_probs.data[idx]
    end

    d_router_out = MicroGPT.backend.softmax_backward(router_probs_raw, d_raw)
    d_ctx = @w_router.backward(d_router_out)  # [seq_len, stream_dim]

    # Backward through cumulative sum: d_ctx[t] flows to all t' <= t
    # Efficient: accumulate from end
    # d_router_emb[tok[t]] += d_ctx_accum[t]
    # d_ctx_accum[t] = d_ctx[t] + d_ctx_accum[t+1]
    @d_router_emb = Mat.new(@router_emb.rows, @stream_dim)
    accum = Array(Float32).new(@stream_dim, 0.0_f32)
    (seq_len - 1).downto(0) do |t|
      @stream_dim.times { |j| accum[j] += d_ctx[t, j] }
      tok = ids[t]
      @stream_dim.times { |j| @d_router_emb[tok, j] += accum[j] }
    end

    @adam_emb.step(@router_emb, @d_router_emb, lr)
    @w_router.update(lr)

    # Context router doesn't contribute to stream gradient
    Mat.new(seq_len, @stream_dim)
  end

  def weight_mats : Array(Mat)
    [@router_emb, @w_router.w, @w_router.b]
  end

  def adam_mats : Array(Mat)
    [@adam_emb.m, @adam_emb.v,
     @w_router.adam_w.m, @w_router.adam_w.v,
     @w_router.adam_b.m, @w_router.adam_b.v]
  end

  def grad_mats : Array(Mat)
    [@d_router_emb, @w_router.dw, @w_router.db]
  end

  def param_count : Int64
    (@router_emb.data.size + @w_router.w.data.size + @w_router.b.data.size).to_i64
  end

  def describe : String
    "context"
  end
end

# =============================================================================
# GatedRouter — gated context accumulation with learned reset
# gate[t] = sigmoid(W_gate · emb[t])
# ctx[t] = gate[t] * ctx[t-1] + emb[t]
# Router: W_router · ctx[t] → softmax per position
# =============================================================================

class GatedRouter < Router
  getter router_emb : Mat         # [vocab_size, stream_dim]
  getter d_router_emb : Mat
  getter adam_emb : AdamParam
  getter w_router : Linear
  getter w_gate : Linear          # stream_dim → 1 (scalar gate per position)

  @ctx : Mat?
  @gates : Mat?                   # [seq_len, 1] — sigmoid outputs
  @gate_pre : Mat?                # [seq_len, 1] — pre-sigmoid
  @router_probs_raw : Mat?
  @last_ids : Array(Int32)?
  @emb_rows : Mat?                # [seq_len, stream_dim] — looked-up embeddings
  @stream_dim : Int32

  def initialize(@n_routed : Int32, @stream_dim : Int32, vocab_size : Int32)
    super(@n_routed)
    scale = Math.sqrt(1.0 / @stream_dim)
    @router_emb = Mat.randn(vocab_size, @stream_dim, scale)
    @d_router_emb = Mat.new(vocab_size, @stream_dim)
    @adam_emb = AdamParam.new(vocab_size, @stream_dim)
    @w_router = Linear.new(@stream_dim, @n_routed)
    @w_router.w.zero!
    @w_router.b.zero!
    # Gate: bias init to +2 so sigmoid ≈ 0.88 → mostly remembers by default
    @w_gate = Linear.new(@stream_dim, 1)
    @w_gate.b.raw_data[0] = 2.0_f32
  end

  def forward(input_ids : Array(Int32), stream : Mat) : Mat
    seq_len = input_ids.size
    @last_ids = input_ids

    # Gather embeddings
    emb = Mat.new(seq_len, @stream_dim)
    seq_len.times do |t|
      tok = input_ids[t]
      @stream_dim.times { |j| emb[t, j] = @router_emb[tok, j] }
    end
    @emb_rows = emb

    # Compute gates: sigmoid(W_gate · emb[t])
    gate_pre = @w_gate.forward(emb)  # [seq_len, 1]
    @gate_pre = gate_pre
    gates = Mat.new(seq_len, 1)
    seq_len.times do |t|
      gates[t, 0] = (1.0_f32 / (1.0_f32 + Math.exp(-gate_pre[t, 0]))).to_f32
    end
    @gates = gates

    # Build gated context: ctx[t] = gate[t] * ctx[t-1] + emb[t]
    ctx = Mat.new(seq_len, @stream_dim)
    seq_len.times do |t|
      g = gates[t, 0]
      @stream_dim.times do |j|
        prev = t > 0 ? ctx[t - 1, j] : 0.0_f32
        ctx[t, j] = g * prev + emb[t, j]
      end
    end
    @ctx = ctx

    # Per-position routing
    router_out = @w_router.forward(ctx)  # [seq_len, nr]
    router_probs_raw = MicroGPT.backend.softmax_rows(router_out)
    @router_probs_raw = router_probs_raw

    apply_epsilon(router_probs_raw)
  end

  def backward(d_router_probs : Mat, lr : Float64) : Mat
    seq_len = d_router_probs.rows
    router_probs_raw = @router_probs_raw.not_nil!
    ctx = @ctx.not_nil!
    gates = @gates.not_nil!
    emb = @emb_rows.not_nil!
    ids = @last_ids.not_nil!

    # ε-greedy backward
    eps = @epsilon.to_f32
    d_raw = Mat.new(seq_len, @n_routed)
    (seq_len * @n_routed).times do |idx|
      d_raw.raw_data[idx] = (1.0_f32 - eps) * d_router_probs.data[idx]
    end

    d_router_out = MicroGPT.backend.softmax_backward(router_probs_raw, d_raw)
    d_ctx = @w_router.backward(d_router_out)  # [seq_len, stream_dim]

    # Backward through gated recurrence:
    # ctx[t] = gate[t] * ctx[t-1] + emb[t]
    # d_gate[t] = d_ctx[t] · ctx[t-1]  (dot product → scalar)
    # d_ctx[t-1] += gate[t] * d_ctx[t]  (accumulate from future)
    # d_emb[t] = d_ctx[t]
    @d_router_emb = Mat.new(@router_emb.rows, @stream_dim)
    d_gate_pre = Mat.new(seq_len, 1)

    # Process reverse
    (seq_len - 1).downto(0) do |t|
      g = gates[t, 0]

      # d_gate[t] = sum_j(d_ctx[t,j] * ctx[t-1,j]) * sigmoid'(gate_pre[t])
      if t > 0
        dot = 0.0_f32
        @stream_dim.times { |j| dot += d_ctx[t, j] * ctx[t - 1, j] }
        d_gate_pre[t, 0] = dot * g * (1.0_f32 - g)  # sigmoid derivative
      end

      # d_emb[t] = d_ctx[t]
      tok = ids[t]
      @stream_dim.times { |j| @d_router_emb[tok, j] += d_ctx[t, j] }

      # d_ctx[t-1] += gate[t] * d_ctx[t]
      if t > 0
        @stream_dim.times { |j| d_ctx[t - 1, j] += g * d_ctx[t, j] }
      end
    end

    # Backward through W_gate
    @w_gate.backward(d_gate_pre)
    @w_gate.update(lr)

    @adam_emb.step(@router_emb, @d_router_emb, lr)
    @w_router.update(lr)

    # Gated router doesn't contribute to stream gradient
    Mat.new(seq_len, @stream_dim)
  end

  def weight_mats : Array(Mat)
    [@router_emb, @w_router.w, @w_router.b, @w_gate.w, @w_gate.b]
  end

  def adam_mats : Array(Mat)
    [@adam_emb.m, @adam_emb.v,
     @w_router.adam_w.m, @w_router.adam_w.v,
     @w_router.adam_b.m, @w_router.adam_b.v,
     @w_gate.adam_w.m, @w_gate.adam_w.v,
     @w_gate.adam_b.m, @w_gate.adam_b.v]
  end

  def grad_mats : Array(Mat)
    [@d_router_emb, @w_router.dw, @w_router.db, @w_gate.dw, @w_gate.db]
  end

  def param_count : Int64
    (@router_emb.data.size + @w_router.w.data.size + @w_router.b.data.size +
     @w_gate.w.data.size + @w_gate.b.data.size).to_i64
  end

  def describe : String
    "gated"
  end
end

end
