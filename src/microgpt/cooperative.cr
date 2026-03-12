# Cooperative μGPT Ensemble — Multiple small models communicating through a shared stream
#
# Architecture:
#   - N independent MiniGPT experts run sequentially
#   - Expert 0 can be an algorithmic expert (bigram/calculator) — stream-only, no logits
#   - Each expert reads from shared stream via W_read (stream_dim → d_model)
#   - Each expert writes to shared stream via W_write (d_model → stream_dim)
#   - Stream-gated router: softmax(W_r · mean(stream)) produces per-expert weights
#   - Router only covers transformer experts (algorithmic E0 excluded from logit mix)
#   - Final output: weighted sum of transformer expert logits
#   - Position: RoPE in each expert's attention (no pos_emb, no counter)
#
# Option C interface: x_in = embedding(tokens) + W_read · stream
#
# Anti-collapse measures:
#   - ε-greedy routing: w = (1-ε)*softmax(z) + ε/n (floor prevents silencing)
#   - Equal gradient: all experts get (1/n_routed)*d_agg (prevents learning rate disparity)
#   - Zero-init router: starts with uniform weights

module MicroGPT

class CooperativeModel
  include MathUtils

  getter experts : Array(MiniGPT)
  getter stream_dim : Int32
  getter n_experts : Int32
  getter has_counter : Bool

  # Per-expert stream interface
  getter w_reads : Array(Linear)
  getter w_writes : Array(Linear)

  # Router: stream_dim → n_routed (transformer experts only)
  getter w_router : Linear

  # True counter: learned positional signal injected directly into stream
  # Only seq_len × stream_dim params — no transformer, no logits
  getter counter_pos : Mat?

  # N-gram / algorithmic expert: fixed lookup table + learned projection
  getter bigram_table : (BigramTable | TrigramTable | CalculatorExpert)?
  getter w_bigram : Linear?          # vocab_size → stream_dim (learned)
  @bigram_features : Mat?            # cached for backward

  # Cached forward state for backward
  @routed_logits : Array(Mat)?       # logits from transformer experts only
  @expert_pre_norm : Array(Mat)?
  @router_probs : Mat?       # blended ε-greedy probs (n_routed)
  @router_probs_raw : Mat?   # pure softmax probs (for backward)
  @final_stream : Mat?

  # Stream contribution tracking per expert
  @stream_norms : Array(Float64) = [] of Float64      # ||delta_i|| (L2 norm)
  @stream_cosines : Array(Float64) = [] of Float64    # cosine change or cosine vs final
  @_e0_delta : Mat? = nil                              # saved for E0 vs final cosine

  # Router exploration rate (ε-greedy)
  property router_epsilon : Float64 = 0.2

  # Stream bandwidth masking: only first active_stream_dims carry signal
  # Allows testing narrower streams without resizing matrices
  property active_stream_dims : Int32 = 0  # 0 = use full stream_dim

  def effective_stream_width : Int32
    @active_stream_dims > 0 ? @active_stream_dims : @stream_dim
  end

  # Is expert i stream-only (no logits)?
  private def stream_only?(i : Int32) : Bool
    i == 0 && (!@bigram_table.nil? || @has_counter)
  end

  # Number of experts that participate in router/logit aggregation
  private def n_routed : Int32
    stream_only?(0) ? @n_experts - 1 : @n_experts
  end

  def initialize(expert_configs : Array(Config), @stream_dim : Int32, @has_counter : Bool = true)
    @n_experts = expert_configs.size
    @experts = expert_configs.map { |cfg| MiniGPT.new(cfg) }

    @w_reads = expert_configs.map { |cfg| Linear.new(@stream_dim, cfg.d_model) }
    @w_writes = expert_configs.map { |cfg| Linear.new(cfg.d_model, @stream_dim) }

    nr = @has_counter ? @n_experts - 1 : @n_experts
    @w_router = Linear.new(@stream_dim, nr)

    # True counter: learned position signal → stream (small random init)
    if @has_counter
      max_seq = expert_configs[0].seq_len
      @counter_pos = Mat.randn(max_seq, @stream_dim, 0.02)
    end

    # Zero-init router → uniform initial routing
    @w_router.w.zero!
    @w_router.b.zero!
  end

  # Attach a bigram table as algorithmic expert 0
  # Call after initialize — replaces counter if present
  def attach_bigram(table : BigramTable | TrigramTable | CalculatorExpert)
    @bigram_table = table
    @w_bigram = Linear.new(table.vocab_size, @stream_dim)
    @has_counter = false  # bigram replaces counter
    # Rebuild router for transformer-only experts
    @w_router = Linear.new(@stream_dim, @n_experts - 1)
    @w_router.w.zero!
    @w_router.b.zero!
  end

  def detach_bigram
    @bigram_table = nil
    @w_bigram = nil
    @bigram_features = nil
    # Rebuild router for all experts (E0 becomes transformer)
    @w_router = Linear.new(@stream_dim, @n_experts)
    @w_router.w.zero!
    @w_router.b.zero!
  end

  def forward(input_ids : Array(Int32)) : Mat
    seq_len = input_ids.size
    stream = Mat.zeros(seq_len, @stream_dim)

    routed_logits = Array(Mat).new(n_routed)
    expert_pre_norm = Array(Mat).new(@n_experts)
    stream_norms = Array(Float64).new(@n_experts, 0.0)
    stream_cosines = Array(Float64).new(@n_experts, 0.0)

    @n_experts.times do |i|
      expert = @experts[i]

      if (bt = @bigram_table) && (wb = @w_bigram) && i == 0
        # Algorithmic expert: lookup features, project to stream
        bg_feats = bt.lookup(input_ids)  # [seq_len, vocab_size]
        @bigram_features = bg_feats
        delta = wb.forward(bg_feats)     # [seq_len, stream_dim] (learned projection)
        stream_norms[i] = Math.sqrt(delta.data.sum { |v| (v * v).to_f64 })
        @_e0_delta = delta  # save for cosine vs final stream
        stream = stream + delta
        expert_pre_norm << Mat.new(seq_len, expert.config.d_model)
        # No logits — stream-only expert
      elsif @has_counter && i == 0
        # True counter: just add learned positional signal to stream
        cp = @counter_pos.not_nil!
        norm_sq = 0.0
        seq_len.times do |r|
          @stream_dim.times do |c|
            norm_sq += (cp[r, c] * cp[r, c]).to_f64
          end
        end
        stream_norms[i] = Math.sqrt(norm_sq)
        # Can't compute cosine for counter (no Mat delta), use 0
        stream_cosines[i] = 0.0
        seq_len.times do |r|
          @stream_dim.times { |c| stream[r, c] += cp[r, c] }
        end
        expert_pre_norm << Mat.new(seq_len, expert.config.d_model)
        # No logits — stream-only expert
      else
        # Read stream → project to expert's d_model
        s_proj = @w_reads[i].forward(stream)

        # Expert embedding
        x = expert.embedding.forward(input_ids)

        # Additive stream injection (Option C)
        x.add!(s_proj)

        # Run through transformer blocks
        expert.blocks.each { |b| x = b.forward(x) }
        expert_pre_norm << x

        # Final norm + output projection
        h = expert.final_norm.forward(x)
        logits_i = expert.output.forward(h)
        routed_logits << logits_i

        # Write to stream: delta = W_write(h)
        delta = @w_writes[i].forward(h)
        stream_norms[i] = Math.sqrt(delta.data.sum { |v| (v * v).to_f64 })
        stream_cosines[i] = stream_cos_change(stream, delta)
        stream = stream + delta
      end

      # Mask: zero out dims beyond active width
      aw = effective_stream_width
      if aw < @stream_dim
        seq_len.times do |r|
          (aw...@stream_dim).each { |c| stream[r, c] = 0.0_f32 }
        end
      end
    end

    # Compute E0's cosine similarity against final stream (how much signal survived)
    if (e0d = @_e0_delta) && stream_only?(0)
      stream_cosines[0] = cosine_similarity(e0d, stream)
    end

    @routed_logits = routed_logits
    @expert_pre_norm = expert_pre_norm
    @final_stream = stream
    @stream_norms = stream_norms
    @stream_cosines = stream_cosines

    # Router: mean pool stream over sequence, then linear + softmax
    s_mean = Mat.new(1, @stream_dim)
    @stream_dim.times do |j|
      sum = 0.0_f32
      seq_len.times { |r| sum += stream[r, j] }
      s_mean[0, j] = sum / seq_len
    end

    nr = n_routed
    router_out = @w_router.forward(s_mean)  # [1, nr]
    router_probs_raw = MicroGPT.backend.softmax_rows(router_out)
    @router_probs_raw = router_probs_raw

    # ε-greedy blending: w = (1-ε)*softmax + ε/n
    eps = @router_epsilon.to_f32
    uniform = 1.0_f32 / nr
    router_probs = Mat.new(1, nr)
    nr.times do |k|
      router_probs[0, k] = (1.0_f32 - eps) * router_probs_raw[0, k] + eps * uniform
    end
    @router_probs = router_probs

    # Weighted aggregation of routed expert logits
    rows = routed_logits[0].rows
    cols = routed_logits[0].cols
    agg = Mat.new(rows, cols)
    nr.times do |k|
      w = router_probs[0, k]
      li = routed_logits[k]
      agg_d = agg.raw_data
      li_d = li.data
      (rows * cols).times do |idx|
        agg_d[idx] += w * li_d[idx]
      end
    end

    agg
  end

  def train_step(input_ids : Array(Int32), target_ids : Array(Int32)) : Float64
    seq_len = input_ids.size
    agg_logits = forward(input_ids)

    # Compute CE loss on aggregated logits
    vocab_size = agg_logits.cols
    b = MicroGPT.backend
    if b.is_a?(CuBLASBackend)
      loss, d_agg = b.fused_softmax_ce_grad(agg_logits, target_ids)
    else
      probs = b.softmax_rows(agg_logits)

      loss = 0.0
      target_ids.each_with_index { |t, i| loss -= Math.log(probs[i, t] + 1e-10) }
      loss /= seq_len

      # d_agg = (probs - one_hot) / seq_len
      d_agg = Mat.new(seq_len, vocab_size)
      seq_len.times do |i|
        vocab_size.times { |j| d_agg[i, j] = probs[i, j] }
        d_agg[i, target_ids[i]] -= 1.0
      end
      d_agg.scale!(1.0 / seq_len)
    end

    # Retrieve cached state
    nr = n_routed
    router_probs = @router_probs.not_nil!
    router_probs_raw = @router_probs_raw.not_nil!
    routed_logits = @routed_logits.not_nil!

    # --- Router gradient ---
    # d_w_k = Σ_{r,c} d_agg[r,c] * logits_k[r,c]
    d_router_probs = Mat.new(1, nr)
    nr.times do |k|
      dot = 0.0_f32
      li = routed_logits[k]
      d_d = d_agg.data
      li_d = li.data
      (seq_len * vocab_size).times do |idx|
        dot += d_d[idx] * li_d[idx]
      end
      d_router_probs[0, k] = dot
    end

    # ε-greedy backward: d_raw = (1-ε) * d_blended
    eps = @router_epsilon.to_f32
    d_router_probs_raw = Mat.new(1, nr)
    nr.times do |k|
      d_router_probs_raw[0, k] = (1.0_f32 - eps) * d_router_probs[0, k]
    end

    d_router_out = MicroGPT.backend.softmax_backward(router_probs_raw, d_router_probs_raw)
    d_s_mean = @w_router.backward(d_router_out)

    # Broadcast d_s_mean → d_stream (mean backward: divide by seq_len)
    d_stream = Mat.new(seq_len, @stream_dim)
    inv_seq = 1.0_f32 / seq_len
    d_s_mean_d = d_s_mean.data
    d_stream_d = d_stream.raw_data
    seq_len.times do |r|
      @stream_dim.times do |c|
        d_stream_d[r * @stream_dim + c] = d_s_mean_d[c] * inv_seq
      end
    end

    # Mask d_stream to match forward: zero gradient for inactive dims
    aw = effective_stream_width
    if aw < @stream_dim
      seq_len.times do |r|
        (aw...@stream_dim).each { |c| d_stream[r, c] = 0.0_f32 }
      end
    end

    # --- Backward through experts (reverse for sequential stream) ---
    lr = @experts[0].config.learning_rate

    # Equal gradient: each transformer expert gets (1/n_routed) * d_agg
    equal_scale = 1.0_f32 / nr

    (@n_experts - 1).downto(0) do |i|
      expert = @experts[i]

      if (bt = @bigram_table) && (wb = @w_bigram) && i == 0
        wb.backward(d_stream)
        wb.update(lr)
      elsif @has_counter && i == 0
        # True counter backward: stream += counter_pos, so d_counter_pos = d_stream
        cp = @counter_pos.not_nil!
        seq_len.times do |r|
          @stream_dim.times do |c|
            cp[r, c] -= lr * d_stream[r, c]
          end
        end
      else
        # Equal gradient distribution (straight-through)
        d_logits_i = d_agg * equal_scale

        # Backprop through output projection (Linear backward)
        d_h = expert.output.proj.backward(d_logits_i)

        # Backprop through W_write: delta = W_write(h), stream += delta
        d_h_write = @w_writes[i].backward(d_stream)
        d_h.add!(d_h_write)

        # Backprop through final_norm
        d_x = expert.final_norm.backward(d_h)

        # Backprop through transformer blocks
        expert.blocks.reverse_each { |b| d_x = b.backward(d_x) }

        # W_read backward: gradient flows to stream
        d_stream_read = @w_reads[i].backward(d_x)
        d_stream.add!(d_stream_read)

        # Embedding backward
        expert.embedding.backward(d_x)
        expert.embedding.update(lr)

        # Update expert parameters
        expert.blocks.each &.update(lr)
        expert.final_norm.update(lr)
        expert.output.update(lr)

        # Update stream interface
        @w_reads[i].update(lr)
        @w_writes[i].update(lr)
      end
    end

    # Update router
    @w_router.update(lr)

    loss
  end

  def generate(start_ids : Array(Int32), max_tokens : Int32, temperature : Float64 = 1.0) : Array(Int32)
    seq_len = @experts[0].config.seq_len
    ids = start_ids.dup
    max_tokens.times do
      context = ids.size > seq_len ? ids[-seq_len..] : ids
      logits = forward(context)

      last_row = logits.rows - 1
      vocab_size = logits.cols

      if temperature <= 0.01
        best_id = 0
        best_val = logits[last_row, 0]
        vocab_size.times do |j|
          if logits[last_row, j] > best_val
            best_val = logits[last_row, j]
            best_id = j
          end
        end
        ids << best_id
      else
        scaled = Mat.new(1, vocab_size)
        vocab_size.times { |j| scaled[0, j] = logits[last_row, j] / temperature }
        probs_row = MicroGPT.backend.softmax_rows(scaled)

        r = rand
        cumulative = 0.0
        chosen = vocab_size - 1
        vocab_size.times do |j|
          cumulative += probs_row[0, j]
          if r <= cumulative
            chosen = j
            break
          end
        end
        ids << chosen
      end
    end
    ids
  end

  def param_count : Int64
    count = 0_i64
    has_bigram = !@bigram_table.nil?
    @experts.each_with_index do |e, i|
      if has_bigram && i == 0
        # Bigram expert: only W_bigram params (no transformer)
        if wb = @w_bigram
          count += wb.w.data.size + wb.b.data.size
        end
      elsif @has_counter && i == 0
        # True counter: only counter_pos params (seq_len × stream_dim)
        count += @counter_pos.not_nil!.data.size
      else
        count += e.param_count
      end
    end
    @w_reads.each_with_index do |l, i|
      next if (has_bigram || @has_counter) && i == 0
      count += l.w.data.size + l.b.data.size
    end
    @w_writes.each_with_index do |l, i|
      next if (has_bigram || @has_counter) && i == 0
      count += l.w.data.size + l.b.data.size
    end
    count += @w_router.w.data.size + @w_router.b.data.size
    count
  end

  # Collect ALL weight matrices in deterministic order (for WeightStore)
  def all_weight_mats : Array(Mat)
    mats = [] of Mat
    has_bigram = !@bigram_table.nil?

    @experts.each_with_index do |e, i|
      if has_bigram && i == 0
        if wb = @w_bigram
          mats << wb.w << wb.b
        end
      elsif @has_counter && i == 0
        mats << @counter_pos.not_nil!
      else
        # Expert weights (same order as MiniGPT#weight_mats)
        mats << e.embedding.token_emb
        e.blocks.each do |b|
          mats << b.attn.wq.w << b.attn.wq.b
          mats << b.attn.wk.w << b.attn.wk.b
          mats << b.attn.wv.w << b.attn.wv.b
          mats << b.attn.wo.w << b.attn.wo.b
          mats << b.ln1.gamma << b.ln1.beta
          mats << b.ff.l1.w << b.ff.l1.b
          mats << b.ff.l2.w << b.ff.l2.b
          mats << b.ln2.gamma << b.ln2.beta
        end
        mats << e.final_norm.gamma << e.final_norm.beta
        mats << e.output.proj.w << e.output.proj.b
      end
    end

    # Stream interface
    @w_reads.each_with_index do |l, i|
      next if (has_bigram || @has_counter) && i == 0
      mats << l.w << l.b
    end
    @w_writes.each_with_index do |l, i|
      next if (has_bigram || @has_counter) && i == 0
      mats << l.w << l.b
    end

    # Router
    mats << @w_router.w << @w_router.b
    mats
  end

  # Collect Adam m/v Mats in same order as all_weight_mats
  # Returns [w0_m, w0_v, w1_m, w1_v, ...] — 2 Mats per weight Mat
  def all_adam_mats : Array(Mat)
    mats = [] of Mat
    has_bigram = !@bigram_table.nil?

    @experts.each_with_index do |e, i|
      if has_bigram && i == 0
        if wb = @w_bigram
          add_linear_adam(mats, wb)
        end
      elsif @has_counter && i == 0
        # counter_pos uses raw SGD, no Adam — add dummy zeros
        cp = @counter_pos.not_nil!
        mats << Mat.new(cp.rows, cp.cols)  # m
        mats << Mat.new(cp.rows, cp.cols)  # v
      else
        # token_emb
        mats << e.embedding.adam_tok.m << e.embedding.adam_tok.v
        e.blocks.each do |b|
          add_linear_adam(mats, b.attn.wq)
          add_linear_adam(mats, b.attn.wk)
          add_linear_adam(mats, b.attn.wv)
          add_linear_adam(mats, b.attn.wo)
          mats << b.ln1.adam_gamma.m << b.ln1.adam_gamma.v
          mats << b.ln1.adam_beta.m << b.ln1.adam_beta.v
          add_linear_adam(mats, b.ff.l1)
          add_linear_adam(mats, b.ff.l2)
          mats << b.ln2.adam_gamma.m << b.ln2.adam_gamma.v
          mats << b.ln2.adam_beta.m << b.ln2.adam_beta.v
        end
        mats << e.final_norm.adam_gamma.m << e.final_norm.adam_gamma.v
        mats << e.final_norm.adam_beta.m << e.final_norm.adam_beta.v
        add_linear_adam(mats, e.output.proj)
      end
    end

    @w_reads.each_with_index do |l, i|
      next if (has_bigram || @has_counter) && i == 0
      add_linear_adam(mats, l)
    end
    @w_writes.each_with_index do |l, i|
      next if (has_bigram || @has_counter) && i == 0
      add_linear_adam(mats, l)
    end

    add_linear_adam(mats, @w_router)
    mats
  end

  private def add_linear_adam(mats : Array(Mat), l : Linear)
    mats << l.adam_w.m << l.adam_w.v
    mats << l.adam_b.m << l.adam_b.v
  end

  # Collect gradient Mats in same order as all_weight_mats
  def all_grad_mats : Array(Mat)
    mats = [] of Mat
    has_bigram = !@bigram_table.nil?

    @experts.each_with_index do |e, i|
      if has_bigram && i == 0
        if wb = @w_bigram
          mats << wb.dw << wb.db
        end
      elsif @has_counter && i == 0
        # counter_pos gradient is d_stream (handled separately)
        # Add a zero-grad placeholder matching the weight
        cp = @counter_pos.not_nil!
        mats << Mat.new(cp.rows, cp.cols)
      else
        mats << e.embedding.d_token_emb
        e.blocks.each do |b|
          mats << b.attn.wq.dw << b.attn.wq.db
          mats << b.attn.wk.dw << b.attn.wk.db
          mats << b.attn.wv.dw << b.attn.wv.db
          mats << b.attn.wo.dw << b.attn.wo.db
          mats << b.ln1.dgamma << b.ln1.dbeta
          mats << b.ff.l1.dw << b.ff.l1.db
          mats << b.ff.l2.dw << b.ff.l2.db
          mats << b.ln2.dgamma << b.ln2.dbeta
        end
        mats << e.final_norm.dgamma << e.final_norm.dbeta
        mats << e.output.proj.dw << e.output.proj.db
      end
    end

    @w_reads.each_with_index do |l, i|
      next if (has_bigram || @has_counter) && i == 0
      mats << l.dw << l.db
    end
    @w_writes.each_with_index do |l, i|
      next if (has_bigram || @has_counter) && i == 0
      mats << l.dw << l.db
    end

    mats << @w_router.dw << @w_router.db
    mats
  end

  # How much expert_i's delta changes the stream direction: 1 - cos(before, after)
  # 0 = no change, 1 = orthogonal, 2 = reversed
  private def stream_cos_change(stream_before : Mat, delta : Mat) : Float64
    sb = stream_before.data
    dd = delta.data
    n = sb.size
    dot_ba = 0.0  # before · after
    norm_b = 0.0  # ||before||²
    norm_a = 0.0  # ||after||²
    n.times do |j|
      b = sb[j].to_f64
      d = dd[j].to_f64
      a = b + d
      dot_ba += b * a
      norm_b += b * b
      norm_a += a * a
    end
    denom = Math.sqrt(norm_b * norm_a)
    return 0.0 if denom < 1e-12
    cos = (dot_ba / denom).clamp(-1.0, 1.0)
    1.0 - cos
  end

  # Cosine similarity between two matrices (flattened)
  # 1.0 = same direction, 0 = orthogonal, -1 = opposite
  private def cosine_similarity(a : Mat, b : Mat) : Float64
    ad = a.data
    bd = b.data
    n = ad.size
    dot = 0.0
    na = 0.0
    nb = 0.0
    n.times do |j|
      av = ad[j].to_f64
      bv = bd[j].to_f64
      dot += av * bv
      na += av * av
      nb += bv * bv
    end
    denom = Math.sqrt(na * nb)
    return 0.0 if denom < 1e-12
    (dot / denom).clamp(-1.0, 1.0)
  end

  def router_weights_str : String
    if rp = @router_probs
      nr = rp.cols
      offset = stream_only?(0) ? 1 : 0
      parts = [] of String

      sn = @stream_norms
      sc = @stream_cosines

      if offset > 0
        parts << "E0=stream|#{"%.1f" % sn[0]}|cos#{"%.3f" % sc[0]}"
      end

      nr.times do |k|
        ei = k + offset
        parts << "E#{ei}=#{"%.3f" % rp[0, k]}|#{"%.1f" % sn[ei]}|Δ#{"%.3f" % sc[ei]}"
      end
      parts.join(", ")
    else
      "N/A"
    end
  end
end

end
