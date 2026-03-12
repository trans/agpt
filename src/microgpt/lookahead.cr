require "../microgpt"

module MicroGPT

# =============================================================================
# Mask Builder — pre-computed attention masks
# =============================================================================

def self.build_mask(seq_len : Int32, k : Int32) : Mat
  mask = Mat.new(seq_len, seq_len)
  seq_len.times do |i|
    seq_len.times do |j|
      val = if k == 0
        # Standard causal: block everything after position i
        j > i ? -1e9 : 0.0
      else
        # Zebra: block only the k positions after i (the gap)
        # Sees past (0..i) and future beyond the gap (i+k+1..)
        (j > i && j <= i + k) ? -1e9 : 0.0
      end
      mask[i, j] = val
    end
  end
  mask
end

# Inverted causal mask: position t sees t..end (self + future, no past)
def self.build_inverted_causal_mask(seq_len : Int32) : Mat
  mask = Mat.new(seq_len, seq_len)
  seq_len.times do |i|
    seq_len.times do |j|
      mask[i, j] = j < i ? -1e9 : 0.0
    end
  end
  mask
end

# =============================================================================
# Lookahead Model — wraps MiniGPT with multiple output heads and zebra masks
# =============================================================================

class LookaheadModel
  getter model : MiniGPT
  getter extra_heads : Array(OutputHead)
  getter masks : Array(Mat)
  getter n_ahead : Int32

  def initialize(@model : MiniGPT, @n_ahead : Int32 = 2)
    config = model.config
    @masks = Array(Mat).new(@n_ahead + 1) { |k| MicroGPT.build_mask(config.seq_len, k) }
    @extra_heads = Array(OutputHead).new(@n_ahead) { OutputHead.new(config.d_model, config.vocab_size) }
  end

  def param_count : Int64
    count = @model.param_count
    @extra_heads.each { |h| count += h.proj.w.data.size + h.proj.b.data.size }
    count
  end

  # Forward through transformer with a specific mask
  private def transformer_forward(ids : Array(Int32), mask : Mat) : Mat
    x = @model.embedding.forward(ids)
    @model.blocks.each { |b| x = b.forward(x, mask) }
    @model.final_norm.forward(x)
  end

  # Training step with multiple output heads and zebra masks
  # Returns {avg_loss, per_head_losses}
  def train_step(input_ids : Array(Int32), targets : Array(Array(Int32))) : {Float64, Array(Float64)}
    n_total = @n_ahead + 1
    losses = Array(Float64).new(n_total, 0.0)

    # --- Pass k=0: standard causal (uses base model's output head) ---
    hidden_0 = transformer_forward(input_ids, @masks[0])
    logits_0 = @model.output.forward(hidden_0)
    loss_0, grad_0 = @model.output.loss_and_backward(logits_0, targets[0])
    losses[0] = loss_0

    # Backward through shared layers (sets gradients)
    grad_0 = @model.final_norm.backward(grad_0)
    @model.blocks.reverse_each { |b| grad_0 = b.backward(grad_0) }
    @model.embedding.backward(grad_0)

    # --- Passes k=1..n_ahead: zebra masks with extra output heads ---
    @n_ahead.times do |ki|
      k = ki + 1
      hidden_k = transformer_forward(input_ids, @masks[k])
      logits_k = @extra_heads[ki].forward(hidden_k)
      loss_k, grad_k = @extra_heads[ki].loss_and_backward(logits_k, targets[k])
      losses[k] = loss_k

      # Backward through shared layers (accumulates gradients)
      grad_k = @model.final_norm.backward_accumulate(grad_k)
      @model.blocks.reverse_each { |b| grad_k = b.backward_accumulate(grad_k) }
      @model.embedding.backward_accumulate(grad_k)
    end

    # --- Update all weights once ---
    lr = @model.config.learning_rate
    @model.embedding.update(lr)
    @model.blocks.each &.update(lr)
    @model.final_norm.update(lr)
    @model.output.update(lr)
    @extra_heads.each &.update(lr)

    avg = losses.sum / n_total
    {avg, losses}
  end

  def generate(start_ids : Array(Int32), max_tokens : Int32, temperature : Float64 = 1.0) : Array(Int32)
    # Generation always uses k=0 (standard causal) — the reliable head
    @model.generate(start_ids, max_tokens, temperature)
  end
end

# =============================================================================
# Future Model — one causal head + one anti-causal (future) head
# =============================================================================

class FutureModel
  getter model : MiniGPT
  getter future_head : OutputHead
  getter causal_mask : Mat
  getter future_mask : Mat

  def initialize(@model : MiniGPT)
    config = model.config
    @causal_mask = MicroGPT.build_mask(config.seq_len, 0)
    @future_mask = MicroGPT.build_inverted_causal_mask(config.seq_len)
    @future_head = OutputHead.new(config.d_model, config.vocab_size)
  end

  def param_count : Int64
    @model.param_count + @future_head.proj.w.data.size + @future_head.proj.b.data.size
  end

  def train_step(input_ids : Array(Int32), targets : Array(Int32)) : {Float64, Float64, Float64}
    # --- Causal pass (sets gradients) ---
    x0 = @model.embedding.forward(input_ids)
    @model.blocks.each { |b| x0 = b.forward(x0, @causal_mask) }
    h0 = @model.final_norm.forward(x0)
    logits_0 = @model.output.forward(h0)
    loss_0, grad_0 = @model.output.loss_and_backward(logits_0, targets)
    grad_0 = @model.final_norm.backward(grad_0)
    @model.blocks.reverse_each { |b| grad_0 = b.backward(grad_0) }
    @model.embedding.backward(grad_0)

    # --- Future pass (anti-causal, accumulates gradients) ---
    x1 = @model.embedding.forward(input_ids)
    @model.blocks.each { |b| x1 = b.forward(x1, @future_mask) }
    h1 = @model.final_norm.forward(x1)
    logits_1 = @future_head.forward(h1)
    loss_1, grad_1 = @future_head.loss_and_backward(logits_1, targets)
    grad_1 = @model.final_norm.backward_accumulate(grad_1)
    @model.blocks.reverse_each { |b| grad_1 = b.backward_accumulate(grad_1) }
    @model.embedding.backward_accumulate(grad_1)

    # --- Update all weights once ---
    lr = @model.config.learning_rate
    @model.embedding.update(lr)
    @model.blocks.each &.update(lr)
    @model.final_norm.update(lr)
    @model.output.update(lr)
    @future_head.update(lr)

    avg = (loss_0 + loss_1) / 2.0
    {avg, loss_0, loss_1}
  end

  def generate(start_ids : Array(Int32), max_tokens : Int32, temperature : Float64 = 1.0) : Array(Int32)
    @model.generate(start_ids, max_tokens, temperature)
  end
end

end
