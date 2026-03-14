require "../microgpt"
require "./graph"

# Translates a validated graph into runnable Crystal model objects.
# This is the bridge between the visual construction kit and the
# existing MicroGPT engine.

module ConstructionKit

  class Builder
    getter dataset : MicroGPT::CharDataset
    getter model : MicroGPT::CooperativeModel
    getter config : ModelConfig

    def initialize(@config : ModelConfig)
      # Load dataset
      text = File.read(@config.data_file)
      @dataset = MicroGPT::CharDataset.new(text)

      # Build expert configs
      expert_configs = @config.expert_specs.map do |spec|
        cfg = MicroGPT::Config.new
        cfg.vocab_size = @dataset.vocab_size
        cfg.d_model = spec.d_model
        cfg.n_heads = Math.max(1, spec.d_model // 16)
        cfg.n_layers = spec.n_layers
        cfg.d_ff = spec.d_ff
        cfg.seq_len = @config.seq_len
        cfg.learning_rate = @config.learning_rate
        cfg
      end

      # Build router
      nr = @config.has_counter ? expert_configs.size - 1 : expert_configs.size
      nr = Math.max(nr, 1)
      router = case @config.router_type
               when "context"
                 MicroGPT::ContextRouter.new(nr, @config.stream_dim, @dataset.vocab_size)
               when "gated"
                 MicroGPT::GatedRouter.new(nr, @config.stream_dim, @dataset.vocab_size)
               else
                 MicroGPT::GlobalRouter.new(nr, @config.stream_dim)
               end
      router.epsilon = @config.router_epsilon

      # Build cooperative model
      @model = MicroGPT::CooperativeModel.new(
        expert_configs,
        @config.stream_dim,
        @config.has_counter,
        router: router,
      )
    end

    # Run a single training step, return metrics
    def train_step : StepResult
      input, targets = @dataset.sample(@config.seq_len, 0)
      loss = @model.train_step(input, targets[0])
      grad_norm = compute_grad_norm
      StepResult.new(loss, @model.router_weights_str, grad_norm)
    end

    # Compute L2 norm of all gradients (after backward pass, before they're cleared)
    private def compute_grad_norm : Float64
      sum_sq = 0.0
      @model.experts.each_with_index do |expert, i|
        next if (@model.has_counter || !@model.bigram_table.nil?) && i == 0
        expert.blocks.each do |b|
          sum_sq += mat_sq_sum(b.attn.wq.dw) + mat_sq_sum(b.attn.wk.dw) +
                    mat_sq_sum(b.attn.wv.dw) + mat_sq_sum(b.attn.wo.dw)
          sum_sq += mat_sq_sum(b.ff.l1.dw) + mat_sq_sum(b.ff.l2.dw)
        end
        sum_sq += mat_sq_sum(expert.output.proj.dw)
      end
      @model.w_reads.each { |l| sum_sq += mat_sq_sum(l.dw) }
      @model.w_writes.each { |l| sum_sq += mat_sq_sum(l.dw) }
      Math.sqrt(sum_sq)
    end

    private def mat_sq_sum(m : MicroGPT::Mat) : Float64
      sum = 0.0
      m.data.each { |v| sum += v.to_f64 * v.to_f64 }
      sum
    end

    # Generate text from a seed
    def generate(seed_text : String, max_tokens : Int32 = 100, temperature : Float64 = 0.8) : String
      seed_ids = @dataset.encode(seed_text)
      # Use last seq_len tokens if seed is too long
      if seed_ids.size > @config.seq_len
        seed_ids = seed_ids[-@config.seq_len..]
      end
      generated = @model.generate(seed_ids, max_tokens, temperature)
      @dataset.decode(generated)
    end

    # Save all model weights to a directory
    def save_weights(dir : String)
      Dir.mkdir_p(dir)
      @model.experts.each_with_index do |expert, i|
        expert.save(File.join(dir, "expert_#{i}.model"))
      end
      # Save router weights
      save_router(File.join(dir, "router.bin"))
      # Save metadata
      File.write(File.join(dir, "meta.json"), {
        config:     @config,
        vocab_size: @dataset.vocab_size,
        vocab:      @dataset.chars.map(&.to_s),
      }.to_json)
    end

    # Load all model weights from a directory
    def load_weights(dir : String)
      @model.experts.each_with_index do |expert, i|
        path = File.join(dir, "expert_#{i}.model")
        if File.exists?(path)
          expert.load(path)
        end
      end
      router_path = File.join(dir, "router.bin")
      load_router(router_path) if File.exists?(router_path)
    end

    private def save_router(path : String)
      File.open(path, "wb") do |f|
        @model.router.weight_mats.each do |mat|
          f.write_bytes(mat.rows.to_i32, IO::ByteFormat::LittleEndian)
          f.write_bytes(mat.cols.to_i32, IO::ByteFormat::LittleEndian)
          mat.raw_data.each { |v| f.write_bytes(v, IO::ByteFormat::LittleEndian) }
        end
      end
    end

    private def load_router(path : String)
      File.open(path, "rb") do |f|
        @model.router.weight_mats.each do |mat|
          rows = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          cols = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          raise "Router shape mismatch" unless rows == mat.rows && cols == mat.cols
          (rows * cols).times { |i| mat.raw_data[i] = f.read_bytes(Float32, IO::ByteFormat::LittleEndian) }
        end
      end
    end

    # Get model summary info
    def summary : ModelSummary
      experts_info = @config.expert_specs.map_with_index do |spec, i|
        ExpertInfo.new(
          index: i,
          type: spec.type,
          spec: spec.to_spec_string,
          params: @model.experts[i].param_count,
        )
      end

      ModelSummary.new(
        total_params: @model.param_count,
        stream_dim: @config.stream_dim,
        seq_len: @config.seq_len,
        n_experts: @config.expert_specs.size,
        router: @model.router.describe,
        router_params: @model.router.param_count,
        experts: experts_info,
        vocab_size: @dataset.vocab_size,
        data_file: @config.data_file,
      )
    end
  end

  struct StepResult
    include JSON::Serializable
    property loss : Float64
    property router_weights : String
    property grad_norm : Float64

    def initialize(@loss, @router_weights, @grad_norm)
    end
  end

  struct ExpertInfo
    include JSON::Serializable
    property index : Int32
    property type : String
    property spec : String
    property params : Int64

    def initialize(@index, @type, @spec, @params)
    end
  end

  struct ModelSummary
    include JSON::Serializable
    property total_params : Int64
    property stream_dim : Int32
    property seq_len : Int32
    property n_experts : Int32
    property router : String
    property router_params : Int64
    property experts : Array(ExpertInfo)
    property vocab_size : Int32
    property data_file : String

    def initialize(@total_params, @stream_dim, @seq_len, @n_experts,
                   @router, @router_params, @experts, @vocab_size, @data_file)
    end
  end
end
