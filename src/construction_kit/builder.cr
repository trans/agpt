require "../microgpt"
require "./graph"
require "./compiler"

# Translates a validated graph into runnable model objects.
# Supports two execution modes:
#   - Legacy: extracts ModelConfig → instantiates CooperativeModel (hardcoded classes)
#   - Graph:  compiles GraphData → ExecutableGraph (graph-driven execution)

module ConstructionKit

  class Builder
    getter dataset : MicroGPT::CharDataset
    getter config : ModelConfig

    # Legacy mode
    getter model : MicroGPT::CooperativeModel?

    # Graph mode
    getter exec_graph : ExecutableGraph?
    getter? graph_mode : Bool

    def initialize(@config : ModelConfig, @graph_data : GraphData? = nil, @graph_mode : Bool = false)
      # Load dataset
      text = File.read(@config.data_file)
      @dataset = MicroGPT::CharDataset.new(text)

      if @graph_mode && (gd = @graph_data)
        # Graph-driven execution
        compiler = Compiler.new(@dataset.vocab_size, @config.seq_len)
        @exec_graph = compiler.compile(gd)
        @model = nil
      else
        # Legacy execution
        @exec_graph = nil
        build_legacy_model
      end
    end

    private def build_legacy_model
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

      if (eg = @exec_graph)
        loss = eg.train_step(input, targets[0], @config.learning_rate)
        StepResult.new(loss, "", 0.0)
      elsif (m = @model)
        loss = m.train_step(input, targets[0])
        grad_norm = compute_grad_norm
        StepResult.new(loss, m.router_weights_str, grad_norm)
      else
        raise "No model or graph built"
      end
    end

    private def compute_grad_norm : Float64
      m = @model
      return 0.0 unless m
      sum_sq = 0.0
      m.experts.each_with_index do |expert, i|
        next if (m.has_counter || !m.bigram_table.nil?) && i == 0
        expert.blocks.each do |b|
          sum_sq += mat_sq_sum(b.attn.wq.dw) + mat_sq_sum(b.attn.wk.dw) +
                    mat_sq_sum(b.attn.wv.dw) + mat_sq_sum(b.attn.wo.dw)
          sum_sq += mat_sq_sum(b.ff.l1.dw) + mat_sq_sum(b.ff.l2.dw)
        end
        sum_sq += mat_sq_sum(expert.output.proj.dw)
      end
      m.w_reads.each { |l| sum_sq += mat_sq_sum(l.dw) }
      m.w_writes.each { |l| sum_sq += mat_sq_sum(l.dw) }
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
      if seed_ids.size > @config.seq_len
        seed_ids = seed_ids[-@config.seq_len..]
      end

      if (eg = @exec_graph)
        generated = eg.generate(seed_ids, max_tokens, temperature, @config.seq_len)
        @dataset.decode(generated)
      elsif (m = @model)
        generated = m.generate(seed_ids, max_tokens, temperature)
        @dataset.decode(generated)
      else
        raise "No model or graph built"
      end
    end

    # Save all model weights to a directory
    def save_weights(dir : String)
      Dir.mkdir_p(dir)
      if (eg = @exec_graph)
        # Graph mode: save all weights in traversal order
        File.open(File.join(dir, "graph_weights.bin"), "wb") do |f|
          eg.all_weight_mats.each do |mat|
            f.write_bytes(mat.rows.to_i32, IO::ByteFormat::LittleEndian)
            f.write_bytes(mat.cols.to_i32, IO::ByteFormat::LittleEndian)
            mat.raw_data.each { |v| f.write_bytes(v, IO::ByteFormat::LittleEndian) }
          end
        end
        File.write(File.join(dir, "meta.json"), {
          config:     @config,
          vocab_size: @dataset.vocab_size,
          vocab:      @dataset.chars.map(&.to_s),
          graph_mode: true,
        }.to_json)
      elsif (m = @model)
        m.experts.each_with_index do |expert, i|
          expert.save(File.join(dir, "expert_#{i}.model"))
        end
        save_router(File.join(dir, "router.bin"))
        File.write(File.join(dir, "meta.json"), {
          config:     @config,
          vocab_size: @dataset.vocab_size,
          vocab:      @dataset.chars.map(&.to_s),
        }.to_json)
      end
    end

    # Load all model weights from a directory
    def load_weights(dir : String)
      if (eg = @exec_graph)
        path = File.join(dir, "graph_weights.bin")
        if File.exists?(path)
          begin
            File.open(path, "rb") do |f|
              eg.all_weight_mats.each do |mat|
                rows = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
                cols = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
                raise "Weight shape mismatch" unless rows == mat.rows && cols == mat.cols
                (rows * cols).times { |i| mat.raw_data[i] = f.read_bytes(Float32, IO::ByteFormat::LittleEndian) }
              end
            end
          rescue ex
            STDERR.puts "Could not load weights (#{ex.message}) — using fresh initialization"
          end
        end
      elsif (m = @model)
        m.experts.each_with_index do |expert, i|
          path = File.join(dir, "expert_#{i}.model")
          expert.load(path) if File.exists?(path)
        end
        router_path = File.join(dir, "router.bin")
        load_router(router_path) if File.exists?(router_path)
      end
    end

    private def save_router(path : String)
      m = @model
      return unless m
      File.open(path, "wb") do |f|
        m.router.weight_mats.each do |mat|
          f.write_bytes(mat.rows.to_i32, IO::ByteFormat::LittleEndian)
          f.write_bytes(mat.cols.to_i32, IO::ByteFormat::LittleEndian)
          mat.raw_data.each { |v| f.write_bytes(v, IO::ByteFormat::LittleEndian) }
        end
      end
    end

    private def load_router(path : String)
      m = @model
      return unless m
      File.open(path, "rb") do |f|
        m.router.weight_mats.each do |mat|
          rows = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          cols = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
          raise "Router shape mismatch" unless rows == mat.rows && cols == mat.cols
          (rows * cols).times { |i| mat.raw_data[i] = f.read_bytes(Float32, IO::ByteFormat::LittleEndian) }
        end
      end
    end

    # Get model summary info
    def summary : ModelSummary
      if (eg = @exec_graph)
        # Introspect the compiled graph for summary info
        expert_nodes = eg.nodes.values.select { |n| n.type == "embedding_table" }
        n_experts = Math.max(expert_nodes.size, 1)

        router_node = eg.nodes.values.find { |n| n.type == "global_router" }
        router_desc = router_node ? "global" : "none"
        router_params = router_node ? router_node.param_count : 0_i64

        # Group nodes by expert (by ID prefix)
        expert_infos = [] of ExpertInfo
        if expert_nodes.size > 0
          expert_nodes.each_with_index do |emb_node, i|
            # Find all nodes sharing this expert's prefix
            prefix = emb_node.id.rpartition(".")[0]
            expert_params = eg.nodes.values
              .select { |n| n.id.starts_with?(prefix) }
              .sum { |n| n.param_count }
            expert_infos << ExpertInfo.new(index: i, type: "transformer", spec: "graph-driven", params: expert_params)
          end
        else
          expert_infos << ExpertInfo.new(index: 0, type: "unknown", spec: "graph-driven", params: eg.total_params)
        end

        ModelSummary.new(
          total_params: eg.total_params,
          stream_dim: @config.stream_dim,
          seq_len: @config.seq_len,
          n_experts: n_experts,
          router: router_desc,
          router_params: router_params,
          experts: expert_infos,
          vocab_size: @dataset.vocab_size,
          data_file: @config.data_file,
        )
      elsif (m = @model)
        experts_info = @config.expert_specs.map_with_index do |spec, i|
          ExpertInfo.new(
            index: i,
            type: spec.type,
            spec: spec.to_spec_string,
            params: m.experts[i].param_count,
          )
        end
        ModelSummary.new(
          total_params: m.param_count,
          stream_dim: @config.stream_dim,
          seq_len: @config.seq_len,
          n_experts: @config.expert_specs.size,
          router: m.router.describe,
          router_params: m.router.param_count,
          experts: experts_info,
          vocab_size: @dataset.vocab_size,
          data_file: @config.data_file,
        )
      else
        raise "No model or graph built"
      end
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
