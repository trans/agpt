require "../microgpt"
require "../agpt"
require "jargon"
require "yaml"

module MicroGPT
  AGPT_TOKENIZER_TAG = "char.sorted_unique.v1"

  SCHEMA = %({
    "type": "object",
    "positional": ["file"],
    "properties": {
      "file": {
        "type": "string",
        "description": "Input text file for training/vocab",
        "short": "f"
      },
      "steps": {
        "type": "integer",
        "description": "Training steps (0 = inference only)",
        "default": 1000,
        "short": "s"
      },
      "seed": {
        "type": "integer",
        "description": "Deterministic RNG seed for model init and sampling"
      },
      "heads": {
        "type": "string",
        "description": "Head type",
        "enum": ["uniform", "exponential", "prime", "pyramid", "ones"],
        "default": "uniform"
      },
      "backend": {
        "type": "string",
        "description": "Compute backend",
        "enum": ["crystal", "openblas", "cublas"],
        "default": "crystal",
        "short": "b"
      },
      "seq-len": {
        "type": "integer",
        "description": "Sequence length (context window)",
        "default": 128
      },
      "lookahead": {
        "type": "integer",
        "description": "Lookahead heads (0=none, -1=future model)",
        "default": 0
      },
      "d-model": {
        "type": "integer",
        "description": "Model embedding dimension",
        "default": 64,
        "short": "d"
      },
      "n-layers": {
        "type": "integer",
        "description": "Number of transformer layers",
        "default": 2,
        "short": "n"
      },
      "eval": {
        "type": "string",
        "description": "Eval prompts file (one per line)",
        "short": "e"
      },
      "model": {
        "type": "string",
        "description": "Model checkpoint path (default: derived from input file)",
        "short": "m"
      },
      "no-save": {
        "type": "boolean",
        "description": "Skip saving model after training",
        "default": false
      },
      "lr": {
        "type": "number",
        "description": "Learning rate",
        "default": 0.0003
      },
      "compare": {
        "type": "string",
        "description": "Train multiple models: d64n2,d128n2,d128n3",
        "short": "c"
      },
      "cooperative": {
        "type": "string",
        "description": "Cooperative ensemble: expert specs (e.g. d16n1,d32n2,d32n2)",
        "short": "C"
      },
      "stream-dim": {
        "type": "integer",
        "description": "Shared stream dimension for cooperative mode",
        "default": 64
      },
      "no-counter": {
        "type": "boolean",
        "description": "Disable counter expert (expert 0 uses tokens normally)",
        "default": false
      },
      "active-dims": {
        "type": "integer",
        "description": "Active stream dimensions (0=full, masks unused dims)",
        "default": 0
      },
      "config": {
        "type": "string",
        "description": "YAML config file with model definitions"
      },
      "run": {
        "type": "string",
        "description": "Model ID to run from config file (or comma-separated IDs)"
      },
      "bigram-off-at": {
        "type": "integer",
        "description": "Step at which to detach bigram expert (0=never)",
        "default": 0
      },
      "router": {
        "type": "string",
        "description": "Router type: global, context, gated",
        "enum": ["global", "context", "gated"],
        "default": "global"
      },
      "agpt": {
        "type": "boolean",
        "description": "Train with AGPT replay-prefix mode instead of sampled windows",
        "default": false
      },
      "agpt-max-starts": {
        "type": "integer",
        "description": "Limit AGPT trie construction to N evenly distributed corpus start positions (0 = all)",
        "default": 10000
      },
      "agpt-start-offset": {
        "type": "integer",
        "description": "Deterministic offset applied to evenly distributed AGPT start positions",
        "default": 0
      },
      "agpt-progress": {
        "type": "integer",
        "description": "Print AGPT trie build progress every N start positions (0 = off)",
        "default": 0
      },
      "agpt-save-index": {
        "type": "string",
        "description": "Save the built AGPT trie index to this path"
      },
      "agpt-load-index": {
        "type": "string",
        "description": "Load a previously saved AGPT trie index from this path"
      },
      "build-only": {
        "type": "boolean",
        "description": "Build/load and report model state without training or generation",
        "default": false
      }
    },
    "required": ["file"]
  })

  # Parse a model spec like "d64n2", "d4x16n2ff512", or "d1+2+4+8+16+32n2"
  # d64     = one head, dim 64
  # d4x16   = 4 uniform heads of dim 16 (d_model=64)
  # d1+2+4  = heterogeneous heads (d_model=7)
  record ModelSpec, d_model : Int32, n_layers : Int32, d_ff : Int32,
                    head_dims : Array(Int32)? do
    def label : String
      ff_str = d_ff == d_model * 4 ? "" : "ff#{d_ff}"
      d_str = if hd = head_dims
        if hd.all? { |h| h == hd[0] }
          "d#{hd.size}x#{hd[0]}"
        else
          "d" + hd.join("+")
        end
      else
        "d#{d_model}"
      end
      "#{d_str}n#{n_layers}#{ff_str}"
    end
  end

  def self.parse_specs(spec_str : String) : Array(ModelSpec)
    spec_str.split(",").map do |s|
      s = s.strip
      # Match: d<dims>n<layers>[ff<dim>]
      unless m = s.match(/^d([\d+x]+)n(\d+)(?:ff(\d+))?$/)
        raise "Invalid model spec '#{s}' — format: d64n2, d4x16n2, or d1+2+4n2"
      end
      dims_str = m[1]
      n = m[2].to_i

      head_dims : Array(Int32)? = nil
      if dims_str.includes?("+")
        # Heterogeneous: d1+2+4+8
        head_dims = dims_str.split("+").map(&.to_i)
        d = head_dims.not_nil!.sum
      elsif dims_str.includes?("x")
        # Uniform: d4x16 = 4 heads of dim 16
        parts = dims_str.split("x")
        count = parts[0].to_i
        dim = parts[1].to_i
        head_dims = Array.new(count, dim)
        d = count * dim
      else
        # Single head: d64
        d = dims_str.to_i
        head_dims = [d]
      end

      ff = m[3]? ? m[3].to_i : d * 4
      ModelSpec.new(d, n, ff, head_dims)
    end
  end

  # Log a result to data/results.tsv (append-only)
  def self.log_result(id : String, params : Int64, steps : Int32, final_loss : Float64,
                      extra : String = "", data_file : String = "")
    log_path = File.join(File.dirname(data_file.empty? ? "." : data_file), "results.tsv")
    header = "timestamp\tid\tparams\tsteps\tfinal_loss\textra"
    unless File.exists?(log_path)
      File.write(log_path, header + "\n")
    end
    ts = Time.local.to_s("%Y-%m-%d %H:%M")
    line = "#{ts}\t#{id}\t#{params}\t#{steps}\t#{"%.4f" % final_loss}\t#{extra}"
    File.open(log_path, "a") { |f| f.puts line }
    puts "Result logged: #{log_path}"
  end

  # Load a model config from YAML and return CLI-equivalent parameters
  def self.load_yaml_config(config_path : String, run_id : String) : Hash(String, YAML::Any)
    yaml = YAML.parse(File.read(config_path))
    unless model = yaml[run_id]?
      available = yaml.as_h.keys.map(&.to_s).join(", ")
      raise "Model '#{run_id}' not found in #{config_path}. Available: #{available}"
    end
    model.as_h.transform_keys(&.to_s)
  end

  def self.main
    cli = Jargon.cli("microgpt", json: SCHEMA)
    cli.run do |result|
      filename    = result["file"].as_s
      steps       = result["steps"].as_i64.to_i
      seed_value  = result["seed"]?.try(&.as_i64)
      head_type   = result["heads"].as_s
      backend     = result["backend"].as_s
      seq_len     = result["seq-len"].as_i64.to_i
      lookahead   = result["lookahead"].as_i64.to_i
      d_model     = result["d-model"].as_i64.to_i
      n_layers    = result["n-layers"].as_i64.to_i
      eval_file   = result["eval"]?.try(&.as_s)
      model_path  = result["model"]?.try(&.as_s)
      no_save     = result["no-save"]?.try(&.as_bool) || false
      lr          = result["lr"].as_f
      compare     = result["compare"]?.try(&.as_s)
      cooperative  = result["cooperative"]?.try(&.as_s)
      stream_dim   = result["stream-dim"].as_i64.to_i
      no_counter   = result["no-counter"]?.try(&.as_bool) || false
      active_dims  = result["active-dims"].as_i64.to_i
      config_file  = result["config"]?.try(&.as_s)
      run_id       = result["run"]?.try(&.as_s)
      use_bigram   = false
      use_trigram  = false
      use_calculator = false
      bigram_off_at = result["bigram-off-at"].as_i64.to_i
      router_type  = result["router"].as_s
      agpt_mode    = result["agpt"]?.try(&.as_bool) || false
      agpt_max_starts = result["agpt-max-starts"].as_i64.to_i
      agpt_start_offset = result["agpt-start-offset"].as_i64.to_i
      agpt_progress = result["agpt-progress"].as_i64.to_i
      agpt_save_index = result["agpt-save-index"]?.try(&.as_s)
      agpt_load_index = result["agpt-load-index"]?.try(&.as_s)
      build_only = result["build-only"]?.try(&.as_bool) || false

      # --- Load config from YAML if provided ---
      # YAML provides base values; explicit CLI flags override them.
      # We detect "explicit CLI" by checking if the raw JSON has non-default values.
      if cf = config_file
        if rid = run_id
          yaml = load_yaml_config(cf, rid)
          mode = yaml["mode"]?.try(&.as_s) || "single"

          # Helper: use YAML value unless CLI explicitly overrode
          # (Jargon sets defaults, so we check against known defaults)
          steps       = result["steps"].as_i64.to_i    == 1000  ? (yaml["steps"]?.try(&.as_i) || steps) : steps
          seq_len     = result["seq-len"].as_i64.to_i   == 128   ? (yaml["seq_len"]?.try(&.as_i) || seq_len) : seq_len
          lr          = result["lr"].as_f               == 3e-4  ? (yaml["lr"]?.try(&.as_f) || lr) : lr
          backend     = result["backend"].as_s          == "crystal" ? (yaml["backend"]?.try(&.as_s) || backend) : backend
          d_model     = result["d-model"].as_i64.to_i   == 64    ? (yaml["d_model"]?.try(&.as_i) || d_model) : d_model
          n_layers    = result["n-layers"].as_i64.to_i  == 2     ? (yaml["n_layers"]?.try(&.as_i) || n_layers) : n_layers
          # no_pos_emb removed — RoPE handles position now

          case mode
          when "cooperative"
            cooperative = yaml["experts"]?.try(&.as_s)
            stream_dim  = result["stream-dim"].as_i64.to_i == 64 ? (yaml["stream_dim"]?.try(&.as_i) || stream_dim) : stream_dim
            active_dims = result["active-dims"].as_i64.to_i == 0 ? (yaml["active_dims"]?.try(&.as_i) || active_dims) : active_dims
            no_counter  = yaml["counter"]?.try(&.as_bool) == false
            use_bigram  = yaml["bigram"]?.try(&.as_bool) || false
            use_trigram = yaml["trigram"]?.try(&.as_bool) || false
            use_calculator = yaml["calculator"]?.try(&.as_bool) || false
            use_bigram = true if use_trigram  # trigram uses the bigram slot
            use_bigram = true if use_calculator  # calculator uses the bigram slot
            router_type = result["router"].as_s == "global" ? (yaml["router"]?.try(&.as_s) || router_type) : router_type
          end
          no_save = true  # config runs don't auto-save (use --model to save)
          puts "Config: #{cf} → #{rid}"
        else
          # No run ID: list available configs
          yaml = YAML.parse(File.read(cf))
          puts "Available models in #{cf}:"
          yaml.as_h.each do |k, v|
            mode = v["mode"]?.try(&.as_s) || "single"
            puts "  #{k} (#{mode})"
          end
          exit 0
        end
      end

      case backend
      when "openblas" then MicroGPT.use_openblas!
      when "cublas"   then MicroGPT.use_cublas!
      else                 MicroGPT.use_crystal!
      end

      if seed = seed_value
        Random.thread_default.new_seed(seed.to_u64)
      end

      unless File.exists?(filename)
        STDERR.puts "File not found: #{filename}"
        exit 1
      end

      text = File.read(filename)
      dataset = CharDataset.new(text)

      if agpt_mode
        if cooperative
          STDERR.puts "AGPT MVP currently supports single-model mode only"
          exit 1
        end
        if compare
          STDERR.puts "AGPT MVP does not support compare mode yet"
          exit 1
        end
        if lookahead != 0
          STDERR.puts "AGPT MVP does not support lookahead/future mode yet"
          exit 1
        end
      end

      # === Cooperative mode: μGPT ensemble with shared stream ===
      if coop_str = cooperative
        specs = parse_specs(coop_str)
        configs = specs.map do |spec|
          cfg = Config.new
          cfg.vocab_size = dataset.vocab_size
          cfg.d_model = spec.d_model
          cfg.n_heads = Math.max(1, spec.d_model // 16)
          cfg.n_layers = spec.n_layers
          cfg.d_ff = spec.d_ff
          cfg.seq_len = seq_len
          cfg.learning_rate = lr
          cfg
        end

        has_counter = !no_counter && !use_bigram

        # Build pluggable router
        nr = has_counter ? specs.size - 1 : specs.size
        router = case router_type
                 when "context" then ContextRouter.new(nr, stream_dim, dataset.vocab_size)
                 when "gated"   then GatedRouter.new(nr, stream_dim, dataset.vocab_size)
                 else                GlobalRouter.new(nr, stream_dim)
                 end

        coop = CooperativeModel.new(configs, stream_dim, has_counter, router: router)
        coop.active_stream_dims = active_dims if active_dims > 0

        # Attach algorithmic expert if requested
        if use_calculator
          calc = CalculatorExpert.new(dataset.vocab_size, dataset.id_to_char, dataset.char_to_id)
          coop.attach_bigram(calc)
        elsif use_trigram
          trigram = TrigramTable.new(dataset.data, dataset.vocab_size)
          coop.attach_bigram(trigram)
        elsif use_bigram
          bigram = BigramTable.new(dataset.data, dataset.vocab_size)
          coop.attach_bigram(bigram)
        end

        puts "Cooperative μGPT ensemble"
        e0_type = use_calculator ? "calculator" : (use_trigram ? "trigram" : (use_bigram ? "bigram" : (has_counter ? "counter" : "transformer")))
        puts "  Experts: #{specs.size} (E0=#{e0_type})"
        specs.each_with_index do |s, i|
          if (use_bigram || use_trigram || use_calculator) && i == 0
            wb_params = dataset.vocab_size * stream_dim + stream_dim  # W + b
            expert_type = use_calculator ? "calculator" : (use_trigram ? "trigram" : "bigram")
            puts "  E#{i}: #{expert_type} expert params=#{wb_params} [#{dataset.vocab_size}→#{stream_dim} projection]"
          elsif has_counter && i == 0
            cp_params = seq_len * stream_dim
            puts "  E#{i}: counter params=#{cp_params} [#{seq_len}×#{stream_dim} pos signal]"
          else
            puts "  E#{i}: #{s.label} params=#{coop.experts[i].param_count}"
          end
        end
        puts "  Stream dim: #{stream_dim}#{active_dims > 0 ? " (active: #{active_dims})" : ""}"
        puts "  Router: #{coop.router.describe} (#{coop.router.param_count} params)"
        puts "  Total params: #{coop.param_count}"
        puts "  seq_len=#{seq_len} lr=#{lr} backend=#{backend}"

        # Create GPU WeightStore for cuBLAS backend (enables bulk Adam)
        weight_store = nil
        if MicroGPT.backend.is_a?(CuBLASBackend)
          weight_mats = coop.all_weight_mats
          adam_mats = coop.all_adam_mats
          weight_store = WeightStore.new(weight_mats, adam_mats)
        end
        puts

        avg_loss = 0.0
        bigram_detached = false
        steps.times do |step|
          # Bigram cutoff: detach at specified step
          if bigram_off_at > 0 && step == bigram_off_at && !bigram_detached
            puts "\n*** BIGRAM DETACHED at step #{step} (avg_loss = #{"%.4f" % avg_loss}) ***\n"
            coop.detach_bigram
            bigram_detached = true
          end

          input, targets = dataset.sample(seq_len, 0)
          loss = coop.train_step(input, targets[0])
          avg_loss = step == 0 ? loss : 0.99 * avg_loss + 0.01 * loss

          GC.collect if step % 10 == 0

          # Fine-grained logging around bigram cutoff (every step for 200 steps after)
          near_cutoff = bigram_detached && step >= bigram_off_at && step < bigram_off_at + 200
          log_interval = near_cutoff ? 10 : 50

          if step % log_interval == 0 || (step == bigram_off_at)
            puts "Step #{step}/#{steps} (epoch #{dataset.epoch}): loss = #{"%.4f" % loss} avg = #{"%.4f" % avg_loss} [#{coop.router_weights_str}]"
            if step % 50 == 0
              seed = input[0, 16]
              generated = coop.generate(seed, 100, temperature: 0.8)
              puts "  seed: #{dataset.decode(seed)}"
              puts "  gen:  #{dataset.decode(generated)}"
              puts
            end
          end

          # Per-position loss breakdown every 1000 steps
          if step > 0 && step % 1000 == 0
            # Forward a sample and compute per-position CE loss
            sample_in, sample_tgt = dataset.sample(seq_len, 0)
            logits = coop.forward(sample_in)
            probs = MicroGPT.backend.softmax_rows(logits)

            eq_id = dataset.char_to_id['=']?
            nl_id = dataset.char_to_id['\n']?

            eq_losses = [] of Float64   # positions after '='
            other_losses = [] of Float64 # positions before/at '='
            in_answer = false

            sample_in.size.times do |pos|
              tok = sample_in[pos]
              tgt = sample_tgt[0][pos]
              ce = -Math.log(probs[pos, tgt] + 1e-10)

              if nl_id && tok == nl_id
                in_answer = false
              end

              if in_answer
                eq_losses << ce
              else
                other_losses << ce
              end

              if eq_id && tok == eq_id
                in_answer = true
              end
            end

            eq_avg = eq_losses.empty? ? 0.0 : eq_losses.sum / eq_losses.size
            other_avg = other_losses.empty? ? 0.0 : other_losses.sum / other_losses.size
            puts "  [loss breakdown] equation=#{"%.4f" % other_avg} (#{other_losses.size} pos) | answer=#{"%.4f" % eq_avg} (#{eq_losses.size} pos)"
          end
        end

        # Download weights from GPU before generation/logging
        if ws = weight_store
          ws.download_all
        end

        if steps > 0
          puts "Final avg loss: #{"%.4f" % avg_loss}"
          puts "Router: #{coop.router_weights_str}"
          # Log result
          rid_str = run_id || "coop-#{coop_str}"
          extra = "router=#{coop.router_weights_str} router_type=#{router_type} stream=#{stream_dim}"
          extra += " active=#{active_dims}" if active_dims > 0
          extra += " counter=#{has_counter}"
          extra += " rope"
          extra += use_calculator ? " calculator" : (use_trigram ? " trigram" : " bigram") if use_bigram
          log_result(rid_str, coop.param_count, steps, avg_loss, extra, filename)
        end
        puts
        puts "Final generation:"
        seed = dataset.data[0, 16]
        generated = coop.generate(seed, 500, temperature: 0.8)
        puts dataset.decode(generated)

        # Eval mode
        if ef = eval_file
          if File.exists?(ef)
            puts
            puts "=" * 60
            puts "Eval prompts from #{ef}:"
            puts "=" * 60
            File.each_line(ef) do |prompt|
              prompt = prompt.strip
              next if prompt.empty?
              seed_ids = dataset.encode(prompt)
              generated = coop.generate(seed_ids, 20, temperature: 0.1)
              output = dataset.decode(generated).split('\n').first
              puts "  #{prompt} -> #{output}"
            end
          end
        end

        next
      end

      # === Compare mode: train multiple models side-by-side ===
      if compare_str = compare
        specs = parse_specs(compare_str)
        models = specs.map do |spec|
          cfg = Config.new
          cfg.vocab_size = dataset.vocab_size
          cfg.d_model = spec.d_model
          cfg.n_heads = spec.d_model // 16
          cfg.n_layers = spec.n_layers
          cfg.d_ff = spec.d_ff
          cfg.seq_len = seq_len
          cfg.learning_rate = lr
          if hd = spec.head_dims
            MiniGPT.new(cfg, hd)
          else
            MiniGPT.new(cfg)
          end
        end

        # Load checkpoints
        base = filename.sub(/\.[^.]+$/, "")
        save_paths = specs.map { |s| "#{base}_#{s.label}.model" }
        models.each_with_index do |m, i|
          if File.exists?(save_paths[i])
            begin
              m.load(save_paths[i])
              puts "Loaded: #{save_paths[i]}"
            rescue ex
              puts "Skipping checkpoint #{save_paths[i]} (#{ex.message})"
            end
          end
        end

        puts "Compare mode: #{specs.size} models"
        specs.each_with_index do |s, i|
          puts "  [#{s.label}] params=#{models[i].param_count} d_model=#{s.d_model} n_layers=#{s.n_layers} d_ff=#{s.d_ff}"
        end
        puts "  seq_len=#{seq_len} lr=#{lr} backend=#{backend}"
        puts

        avg_losses = Array(Float64).new(specs.size, 0.0)
        max_seq = models.map(&.config.seq_len).max

        steps.times do |step|
          # Same batch for all models
          input, targets = dataset.sample(max_seq, 0)

          models.each_with_index do |m, i|
            sl = m.config.seq_len
            inp = input[0, sl]
            tgt = targets[0][0, sl]
            loss = m.train_step(inp, tgt)
            avg_losses[i] = step == 0 ? loss : 0.99 * avg_losses[i] + 0.01 * loss
          end

          GC.collect if step % 10 == 0

          if step % 50 == 0
            puts "Step #{step}/#{steps} (epoch #{dataset.epoch}):"
            specs.each_with_index do |s, i|
              puts "  [#{s.label}] avg=#{"%.4f" % avg_losses[i]}"
            end

            # Generate from first model as sample
            seed = input[0, 16]
            generated = models[0].generate(seed, 100, temperature: 0.8)
            puts "  [#{specs[0].label}] gen: #{dataset.decode(generated)}"
            puts
          end
        end

        if steps > 0
          puts "Final results:"
          specs.each_with_index do |s, i|
            puts "  [#{s.label}] avg_loss=#{"%.4f" % avg_losses[i]} params=#{models[i].param_count}"
          end
          unless no_save
            models.each_with_index do |m, i|
              m.save(save_paths[i])
              puts "  Saved: #{save_paths[i]}"
            end
          end
        end

        puts
        puts "Final generation (each model):"
        seed = dataset.data[0, 16]
        specs.each_with_index do |s, i|
          generated = models[i].generate(seed, 200, temperature: 0.8)
          puts "--- [#{s.label}] ---"
          puts dataset.decode(generated)
          puts
        end

        # Eval mode for compare
        if ef = eval_file
          if File.exists?(ef)
            puts "=" * 60
            puts "Eval prompts from #{ef}:"
            puts "=" * 60
            File.each_line(ef) do |prompt|
              prompt = prompt.strip
              next if prompt.empty?
              seed_ids = dataset.encode(prompt)
              results = specs.map_with_index do |s, i|
                gen = models[i].generate(seed_ids, 20, temperature: 0.1)
                output = dataset.decode(gen).split('\n').first
                "#{s.label}: #{output}"
              end
              puts "  #{prompt}"
              results.each { |r| puts "    #{r}" }
            end
          end
        end

        next  # skip single-model path
      end

      config = Config.new
      config.vocab_size = dataset.vocab_size
      config.n_layers = n_layers
      config.seq_len = seq_len
      config.learning_rate = lr

      head_dims : Array(Int32)? = nil

      case head_type
      when "exponential"
        head_dims = [1, 2, 4, 8, 16, 32]
        config.d_model = head_dims.not_nil!.sum
      when "prime"
        head_dims = [2, 3, 5, 7, 11, 13, 23]
        config.d_model = head_dims.not_nil!.sum
      when "pyramid"
        head_dims = [16, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2]
        config.d_model = head_dims.not_nil!.sum
      when "ones"
        head_dims = Array.new(64, 1)
        config.d_model = head_dims.not_nil!.sum
      else
        config.d_model = d_model
        config.n_heads = d_model // 16
      end

      config.d_ff = config.d_model * 4

      model = if hd = head_dims
        MiniGPT.new(config, hd)
      else
        MiniGPT.new(config)
      end

      la_model = if lookahead > 0
        LookaheadModel.new(model, lookahead)
      else
        nil
      end

      future_model = if lookahead == -1
        FutureModel.new(model)
      else
        nil
      end

      active_params = if fm = future_model
        fm.param_count
      elsif lam = la_model
        lam.param_count
      else
        model.param_count
      end

      # Auto-derive model save path from input filename
      save_path = model_path || filename.sub(/\.[^.]+$/, "") + (agpt_mode ? ".agpt.model" : ".model")

      # Load existing checkpoint if present
      if File.exists?(save_path)
        begin
          model.load(save_path)
          puts "Loaded checkpoint: #{save_path}"
        rescue ex
          puts "Skipping checkpoint (#{ex.message})"
        end
      end

      agpt_trainer = nil
      if agpt_mode
        trie_source = "built"
        trie_index_path = agpt_load_index || agpt_save_index
        corpus_hash = AGPT::TrieCorpus.token_hash(dataset.data)
        if index_path = agpt_load_index
          begin
            build_started_at = Time.instant
            trie = AGPT::TrieCorpus.load(index_path)
            trie_build_time = Time.instant - build_started_at
            trie.validate_metadata!(
              corpus_token_count: dataset.data.size,
              vocab_size: dataset.vocab_size,
              corpus_hash: corpus_hash,
              tokenizer_tag: AGPT_TOKENIZER_TAG,
              max_depth: config.seq_len
            )
          rescue ex
            STDERR.puts "AGPT index error: #{ex.message}"
            exit 1
          end
          trie_source = "loaded"
        else
          max_starts = agpt_max_starts > 0 ? agpt_max_starts : nil
          build_started_at = Time.instant
          trie = AGPT::TrieCorpus.from_token_ids(
            dataset.data,
            max_depth: config.seq_len,
            max_starts: max_starts,
            start_offset: agpt_start_offset,
            progress_interval: agpt_progress,
            vocab_size: dataset.vocab_size,
            corpus_hash: corpus_hash,
            tokenizer_tag: AGPT_TOKENIZER_TAG
          )
          trie_build_time = Time.instant - build_started_at

          if index_path = agpt_save_index
            save_started_at = Time.instant
            trie.save(index_path)
            trie_save_time = Time.instant - save_started_at
            STDERR.puts "[agpt index] saved #{index_path} in #{trie_save_time.total_milliseconds.round(1)} ms"
          end
        end

        agpt_walk_trainer = AGPT::TrieWalkTrainer.new(trie)
        agpt_trainer = AGPT::Trainer.new(trie)  # kept for fallback/comparison
        trie_shape = trie.shape_stats
        puts "MiniGPT ready."
        puts "  Mode: AGPT trie-walk (BFS + KV cache)"
        puts "  Vocab: #{dataset.vocab_size} chars"
        puts "  Params: #{active_params}"
        puts "  Config: d_model=#{config.d_model} n_layers=#{config.n_layers} d_ff=#{config.d_ff} seq_len=#{config.seq_len} lr=#{config.learning_rate}"
        puts "  Backend: #{backend}"
        puts "  Corpus tokens: #{dataset.data.size}"
        puts "  AGPT max depth: #{trie.max_depth || "full"}"
        puts "  Trie starts used: #{trie.starts_used} / #{dataset.data.size - 1}"
        puts "  Start offset: #{agpt_start_offset}"
        puts "  Trie nodes: #{trie.node_count}"
        puts "  Prefix examples: #{agpt_walk_trainer.observed_count}"
        puts "  Trie source: #{trie_source}"
        puts "  Trie index: #{trie_index_path}" if trie_index_path
        puts "  Trie load: #{trie_build_time.total_milliseconds.round(1)} ms" if trie_source == "loaded"
        puts "  Trie build: #{trie_build_time.total_milliseconds.round(1)} ms" if trie_source == "built"
        puts "  Trie shape: root=#{trie_shape.root_children} leaves=#{trie_shape.leaves} unary=#{trie_shape.unary_nodes} branching=#{trie_shape.branching_nodes}"
        puts "  Trie breadth: peak=#{trie_shape.peak_width} at depth #{trie_shape.peak_width_depth} max_children=#{trie_shape.max_children} avg_internal=#{trie_shape.avg_children_per_internal.round(2)}"
        puts "  Model: #{save_path}"
        puts

        next if build_only
      else
        puts "MiniGPT ready."
        puts "  Vocab: #{dataset.vocab_size} chars"
        puts "  Params: #{active_params}"
        puts "  Config: d_model=#{config.d_model} n_layers=#{config.n_layers} d_ff=#{config.d_ff} seq_len=#{config.seq_len} lr=#{config.learning_rate}"
        puts "  Backend: #{backend}"
        puts "  Lookahead: #{lookahead}" if lookahead > 0
        puts "  Mode: future (causal + anti-causal)" if lookahead == -1
        puts "  Model: #{save_path}"
        puts

        next if build_only
      end

      avg_loss = 0.0
      avg_causal_loss = 0.0
      avg_future_loss = 0.0
      avg_head_losses = Array(Float64).new((lookahead > 0 ? lookahead : 0) + 1, 0.0)

      # AGPT trie-walk mode: each "step" is a full BFS epoch over the trie
      if walk_trainer = agpt_walk_trainer
        steps.times do |step|
          epoch_started = Time.instant
          loss, nodes = walk_trainer.train_epoch(model)
          elapsed = Time.instant - epoch_started
          avg_loss = step == 0 ? loss : 0.99 * avg_loss + 0.01 * loss

          GC.collect if step % 2 == 0

          puts "Epoch #{step}/#{steps} (#{nodes} nodes, #{"%.1f" % elapsed.total_seconds}s): loss = #{"%.4f" % loss} avg = #{"%.4f" % avg_loss} [agpt trie-walk]"
          if step % 5 == 0
            seed = dataset.data[0, Math.min(16, dataset.data.size)]
            generated = model.generate(seed, 100, temperature: 0.8)
            puts "  seed: #{dataset.decode(seed)}"
            puts "  gen:  #{dataset.decode(generated)}"
            puts
          end
        end
      else

      steps.times do |step|
        if fm = future_model
          input, targets = dataset.sample(config.seq_len, 0)
          loss, causal_loss, future_loss = fm.train_step(input, targets[0])
          avg_causal_loss = step == 0 ? causal_loss : 0.99 * avg_causal_loss + 0.01 * causal_loss
          avg_future_loss = step == 0 ? future_loss : 0.99 * avg_future_loss + 0.01 * future_loss
        elsif lam = la_model
          input, targets = dataset.sample(config.seq_len, lookahead)
          loss, head_losses = lam.train_step(input, targets)
          head_losses.each_with_index do |hl, i|
            avg_head_losses[i] = step == 0 ? hl : 0.99 * avg_head_losses[i] + 0.01 * hl
          end
        else
          input, targets = dataset.sample(config.seq_len, 0)
          loss = model.train_step(input, targets[0])
        end
        avg_loss = step == 0 ? loss : 0.99 * avg_loss + 0.01 * loss

        GC.collect if step % 10 == 0

        if step % 50 == 0
          head_str = if lookahead == -1
            " [causal=#{"%.4f" % avg_causal_loss}, future=#{"%.4f" % avg_future_loss}]"
          elsif lookahead > 0
            " [" + avg_head_losses.map_with_index { |l, i| "W#{i}=#{"%.4f" % l}" }.join(", ") + "]"
          else
            ""
          end
          step_label = "epoch #{dataset.epoch}"
          puts "Step #{step}/#{steps} (#{step_label}): loss = #{"%.4f" % loss} avg = #{"%.4f" % avg_loss}#{head_str}"
          seed = input[0, 16]
          gen_model = future_model || la_model
          generated = gen_model ? gen_model.generate(seed, 100, temperature: 0.8) : model.generate(seed, 100, temperature: 0.8)
          puts "  seed: #{dataset.decode(seed)}"
          puts "  gen:  #{dataset.decode(generated)}"
          puts
        end
      end

      end  # close agpt_walk_trainer else branch

      if steps > 0
        puts "Final avg loss: #{"%.4f" % avg_loss}"
        if lookahead == -1
          puts "  Causal: #{"%.4f" % avg_causal_loss}, Future: #{"%.4f" % avg_future_loss}"
        elsif lookahead > 0
          puts "  Per-head: " + avg_head_losses.map_with_index { |l, i| "W#{i}=#{"%.4f" % l}" }.join(", ")
        end
        unless no_save
          model.save(save_path)
          puts "Model saved: #{save_path}"
        end
        # Log result
        rid_str = run_id || "d#{config.d_model}n#{config.n_layers}"
        extra = "d_model=#{config.d_model} n_layers=#{config.n_layers} d_ff=#{config.d_ff}"
        extra += " mode=agpt" if agpt_mode
        log_result(rid_str, active_params, steps, avg_loss, extra, filename)
      end
      puts
      puts "Final generation:"
      seed = dataset.data[0, 16]
      gen_model = future_model || la_model
      generated = gen_model ? gen_model.generate(seed, 500, temperature: 0.8) : model.generate(seed, 500, temperature: 0.8)
      puts dataset.decode(generated)

      # --- Eval mode: complete prompts from file ---
      if ef = eval_file
        if File.exists?(ef)
          puts
          puts "=" * 60
          puts "Eval prompts from #{ef}:"
          puts "=" * 60
          File.each_line(ef) do |prompt|
            prompt = prompt.strip
            next if prompt.empty?
            seed_ids = dataset.encode(prompt)
            gen_model = future_model || la_model
            generated = gen_model ? gen_model.generate(seed_ids, 20, temperature: 0.1) : model.generate(seed_ids, 20, temperature: 0.1)
            output = dataset.decode(generated)
            # Extract just the first line of output (up to newline)
            first_line = output.split('\n').first
            puts "  #{prompt} -> #{first_line}"
          end
        else
          STDERR.puts "Eval file not found: #{ef}"
        end
      end
    end
  end

end

MicroGPT.main
