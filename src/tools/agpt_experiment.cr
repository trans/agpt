# Experiment runner — binds carved corpus, trie, checkpoint, logs, and
# result.json into one immutable per-run artifact.
#
# Workflow:
#   1. Read YAML config (new-schema; see docs/yaml-schema.md).
#   2. Verify carved files; invoke bin/agpt_carve if missing + carve block present.
#   3. (AGPT only) build/cache radix trie at data/.tries/<hash>/ when trie.path absent.
#   4. Apply defaults (model.save_file → <rundir>/checkpoint.model; trie.path → cache).
#   5. Write resolved_config.yml to the run dir.
#   6. Spawn the trainer chosen via --trainer (v1|v2|microgpt|<path>).
#   7. Convert checkpoint → HF; run agpt_lm_eval.py against held-out / external / benchmark.
#   8. Parse metrics → write result.json + update meta.json + runs.json + README table.
#
# See notes/operations/experiment-runner-design.md for the prose spec and
# notes/operations/orchestrator-rewrite-plan.md for the #29 rewrite plan.
#
# Usage:
#   bin/agpt_experiment --config FOO.yml --trainer <v1|v2|microgpt|/path/to/binary> [--seed N]
#   bin/agpt_experiment --validate FOO.yml
#
# Exit codes:
#   0 success
#   2 config validation error
#   3 trainer failed
#   4 evaluator failed
#   5 duplicate run (same config hash already present under rnd/<experiment>/)

require "yaml"
require "json"
require "option_parser"
require "digest/sha256"
require "file_utils"
require "time"

module AgptExperiment
  VERSION = "0.2.0"

  # ---------------------------------------------------------------------------
  # Config types — mirror docs/yaml-schema.md.
  # Round-tripped via YAML::Serializable to produce resolved_config.yml.
  # ---------------------------------------------------------------------------

  struct CarveBlock
    include YAML::Serializable
    include JSON::Serializable
    property source : String
    property mode : String       # "sample" | "tail"
    property ratio : Float64
    property chunks : Int32?
    property seed : Int32?
  end

  struct CorpusBlock
    include YAML::Serializable
    include JSON::Serializable
    property path : String
    property heldout : String?
    property vocab_source : String?
    property carve : CarveBlock?
  end

  struct TrieBlock
    include YAML::Serializable
    include JSON::Serializable
    property max_depth : Int32?
    property prune_min_mass : Int32 = 1
    property prune_min_depth : Int32 = 0
    property path : String?
    property virtual_tree : Bool = false
  end

  struct ModelBlock
    include YAML::Serializable
    include JSON::Serializable
    property d_model : Int32?
    property n_layers : Int32?
    property n_heads : Int32?
    property d_ff : Int32?
    property head_dim : Int32?
    property init_file : String?
    property init_seed : Int32?
    property save_file : String?
  end

  struct BudgetBlock
    include YAML::Serializable
    include JSON::Serializable
    property unit : String        # "epochs" | "steps" | "wall_seconds"
    property value : Int32
  end

  struct OptimizerBlock
    include YAML::Serializable
    include JSON::Serializable
    property name : String
    property lr : Float64?
    property beta : Float64?
    property momentum_beta : Float64?
    property weight_decay : Float64?
    property grad_clip_norm : Float64?
  end

  struct LrScheduleBlock
    include YAML::Serializable
    include JSON::Serializable
    property name : String = "constant"
    property warmup_epochs : Int32 = 0
  end

  struct GrowthBlock
    include YAML::Serializable
    include JSON::Serializable
    property divisions : Int32
    property min_epochs : Int32
    property epoch_ramp : String = "fixed"
  end

  struct TrainBlock
    include YAML::Serializable
    include JSON::Serializable
    property budget : BudgetBlock
    property seed : Int32 = 42
    property quiet : Bool = true
    property optimizer : OptimizerBlock
    property lr_schedule : LrScheduleBlock?
    # Context window — at least one required (cross-check rule if both set).
    property max_depth : Int32?
    property seq_len : Int32?
    # AGPT-only knobs.
    property partition_depth : Int32?
    property chunk_queries : Int32?
    property anc_grad : Bool?
    property mass_weight : String?
    property fire_norm : String?
    property entropy_lambda : Float64?
    property ce_only : Bool?
    property checkpoint_epochs : Array(Int32)?
    property growth : GrowthBlock?
    # Microgpt-only.
    property backend : String?
    property heads : String?
    property lookahead : Int32?
  end

  struct EvalBlock
    include YAML::Serializable
    include JSON::Serializable
    property external_file : String?
    property benchmark : String?
    property train_sanity : Bool = false
    property batch_size : Int32 = 1
    property device : String = "cpu"
    property limit : Int32?
  end

  class Config
    include YAML::Serializable
    include JSON::Serializable
    property description : String
    property experiment : String
    property run_slug : String?
    property corpus : CorpusBlock
    property trie : TrieBlock?
    property model : ModelBlock?
    property train : TrainBlock
    property eval : EvalBlock?
    # Free-form pass-through for in-development knobs. The orchestrator does
    # NOT validate field names inside `experimental`; trainers warn on
    # unknown keys. See docs/yaml-schema.md "experimental" section.
    property experimental : Hash(String, YAML::Any)?

    # Top-level shape validation. Trainer-domain validation (typo detection
    # on within-block fields) happens at the trainer itself.
    def validate! : Nil
      raise "description is required" if description.empty?
      raise "experiment is required" if experiment.empty?

      raise "corpus.path is required" if corpus.path.empty?
      if c = corpus.carve
        raise "corpus.carve.source is required" if c.source.empty?
        unless {"sample", "tail"}.includes?(c.mode)
          raise "corpus.carve.mode must be sample|tail (got #{c.mode.inspect})"
        end
        raise "corpus.carve.ratio must be in (0, 1)" unless 0.0 < c.ratio < 1.0
        if c.mode == "sample"
          raise "corpus.carve.chunks required when mode=sample" if c.chunks.nil?
          raise "corpus.carve.seed required when mode=sample" if c.seed.nil?
        end
      end

      unless {"epochs", "steps", "wall_seconds"}.includes?(train.budget.unit)
        raise "train.budget.unit must be epochs|steps|wall_seconds (got #{train.budget.unit.inspect})"
      end
      raise "train.budget.value must be positive" unless train.budget.value > 0
      if ce = train.checkpoint_epochs
        ce.each do |epoch|
          raise "train.checkpoint_epochs values must be positive (got #{epoch})" unless epoch > 0
          if train.budget.unit == "epochs" && epoch > train.budget.value
            raise "train.checkpoint_epochs value #{epoch} exceeds train.budget.value #{train.budget.value}"
          end
        end
      end

      # Context-window cross-check.
      sl = train.seq_len
      md = train.max_depth
      if sl && md && sl != md
        raise "train.seq_len (#{sl}) != train.max_depth (#{md}); AGPT cannot support unequal values today"
      end

      if ev = eval
        if ev.external_file && ev.benchmark
          raise "eval.external_file and eval.benchmark are mutually exclusive"
        end
      end
    end

    def eval_or_default : EvalBlock
      eval || EvalBlock.from_yaml("{}")
    end

    def trie_or_default : TrieBlock
      trie || TrieBlock.from_yaml("{}")
    end

    def model_or_default : ModelBlock
      model || ModelBlock.from_yaml("{}")
    end
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  def self.sha256_file(path : String) : String
    Digest::SHA256.hexdigest(&.file(path))
  end

  def self.epoch_checkpoint_path(save_path : String, epoch : Int32) : String
    suffix = ".epoch_#{epoch.to_s.rjust(6, '0')}.model"
    if save_path.ends_with?(".model")
      save_path[0, save_path.size - ".model".size] + suffix
    else
      save_path + suffix
    end
  end

  # Parse the .model header (matches src/cuda/agpt_train.cu's save_model_weights).
  def self.read_model_header(path : String) : Hash(String, Int32 | UInt32)?
    return nil unless File.exists?(path) && File.size(path) >= 28
    File.open(path, "rb") do |io|
      magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      return nil if magic != 0x4D475054_u32
      d_model  = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      n_heads  = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      n_layers = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      d_ff     = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      vocab    = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      seq_len  = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      {
        "magic" => magic, "d_model" => d_model, "n_heads" => n_heads,
        "n_layers" => n_layers, "d_ff" => d_ff, "vocab" => vocab, "seq_len" => seq_len,
      } of String => Int32 | UInt32
    end
  end

  def self.git_info : Hash(String, String | Bool)
    sha = `git rev-parse HEAD 2>/dev/null`.strip
    branch = `git rev-parse --abbrev-ref HEAD 2>/dev/null`.strip
    dirty_out = `git status --porcelain 2>/dev/null`
    {
      "git_sha"    => sha.empty? ? "unknown" : sha,
      "git_branch" => branch.empty? ? "unknown" : branch,
      "git_dirty"  => !dirty_out.empty?,
    } of String => String | Bool
  end

  # Capture host/runtime environment for reproducibility. AGPT_* env vars are
  # NOT recorded — they were removed from the schema (see "Removed / not in
  # the schema" in docs/yaml-schema.md). If any are set we emit a warning so
  # users notice. CUDA_VISIBLE_DEVICES / OMP_NUM_THREADS / CUBLAS_WORKSPACE_CONFIG
  # remain permitted and are recorded for provenance.
  def self.environment_info : Hash(String, String | Hash(String, String))
    cuda_version = if File.exists?("/opt/cuda/bin/nvcc")
      `/opt/cuda/bin/nvcc --version 2>/dev/null | grep release | head -1`.strip
    else
      `nvcc --version 2>/dev/null | grep release | head -1`.strip
    end
    nvidia_smi_query = `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1`.strip
    nvidia_smi_query = "" if nvidia_smi_query.starts_with?("Failed")
    crystal_version = `crystal --version 2>/dev/null | head -1`.strip
    python_version  = `python3 --version 2>/dev/null`.strip
    pkg = {} of String => String
    {"torch", "transformers", "lm_eval", "accelerate", "datasets"}.each do |mod|
      out = `python3 -c "import #{mod}; print(#{mod}.__version__)" 2>/dev/null`.strip
      pkg[mod] = out.empty? ? "unknown" : out
    end
    permitted_env = {} of String => String
    {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"}.each do |k|
      if v = ENV[k]?
        permitted_env[k] = v
      end
    end
    {
      "cuda"            => cuda_version.empty? ? "unknown" : cuda_version,
      "gpu"             => nvidia_smi_query.empty? ? "unknown" : nvidia_smi_query,
      "crystal"         => crystal_version.empty? ? "unknown" : crystal_version,
      "python"          => python_version.empty? ? "unknown" : python_version,
      "python_packages" => pkg,
      "env_vars"        => permitted_env,
    } of String => String | Hash(String, String)
  end

  def self.warn_about_agpt_env_vars : Nil
    found = ENV.keys.select &.starts_with?("AGPT_")
    return if found.empty?
    STDERR.puts "warning: AGPT_* env vars are no longer honored; ignoring: #{found.join(", ")}"
  end

  def self.utc_stamp : String
    Time.utc.to_s("%Y%m%dT%H%M%S")
  end

  def self.slugify(s : String) : String
    s.downcase.gsub(/[^a-z0-9]+/, "-").gsub(/^-+|-+$/, "")[0, 60]
  end

  def self.config_sha(yaml_text : String) : String
    Digest::SHA256.hexdigest(yaml_text)
  end

  # Hash for the trie cache dir: SHA256 of (corpus_sha, max_depth, prune_min_mass,
  # prune_min_depth, virtual_tree). Identical inputs → identical hash → cache reuse.
  def self.trie_cache_key(corpus_sha : String, trie : TrieBlock, max_depth : Int32) : String
    parts = "#{corpus_sha}|#{max_depth}|#{trie.prune_min_mass}|#{trie.prune_min_depth}|#{trie.virtual_tree}"
    Digest::SHA256.hexdigest(parts)[0, 16]
  end

  # ---------------------------------------------------------------------------
  # Run-dir lifecycle
  # ---------------------------------------------------------------------------

  struct RunDir
    property root : String
    property experiment : String
    property run_id : String
    property path : String

    def initialize(@root : String, @experiment : String, @run_id : String)
      @path = File.join(@root, @experiment, @run_id)
    end

    def config_yml : String        ; File.join(@path, "config.yml")            ; end
    def resolved_yml : String      ; File.join(@path, "resolved_config.yml")   ; end
    def meta_json : String         ; File.join(@path, "meta.json")             ; end
    def train_log : String         ; File.join(@path, "train.log")             ; end
    def eval_log : String          ; File.join(@path, "eval.log")              ; end
    def checkpoint : String        ; File.join(@path, "checkpoint.model")      ; end
    def epoch_hf_dir(epoch : Int32) : String
      File.join(@path, "hf_checkpoint_epoch_#{epoch.to_s.rjust(6, '0')}")
    end
    def epoch_eval_raw_json(epoch : Int32) : String
      File.join(@path, "eval_raw_epoch_#{epoch.to_s.rjust(6, '0')}.json")
    end
    def hf_dir : String            ; File.join(@path, "hf_checkpoint")         ; end
    def result_json : String       ; File.join(@path, "result.json")           ; end
    def eval_raw_json : String     ; File.join(@path, "eval_raw.json")         ; end
  end

  def self.find_duplicate(rnd_root : String, experiment : String, target_hash : String) : String?
    exp_dir = File.join(rnd_root, experiment)
    return nil unless Dir.exists?(exp_dir)
    Dir.children(exp_dir).each do |run_id|
      meta_path = File.join(exp_dir, run_id, "meta.json")
      next unless File.exists?(meta_path)
      begin
        meta = JSON.parse(File.read(meta_path))
        return run_id if meta["config_sha256"]?.try &.as_s? == target_hash
      rescue
        # malformed meta.json — skip
      end
    end
    nil
  end

  # ---------------------------------------------------------------------------
  # Subprocess wrappers
  # ---------------------------------------------------------------------------

  def self.spawn_tee(cmd : String, args : Array(String), log_path : String, append : Bool = false) : Process::Status
    mode = append ? "a" : "w"
    File.open(log_path, mode) do |log_io|
      proc = Process.new(
        cmd, args,
        input: Process::Redirect::Close,
        output: Process::Redirect::Pipe,
        error: Process::Redirect::Pipe,
      )
      done = Channel(Nil).new(2)
      spawn { IO.copy(proc.output, log_io) ; done.send(nil) }
      spawn { IO.copy(proc.error, log_io)  ; done.send(nil) }
      status = proc.wait
      done.receive ; done.receive
      log_io.flush
      status
    end
  end

  # ---------------------------------------------------------------------------
  # Carve invocation — delegates to bin/agpt_carve when corpus files missing.
  # ---------------------------------------------------------------------------

  def self.ensure_carved!(cfg : Config, log_path : String) : Nil
    need_train = !File.exists?(cfg.corpus.path)
    heldout = cfg.corpus.heldout
    need_heldout = heldout && !File.exists?(heldout)
    return unless need_train || need_heldout

    if cfg.corpus.carve.nil?
      raise "corpus.path (or corpus.heldout) missing on disk and no corpus.carve block provided. " \
            "Either pre-carve via bin/agpt_carve or add a corpus.carve block to the YAML."
    end

    STDERR.puts "agpt_experiment: invoking bin/agpt_carve --config (carved files missing)"
    cmd = "bin/agpt_carve"
    args = ["--config", @@_input_config_path.not_nil!, "--quiet"]
    status = spawn_tee(cmd, args, log_path, append: true)
    unless status.success?
      raise "bin/agpt_carve failed (exit #{status.exit_code}); see #{log_path}"
    end
    # Sanity: the files we expected now exist.
    raise "carve reported success but corpus.path still missing: #{cfg.corpus.path}" unless File.exists?(cfg.corpus.path)
    if heldout
      raise "carve reported success but corpus.heldout still missing: #{heldout}" unless File.exists?(heldout)
    end
  end

  # Stash the input config path so ensure_carved! can pass it to agpt_carve.
  # (Could be threaded through, but a module-level slot keeps the call sites tidy.)
  @@_input_config_path : String? = nil

  # ---------------------------------------------------------------------------
  # Trie build — auto-cache at data/.tries/<hash>/ for AGPT trainers.
  # ---------------------------------------------------------------------------

  def self.ensure_trie!(cfg : Config, log_path : String) : String
    trie = cfg.trie_or_default
    if p = trie.path
      raise "trie.path set to #{p} but directory does not exist" unless Dir.exists?(p)
      return p
    end
    max_depth = trie.max_depth || cfg.train.max_depth ||
      raise "trie.path not set and no train.max_depth (or trie.max_depth) given; cannot auto-build trie"
    corpus_sha = sha256_file(cfg.corpus.path)
    key = trie_cache_key(corpus_sha, trie, max_depth)
    cache_dir = File.join("data", ".tries", key)
    manifest = File.join(cache_dir, "manifest.json")

    if Dir.exists?(cache_dir) && File.exists?(manifest)
      STDERR.puts "agpt_experiment: trie cache hit at #{cache_dir}"
      return cache_dir
    end

    STDERR.puts "agpt_experiment: building trie at #{cache_dir} (depth=#{max_depth}, prune_min_mass=#{trie.prune_min_mass}, prune_min_depth=#{trie.prune_min_depth})"
    FileUtils.mkdir_p(cache_dir)
    args = [
      "--corpus", cfg.corpus.path,
      "--max-depth", max_depth.to_s,
      "--out", cache_dir,
      "--prune-min-mass", trie.prune_min_mass.to_s,
      "--prune-min-depth", trie.prune_min_depth.to_s,
    ]
    if vs = cfg.corpus.vocab_source
      args << "--vocab-file" << vs
    end
    status = spawn_tee("bin/agpt_build_radix_corpus", args, log_path, append: true)
    unless status.success?
      raise "bin/agpt_build_radix_corpus failed (exit #{status.exit_code}); see #{log_path}"
    end
    File.write(manifest, {
      "corpus_path"     => cfg.corpus.path,
      "corpus_sha256"   => corpus_sha,
      "max_depth"       => max_depth,
      "prune_min_mass"  => trie.prune_min_mass,
      "prune_min_depth" => trie.prune_min_depth,
      "virtual_tree"    => trie.virtual_tree,
    }.to_pretty_json)
    cache_dir
  end

  # ---------------------------------------------------------------------------
  # Resolved YAML — Config struct + applied defaults, round-tripped to YAML.
  # ---------------------------------------------------------------------------

  def self.apply_defaults!(cfg : Config, run : RunDir, trie_path : String?) : Nil
    model = cfg.model_or_default
    model.save_file = run.checkpoint if model.save_file.nil?
    cfg.model = model

    if trie_path
      trie = cfg.trie_or_default
      trie.path = trie_path if trie.path.nil?
      cfg.trie = trie
    end
  end

  # ---------------------------------------------------------------------------
  # Trainer selection
  # ---------------------------------------------------------------------------

  enum Trainer
    V1
    V2
    Microgpt
    Custom
  end

  def self.resolve_trainer(name : String) : {Trainer, String}
    case name
    when "v1"       then {Trainer::V1,       "bin/agpt_train"}
    when "v2"       then {Trainer::V2,       "bin/agpt_train_v2"}
    when "microgpt" then {Trainer::Microgpt, "bin/microgpt_yaml"}
    else
      if name.includes?('/') && File.exists?(name)
        {Trainer::Custom, name}
      else
        raise "unknown --trainer #{name.inspect}; expected v1|v2|microgpt or a path to a binary"
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Eval result parsing (unchanged from prior orchestrator).
  # ---------------------------------------------------------------------------

  def self.parse_eval_json(json_path : String, task_name : String) : Hash(String, Float64)?
    return nil unless File.exists?(json_path)
    parsed = JSON.parse(File.read(json_path))
    if metrics_any = parsed["metrics"]?
      if metrics_h = metrics_any.as_h?
        h = {} of String => Float64
        metrics_h.each do |k, v|
          if f = v.as_f?
            h[k] = f
          end
        end
        return h unless h.empty?
      end
    end
    task_results = parsed[task_name]? || parsed.as_h.values.first?
    return nil unless task_results
    h = {} of String => Float64
    {"word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none"}.each do |k|
      v = task_results[k]?
      next unless v
      f = v.as_f?
      h[k.split(',').first] = f if f
    end
    h
  rescue
    nil
  end

  # ---------------------------------------------------------------------------
  # Aggregation: runs.json + README table
  # ---------------------------------------------------------------------------

  def self.update_runs_json(rnd_root : String, experiment : String) : Array(JSON::Any)
    exp_dir = File.join(rnd_root, experiment)
    runs = [] of JSON::Any
    Dir.children(exp_dir).sort.each do |run_id|
      result_path = File.join(exp_dir, run_id, "result.json")
      next unless File.exists?(result_path)
      runs << JSON.parse(File.read(result_path))
    end
    File.write(File.join(exp_dir, "runs.json"), JSON.parse(runs.to_json).to_pretty_json)
    runs
  end

  def self.regenerate_readme(rnd_root : String, experiment : String) : Nil
    exp_dir = File.join(rnd_root, experiment)
    readme_path = File.join(exp_dir, "README.md")
    runs_json = File.join(exp_dir, "runs.json")
    return unless File.exists?(runs_json)
    runs = JSON.parse(File.read(runs_json)).as_a

    table = String.build do |sb|
      sb << "| Run ID | byte_perplexity | bits/byte | train (s) | total (s) |\n"
      sb << "|--------|----------------:|----------:|----------:|----------:|\n"
      runs.each do |r|
        run_id = r["run_id"]?.try(&.as_s?) || "?"
        metrics = r["metrics"]?
        bp  = metrics.try(&.["byte_perplexity"]?).try(&.as_f?)
        bpb = metrics.try(&.["bits_per_byte"]?).try(&.as_f?)
        train_wall = r["train_wall_seconds"]?.try(&.as_f?)
        wall = r["wall_seconds"]?.try(&.as_f?)
        sb << "| `" << run_id << "` | "
        sb << (bp ? bp.round(4).to_s : "—") << " | "
        sb << (bpb ? bpb.round(4).to_s : "—") << " | "
        sb << (train_wall ? train_wall.round(0).to_s : "—") << " | "
        sb << (wall ? wall.round(0).to_s : "—") << " |\n"
      end
    end

    body = File.exists?(readme_path) ? File.read(readme_path) : default_readme(experiment)
    start_marker = "<!-- agpt-experiment-table:start -->"
    end_marker = "<!-- agpt-experiment-table:end -->"
    block = "#{start_marker}\n#{table}#{end_marker}"
    if body.includes?(start_marker) && body.includes?(end_marker)
      body = body.sub(/#{Regex.escape(start_marker)}.*?#{Regex.escape(end_marker)}/m, block)
    else
      body = "#{body.rstrip}\n\n## Results\n\n#{block}\n"
    end
    File.write(readme_path, body)
  end

  def self.default_readme(experiment : String) : String
    <<-MD
    # #{experiment}

    **Status:** active

    ## Hypothesis

    (fill in)

    ## Results

    <!-- agpt-experiment-table:start -->
    (table will be auto-populated by agpt_experiment)
    <!-- agpt-experiment-table:end -->
    MD
  end

  # ---------------------------------------------------------------------------
  # Main
  # ---------------------------------------------------------------------------

  def self.main(argv : Array(String))
    config_path = nil
    trainer_name = nil
    rnd_root = "rnd"
    seed_override : Int32? = nil
    validate_only = false

    OptionParser.parse(argv) do |p|
      p.banner = "Usage: agpt_experiment --config X.yml --trainer <v1|v2|microgpt|path> [--seed N]"
      p.on("--config PATH", "YAML config path (required)") { |v| config_path = v }
      p.on("--trainer NAME", "Trainer: v1|v2|microgpt|<path>") { |v| trainer_name = v }
      p.on("--rnd-root PATH", "Root dir for experiments (default: rnd)") { |v| rnd_root = v }
      p.on("--seed N", "Override train.seed for this run") { |v| seed_override = v.to_i }
      p.on("--validate PATH", "Parse + show resolved config, take no other action") do |v|
        config_path = v
        validate_only = true
      end
      p.on("-h", "--help", "Show help") { puts p ; exit 0 }
    end

    warn_about_agpt_env_vars

    cp = config_path
    if cp.nil?
      STDERR.puts "error: --config (or --validate) is required"
      exit 2
    end
    unless File.exists?(cp)
      STDERR.puts "error: config file not found: #{cp}"
      exit 2
    end
    @@_input_config_path = File.expand_path(cp)

    input_yaml_text = File.read(cp)
    cfg = begin
      Config.from_yaml(input_yaml_text)
    rescue ex
      STDERR.puts "error: cannot parse YAML: #{ex.message}"
      exit 2
    end

    begin
      cfg.validate!
    rescue ex
      STDERR.puts "error: config invalid: #{ex.message}"
      exit 2
    end

    if validate_only
      puts cfg.to_yaml
      exit 0
    end

    tn = trainer_name
    if tn.nil?
      STDERR.puts "error: --trainer is required (one of: v1|v2|microgpt|<path>)"
      exit 2
    end
    trainer, trainer_bin = begin
      resolve_trainer(tn)
    rescue ex
      STDERR.puts "error: #{ex.message}"
      exit 2
    end

    run_slug = cfg.run_slug || "auto"
    run_id = "#{utc_stamp}-#{slugify(run_slug)}"
    run = RunDir.new(rnd_root, cfg.experiment, run_id)

    cfg_hash = config_sha(input_yaml_text)
    if existing = find_duplicate(rnd_root, cfg.experiment, cfg_hash)
      STDERR.puts "error: same config already run as #{existing}"
      exit 5
    end

    FileUtils.mkdir_p(run.path)
    File.write(run.config_yml, input_yaml_text)

    started = Time.utc
    git = git_info

    # ---- carve (delegated) ----
    ensure_carved!(cfg, run.train_log)
    corpus_sha = sha256_file(cfg.corpus.path)

    # ---- trie (AGPT trainers only) ----
    trie_path : String? = nil
    if trainer == Trainer::V1 || trainer == Trainer::V2
      trie_path = ensure_trie!(cfg, run.train_log)
    end

    # ---- apply defaults + write resolved YAML ----
    apply_defaults!(cfg, run, trie_path)
    File.write(run.resolved_yml, cfg.to_yaml)

    # ---- model header (for provenance) ----
    init_file = cfg.model_or_default.init_file
    init_sha = init_file ? sha256_file(init_file) : nil
    init_header = init_file ? (read_model_header(init_file) || ({} of String => Int32 | UInt32)) : ({} of String => Int32 | UInt32)

    meta = {
      "config_sha256" => cfg_hash,
      "started_utc"   => started.to_rfc3339,
      "trainer"       => tn,
      "trainer_bin"   => trainer_bin,
      "git_sha"       => git["git_sha"],
      "git_branch"    => git["git_branch"],
      "git_dirty"     => git["git_dirty"],
      "host"          => `hostname`.strip,
      "command"       => (["bin/agpt_experiment"] + ARGV).join(" "),
      "corpus"        => {
        "path"       => cfg.corpus.path,
        "sha256"     => corpus_sha,
        "byte_count" => File.size(cfg.corpus.path),
        "carve"      => cfg.corpus.carve,
      },
      "model_init" => {
        "path"   => init_file,
        "sha256" => init_sha,
        "header" => init_header,
      },
      "trie_path"   => trie_path,
      "environment" => environment_info,
    }
    File.write(run.meta_json, JSON.parse(meta.to_json).to_pretty_json)

    # ---- train ----
    # All trainers (v1, v2, microgpt, custom) now accept `--config <yaml> [--seed N]`.
    trainer_args = ["--config", run.resolved_yml]
    if so = seed_override
      trainer_args << "--seed" << so.to_s
    end
    STDERR.puts "agpt_experiment: trainer cmd: #{trainer_bin} #{trainer_args.join(" ")}"
    train_started = Time.utc
    train_status = spawn_tee(trainer_bin, trainer_args, run.train_log, append: true)
    train_ended = Time.utc
    train_wall = (train_ended - train_started).total_seconds
    unless train_status.success?
      STDERR.puts "trainer failed (exit #{train_status.exit_code}); tail of #{run.train_log}:"
      STDERR.puts File.read(run.train_log).split('\n').last(20).join('\n')
      exit 3
    end
    unless File.exists?(run.checkpoint)
      STDERR.puts "trainer succeeded but checkpoint not at #{run.checkpoint}"
      exit 3
    end

    # ---- convert to HF ----
    vocab_source = cfg.corpus.vocab_source || cfg.corpus.path
    convert_args = [
      "src/tools/agpt_hf.py", "convert",
      "--model", run.checkpoint,
      "--vocab-file", vocab_source,
      "--out", run.hf_dir,
    ]
    convert_started = Time.utc
    convert_status = spawn_tee("python3", convert_args, run.eval_log)
    convert_ended = Time.utc
    convert_wall = (convert_ended - convert_started).total_seconds
    unless convert_status.success?
      STDERR.puts "HF convert failed; tail of #{run.eval_log}:"
      STDERR.puts File.read(run.eval_log).split('\n').last(20).join('\n')
      exit 4
    end

    # ---- eval ----
    ev = cfg.eval_or_default
    task_base = slugify(run_slug)
    task_name : String
    eval_source_path : String? = nil
    eval_chunks_dir : String? = nil
    eval_split_kind : String
    eval_args = [
      "src/tools/agpt_lm_eval.py",
      "--hf-dir", run.hf_dir,
      "--batch-size", ev.batch_size.to_s,
      "--device", ev.device,
      "--out-json", run.eval_raw_json,
    ]
    if bench = ev.benchmark
      task_name = bench
      eval_split_kind = "benchmark"
      eval_args << "--builtin-task" << bench
    elsif ext = ev.external_file
      raise "eval.external_file does not exist: #{ext}" unless File.exists?(ext)
      eval_source_path = ext
      task_name = "#{task_base}_external"
      eval_split_kind = "external-heldout"
      eval_args << "--text-file" << ext << "--task-name" << task_name
    else
      heldout = cfg.corpus.heldout || raise "eval requires corpus.heldout (or eval.external_file / eval.benchmark)"
      raise "corpus.heldout does not exist: #{heldout}" unless File.exists?(heldout)
      # Multi-chunk if a heldout_chunks/ sits alongside the heldout file.
      heldout_dir = File.dirname(heldout)
      chunks_dir = File.join(heldout_dir, "heldout_chunks")
      task_name = "#{task_base}_holdout"
      eval_split_kind = "tail-heldout"
      if Dir.exists?(chunks_dir)
        eval_chunks_dir = chunks_dir
        eval_args << "--chunks-dir" << chunks_dir << "--task-name" << task_name
        eval_split_kind = "multi-chunk-heldout"
      else
        eval_source_path = heldout
        eval_args << "--text-file" << heldout << "--task-name" << task_name
      end
    end
    if eval_source_path || eval_chunks_dir
      eval_args << "--agpt-model" << run.checkpoint
      eval_args << "--vocab-file" << vocab_source
    end
    if v = ev.limit
      eval_args << "--limit" << v.to_s
    end
    File.open(run.eval_log, "a") { |io| io.puts "\n--- lm-eval run ---" }
    eval_started = Time.utc
    eval_status = spawn_tee("python3", eval_args, run.eval_log, append: true)
    eval_ended = Time.utc
    eval_wall = (eval_ended - eval_started).total_seconds
    unless eval_status.success?
      STDERR.puts "lm-eval failed; tail of #{run.eval_log}:"
      STDERR.puts File.read(run.eval_log).split('\n').last(20).join('\n')
      exit 4
    end

    checkpoint_results = [] of JSON::Any
    if checkpoint_epochs = cfg.train.checkpoint_epochs
      checkpoint_epochs.uniq.sort.each do |epoch|
        checkpoint_path = epoch_checkpoint_path(run.checkpoint, epoch)
        unless File.exists?(checkpoint_path)
          STDERR.puts "trainer did not write requested checkpoint epoch #{epoch}: #{checkpoint_path}"
          exit 3
        end

        hf_dir = run.epoch_hf_dir(epoch)
        raw_json = run.epoch_eval_raw_json(epoch)
        checkpoint_convert_args = [
          "src/tools/agpt_hf.py", "convert",
          "--model", checkpoint_path,
          "--vocab-file", vocab_source,
          "--out", hf_dir,
        ]
        checkpoint_convert_started = Time.utc
        File.open(run.eval_log, "a") { |io| io.puts "\n--- convert checkpoint epoch #{epoch} ---" }
        checkpoint_convert_status = spawn_tee("python3", checkpoint_convert_args, run.eval_log, append: true)
        checkpoint_convert_ended = Time.utc
        checkpoint_convert_wall = (checkpoint_convert_ended - checkpoint_convert_started).total_seconds
        unless checkpoint_convert_status.success?
          STDERR.puts "HF convert failed for checkpoint epoch #{epoch}; tail of #{run.eval_log}:"
          STDERR.puts File.read(run.eval_log).split('\n').last(20).join('\n')
          exit 4
        end

        checkpoint_task_name = task_name
        checkpoint_eval_args = [
          "src/tools/agpt_lm_eval.py",
          "--hf-dir", hf_dir,
          "--batch-size", ev.batch_size.to_s,
          "--device", ev.device,
          "--out-json", raw_json,
        ]
        if bench = ev.benchmark
          checkpoint_eval_args << "--builtin-task" << bench
        elsif ext = ev.external_file
          checkpoint_task_name = "#{task_base}_epoch_#{epoch.to_s.rjust(6, '0')}_external"
          checkpoint_eval_args << "--text-file" << ext << "--task-name" << checkpoint_task_name
        elsif eval_chunks_dir
          checkpoint_task_name = "#{task_base}_epoch_#{epoch.to_s.rjust(6, '0')}_holdout"
          checkpoint_eval_args << "--chunks-dir" << eval_chunks_dir.not_nil! << "--task-name" << checkpoint_task_name
        elsif eval_source_path
          checkpoint_task_name = "#{task_base}_epoch_#{epoch.to_s.rjust(6, '0')}_holdout"
          checkpoint_eval_args << "--text-file" << eval_source_path.not_nil! << "--task-name" << checkpoint_task_name
        end
        if eval_source_path || eval_chunks_dir
          checkpoint_eval_args << "--agpt-model" << checkpoint_path
          checkpoint_eval_args << "--vocab-file" << vocab_source
        end
        if v = ev.limit
          checkpoint_eval_args << "--limit" << v.to_s
        end

        File.open(run.eval_log, "a") { |io| io.puts "\n--- lm-eval checkpoint epoch #{epoch} ---" }
        checkpoint_eval_started = Time.utc
        checkpoint_eval_status = spawn_tee("python3", checkpoint_eval_args, run.eval_log, append: true)
        checkpoint_eval_ended = Time.utc
        checkpoint_eval_wall = (checkpoint_eval_ended - checkpoint_eval_started).total_seconds
        unless checkpoint_eval_status.success?
          STDERR.puts "lm-eval failed for checkpoint epoch #{epoch}; tail of #{run.eval_log}:"
          STDERR.puts File.read(run.eval_log).split('\n').last(20).join('\n')
          exit 4
        end

        checkpoint_metrics = parse_eval_json(raw_json, checkpoint_task_name) || {} of String => Float64
        checkpoint_result = {
          "epoch"                => epoch,
          "checkpoint_path"      => checkpoint_path,
          "checkpoint_sha"       => sha256_file(checkpoint_path),
          "hf_dir"               => hf_dir,
          "eval_raw_json"        => raw_json,
          "task_name"            => checkpoint_task_name,
          "convert_wall_seconds" => checkpoint_convert_wall,
          "eval_wall_seconds"    => checkpoint_eval_wall,
          "metrics"              => checkpoint_metrics,
        }
        checkpoint_results << JSON.parse(checkpoint_result.to_json)
      end
    end

    # ---- write result.json ----
    metrics = parse_eval_json(run.eval_raw_json, task_name) || {} of String => Float64
    ended = Time.utc
    wall = (ended - started).total_seconds

    eval_record = {
      "split"           => eval_split_kind,
      "source_path"     => eval_source_path,
      "source_sha256"   => eval_source_path ? sha256_file(eval_source_path.not_nil!) : nil,
      "chunks_dir"      => eval_chunks_dir,
      "benchmark"       => ev.benchmark,
      "task_name"       => task_name,
    }

    # Record any experimental keys present in the input so canonical-vs-experimental
    # runs are filterable. Namespaced by --trainer choice; we record what was sent,
    # not what the trainer ultimately honored (the trainer's WARN lines distinguish).
    experimental_used : Hash(String, Array(String)) | Nil = nil
    if exp = cfg.experimental
      if !exp.empty?
        experimental_used = {tn => exp.keys.sort}
      end
    end

    result = {
      "run_id"               => run_id,
      "experiment"           => cfg.experiment,
      "wall_seconds"         => wall,
      "train_wall_seconds"   => train_wall,
      "convert_wall_seconds" => convert_wall,
      "eval_wall_seconds"    => eval_wall,
      "started_utc"          => started.to_rfc3339,
      "ended_utc"            => ended.to_rfc3339,
      "trainer"              => tn,
      "evaluator"            => "lm-evaluation-harness via agpt_lm_eval.py",
      "eval"                 => eval_record,
      "metrics"              => metrics,
      "checkpoint_sha"       => sha256_file(run.checkpoint),
      "checkpoint_results"   => checkpoint_results.empty? ? nil : checkpoint_results,
      "experimental_used"    => experimental_used,
    }
    File.write(run.result_json, JSON.parse(result.to_json).to_pretty_json)

    # Update meta with end-time fields.
    updated_meta = JSON.parse(File.read(run.meta_json)).as_h
    updated_meta["ended_utc"] = JSON::Any.new(ended.to_rfc3339)
    updated_meta["wall_seconds"] = JSON::Any.new(wall)
    updated_meta["train_wall_seconds"] = JSON::Any.new(train_wall)
    updated_meta["convert_wall_seconds"] = JSON::Any.new(convert_wall)
    updated_meta["eval_wall_seconds"] = JSON::Any.new(eval_wall)
    File.write(run.meta_json, JSON.parse(updated_meta.to_json).to_pretty_json)

    update_runs_json(rnd_root, cfg.experiment)
    regenerate_readme(rnd_root, cfg.experiment)

    STDERR.puts "agpt_experiment: done."
    STDERR.puts "  run dir : #{run.path}"
    metrics.each do |k, v|
      STDERR.puts "  #{k} = #{v.round(4)}"
    end
    STDERR.puts "  train   : #{train_wall.round(1)}s"
    STDERR.puts "  wall    : #{wall.round(1)}s"
  end
end

AgptExperiment.main(ARGV)
