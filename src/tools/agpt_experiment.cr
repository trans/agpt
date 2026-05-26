# Experiment runner — binds config, checkpoint, logs, and result.json into
# one immutable artifact per run.
#
# Workflow: read YAML config → create rnd/<experiment>/<run-id>/ →
# spawn trainer → convert checkpoint to HF format → spawn lm-eval driver →
# parse PPL → write result.json + meta.json → update runs.json + README.md.
#
# See notes/operations/experiment-runner-design.md for the full spec.
#
# Usage:
#   bin/agpt_experiment --config configs/foo.yml [--rnd-root rnd]
#   bin/agpt_experiment --validate configs/foo.yml   # parse + show resolved, do nothing
#
# Exit codes:
#   0 success
#   2 config validation error
#   3 trainer failed
#   4 evaluator failed
#   5 duplicate run (same config hash + corpus sha already exists)

require "yaml"
require "json"
require "option_parser"
require "digest/sha256"
require "file_utils"
require "time"

module AgptExperiment
  VERSION = "0.1.0"

  # ---------------------------------------------------------------------------
  # Config types
  # ---------------------------------------------------------------------------

  struct MetaBlock
    include YAML::Serializable
    include JSON::Serializable
    property description : String
    property experiment : String
    property hypothesis_ref : String?
    property run_slug : String?
  end

  struct CorpusBlock
    include YAML::Serializable
    include JSON::Serializable
    property path : String
    # vocab_source defaults to `path` when nil.
    property vocab_source : String?
    # Fraction of the corpus used for training (0 < frac <= 1). When < 1.0
    # the orchestrator carves the corpus into train_corpus.txt (head) and
    # heldout_corpus.txt (tail) inside the run dir; the trainer only sees
    # the head, and eval.split=tail-heldout uses the tail.
    property train_frac : Float64 = 1.0
  end

  struct ModelBlock
    include YAML::Serializable
    include JSON::Serializable
    property init_from : String
  end

  # train: block — fields map to agpt_train_v2 flags. `flags:` is a free
  # map for anything not covered by named fields.
  struct TrainBlock
    include YAML::Serializable
    include JSON::Serializable
    property tool : String = "bin/agpt_train_v2"
    property mode : String = "train-growth"
    property growth_frontiers : String?
    property growth_divisions : Int32?
    property growth_max_depth : Int32?
    property growth_min_epochs : Int32?
    property growth_epoch_ramp : String?
    property epochs : Int32?
    property optimizer : String?
    property lr : Float64?
    property rmsprop_beta : Float64?
    property momentum_beta : Float64?
    property lr_schedule : String?
    property warmup_epochs : Int32?
    property partition_depth : Int32?
    property chunk_queries : Int32?
    property anc_grad : Bool = false
    property accumulate : Bool? # nil = don't pass either flag
    property quiet : Bool = true
    property extra_args : Array(String) = [] of String
  end

  # eval: block — runs agpt_lm_eval.py against an HF-converted checkpoint.
  # `split` MUST be set explicitly; there is no safe default. Numbers from
  # different splits are not comparable, so result.json records the split.
  struct EvalBlock
    include YAML::Serializable
    include JSON::Serializable
    property tool : String = "src/tools/agpt_lm_eval.py"
    # split: train | tail-heldout | external-heldout | benchmark
    property split : String
    # Required when split=external-heldout — path to a text file that the
    # model never saw during training.
    property external_file : String?
    # Required when split=benchmark — built-in lm-eval task name.
    property benchmark : String?
    property task_name : String = "experiment_ppl"
    property batch_size : Int32 = 1
    property device : String = "cpu"
    property limit : Int32?

    def validate!(corpus : CorpusBlock) : Nil
      case split
      when "train"
        # ok — evaluator gets the train slice
      when "tail-heldout"
        if corpus.train_frac >= 1.0
          raise "eval.split=tail-heldout requires corpus.train_frac < 1.0 " \
                "(else there is no held-out tail)"
        end
      when "external-heldout"
        ext = external_file
        raise "eval.split=external-heldout requires eval.external_file" if ext.nil?
        raise "eval.external_file does not exist: #{ext}" unless File.exists?(ext.not_nil!)
      when "benchmark"
        raise "eval.split=benchmark requires eval.benchmark" if benchmark.nil?
      else
        raise "eval.split must be one of: train | tail-heldout | external-heldout | benchmark"
      end
    end
  end

  struct Config
    include YAML::Serializable
    include JSON::Serializable
    property meta : MetaBlock
    property corpus : CorpusBlock
    property model : ModelBlock
    property train : TrainBlock
    property eval : EvalBlock

    def validate! : Nil
      raise "meta.experiment is required" if meta.experiment.empty?
      raise "corpus.path does not exist: #{corpus.path}" unless File.exists?(corpus.path)
      raise "model.init_from does not exist: #{model.init_from}" unless File.exists?(model.init_from)
      raise "corpus.train_frac must be in (0, 1]" unless 0.0 < corpus.train_frac <= 1.0
      eval.validate!(corpus)
    end
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  def self.sha256_file(path : String) : String
    Digest::SHA256.hexdigest(&.file(path))
  end

  # Parse the .model header (matches src/cuda/agpt_train.cu's save_model_weights).
  # Returns nil if file too short or magic mismatch.
  def self.read_model_header(path : String) : Hash(String, Int32 | UInt32)?
    return nil unless File.exists?(path) && File.size(path) >= 28
    File.open(path, "rb") do |io|
      magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      return nil if magic != 0x4D475054_u32
      d_model = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      n_heads = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      n_layers = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      d_ff = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      vocab = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      seq_len = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      {
        "magic"     => magic,
        "d_model"   => d_model,
        "n_heads"   => n_heads,
        "n_layers"  => n_layers,
        "d_ff"      => d_ff,
        "vocab"     => vocab,
        "seq_len"   => seq_len,
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

  # Capture the host/runtime environment so a result can be reproduced or
  # debugged across machines. None of these are required to exist; missing
  # fields are recorded as "unknown" so the meta.json shape is stable.
  def self.environment_info : Hash(String, String | Hash(String, String))
    # nvcc may be at /opt/cuda/bin/nvcc on Arch; fall back to PATH.
    cuda_version = if File.exists?("/opt/cuda/bin/nvcc")
      `/opt/cuda/bin/nvcc --version 2>/dev/null | grep release | head -1`.strip
    else
      `nvcc --version 2>/dev/null | grep release | head -1`.strip
    end
    nvidia_smi_query = `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1`.strip
    # nvidia-smi sometimes prints "Failed to initialize NVML: ..." on stdout
    # even with exit 0. Treat any "Failed" prefix as unknown.
    nvidia_smi_query = "" if nvidia_smi_query.starts_with?("Failed")
    crystal_version = `crystal --version 2>/dev/null | head -1`.strip
    python_version = `python3 --version 2>/dev/null`.strip
    pkg = {} of String => String
    {"torch", "transformers", "lm_eval", "accelerate", "datasets"}.each do |mod|
      out = `python3 -c "import #{mod}; print(#{mod}.__version__)" 2>/dev/null`.strip
      pkg[mod] = out.empty? ? "unknown" : out
    end
    agpt_env = {} of String => String
    ENV.each do |k, v|
      agpt_env[k] = v if k.starts_with?("AGPT_") || k.starts_with?("CUDA_") || k == "CUBLAS_WORKSPACE_CONFIG"
    end
    cuda = cuda_version.empty? ? "unknown" : cuda_version
    gpu = nvidia_smi_query.empty? ? "unknown" : nvidia_smi_query
    crystal = crystal_version.empty? ? "unknown" : crystal_version
    py = python_version.empty? ? "unknown" : python_version
    {
      "cuda"             => cuda,
      "gpu"              => gpu,
      "crystal"          => crystal,
      "python"           => py,
      "python_packages"  => pkg,
      "env_vars"         => agpt_env,
    } of String => String | Hash(String, String)
  end

  def self.utc_stamp : String
    Time.utc.to_s("%Y%m%dT%H%M%S")
  end

  def self.slugify(s : String) : String
    s.downcase
      .gsub(/[^a-z0-9]+/, "-")
      .gsub(/^-+|-+$/, "")
      .[0, 60]
  end

  def self.config_sha(cfg : Config) : String
    # Serialize back to YAML so the hash is invariant to comment/whitespace.
    Digest::SHA256.hexdigest(cfg.to_yaml)
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

    def config_yml : String      ; File.join(@path, "config.yml") ; end
    def resolved_json : String   ; File.join(@path, "resolved_config.json") ; end
    def meta_json : String       ; File.join(@path, "meta.json") ; end
    def train_log : String       ; File.join(@path, "train.log") ; end
    def eval_log : String        ; File.join(@path, "eval.log") ; end
    def checkpoint : String      ; File.join(@path, "checkpoint.model") ; end
    def hf_dir : String          ; File.join(@path, "hf_checkpoint") ; end
    def result_json : String     ; File.join(@path, "result.json") ; end
    def eval_raw_json : String   ; File.join(@path, "eval_raw.json") ; end
    def train_corpus : String    ; File.join(@path, "train_corpus.txt") ; end
    def heldout_corpus : String  ; File.join(@path, "heldout_corpus.txt") ; end
  end

  # Carve corpus.path into train_corpus.txt and heldout_corpus.txt under
  # the run dir. Returns the byte offset where the split happens.
  # When train_frac == 1.0, returns nil and writes nothing.
  def self.carve_corpus(corpus : CorpusBlock, run : RunDir) : Int64?
    return nil if corpus.train_frac >= 1.0
    bytes = File.read(corpus.path).to_slice
    total = bytes.size.to_i64
    split = (total.to_f64 * corpus.train_frac).to_i64
    File.write(run.train_corpus, String.new(bytes[0, split]))
    File.write(run.heldout_corpus, String.new(bytes[split, (total - split).to_i32]))
    split
  end

  def self.find_duplicate(rnd_root : String, experiment : String, target_hash : String) : String?
    exp_dir = File.join(rnd_root, experiment)
    return nil unless Dir.exists?(exp_dir)
    Dir.children(exp_dir).each do |run_id|
      meta_path = File.join(exp_dir, run_id, "meta.json")
      next unless File.exists?(meta_path)
      begin
        meta = JSON.parse(File.read(meta_path))
        existing_hash = meta["config_sha256"]?.try &.as_s?
        return run_id if existing_hash == target_hash
      rescue
        # malformed meta.json — skip
      end
    end
    nil
  end

  # ---------------------------------------------------------------------------
  # Trainer + evaluator subprocess wrappers
  # ---------------------------------------------------------------------------

  def self.build_trainer_args(cfg : Config, run : RunDir, train_corpus_path : String) : Array(String)
    args = [] of String
    args << "--mode" << cfg.train.mode
    args << "--model" << cfg.model.init_from
    args << "--corpus" << train_corpus_path
    args << "--save" << run.checkpoint

    if v = cfg.train.growth_frontiers      ; args << "--growth-frontiers" << v ; end
    if v = cfg.train.growth_divisions      ; args << "--growth-divisions" << v.to_s ; end
    if v = cfg.train.growth_max_depth      ; args << "--growth-max-depth" << v.to_s ; end
    if v = cfg.train.growth_min_epochs     ; args << "--growth-min-epochs" << v.to_s ; end
    if v = cfg.train.growth_epoch_ramp     ; args << "--growth-epoch-ramp" << v ; end
    if v = cfg.train.epochs                ; args << "--epochs" << v.to_s ; end
    if v = cfg.train.optimizer             ; args << "--optimizer" << v ; end
    if v = cfg.train.lr                    ; args << "--lr" << v.to_s ; end
    if v = cfg.train.rmsprop_beta          ; args << "--rmsprop-beta" << v.to_s ; end
    if v = cfg.train.momentum_beta         ; args << "--momentum-beta" << v.to_s ; end
    if v = cfg.train.lr_schedule           ; args << "--lr-schedule" << v ; end
    if v = cfg.train.warmup_epochs         ; args << "--warmup-epochs" << v.to_s ; end
    if v = cfg.train.partition_depth       ; args << "--partition-depth" << v.to_s ; end
    if v = cfg.train.chunk_queries         ; args << "--chunk-queries" << v.to_s ; end
    args << "--anc-grad" if cfg.train.anc_grad
    case cfg.train.accumulate
    when true  then args << "--accumulate"
    when false then args << "--no-accumulate"
    end
    args << "--quiet" if cfg.train.quiet
    args.concat(cfg.train.extra_args)
    args
  end

  def self.spawn_tee(cmd : String, args : Array(String), log_path : String) : Process::Status
    File.open(log_path, "w") do |log_io|
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

  # Parse the JSON produced by agpt_lm_eval.py --out-json. Pick the metrics
  # we care about. Returns nil on failure.
  def self.parse_eval_json(json_path : String, task_name : String) : Hash(String, Float64)?
    return nil unless File.exists?(json_path)
    parsed = JSON.parse(File.read(json_path))
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
      sb << "| Run ID | byte_ppl | bits/byte | word_ppl | wall (s) |\n"
      sb << "|--------|---------:|----------:|---------:|---------:|\n"
      runs.each do |r|
        run_id = r["run_id"]?.try(&.as_s?) || "?"
        bp = r["metrics"]?.try(&.["byte_perplexity"]?).try(&.as_f?)
        bpb = r["metrics"]?.try(&.["bits_per_byte"]?).try(&.as_f?)
        wp = r["metrics"]?.try(&.["word_perplexity"]?).try(&.as_f?)
        wall = r["wall_seconds"]?.try(&.as_f?)
        sb << "| `" << run_id << "` | "
        sb << (bp ? bp.round(4).to_s : "—") << " | "
        sb << (bpb ? bpb.round(4).to_s : "—") << " | "
        sb << (wp ? wp.round(2).to_s : "—") << " | "
        sb << (wall ? wall.round(0).to_s : "—") << " |\n"
      end
    end

    # Preserve existing README prose; regenerate only the table inside markers.
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

    ## Scope

    (fill in)

    ## Results

    <!-- agpt-experiment-table:start -->
    (table will be auto-populated by agpt_experiment)
    <!-- agpt-experiment-table:end -->

    ## Conclusion

    (fill in once enough runs have landed)
    MD
  end

  # ---------------------------------------------------------------------------
  # Main
  # ---------------------------------------------------------------------------

  def self.main(argv : Array(String))
    config_path = nil
    rnd_root = "rnd"
    run_slug_override = nil
    validate_only = false

    OptionParser.parse(argv) do |p|
      p.banner = "Usage: agpt_experiment --config X.yml [--rnd-root rnd]"
      p.on("--config PATH", "YAML config path (required)") { |v| config_path = v }
      p.on("--rnd-root PATH", "Root dir for experiments (default: rnd)") { |v| rnd_root = v }
      p.on("--run-slug SLUG", "Override run slug component") { |v| run_slug_override = v }
      p.on("--validate", "Parse + show resolved config, take no other action") { validate_only = true }
      p.on("-h", "--help", "Show help") { puts p ; exit 0 }
    end

    cp = config_path
    if cp.nil?
      STDERR.puts "error: --config is required"
      exit 2
    end
    unless File.exists?(cp)
      STDERR.puts "error: config file not found: #{cp}"
      exit 2
    end

    cfg = begin
      Config.from_yaml(File.read(cp))
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

    run_slug = run_slug_override || cfg.meta.run_slug || "auto"
    run_id = "#{utc_stamp}-#{slugify(run_slug)}"
    run = RunDir.new(rnd_root, cfg.meta.experiment, run_id)

    cfg_hash = config_sha(cfg)
    if existing = find_duplicate(rnd_root, cfg.meta.experiment, cfg_hash)
      STDERR.puts "error: same config already run as #{existing}"
      exit 5
    end

    FileUtils.mkdir_p(run.path)
    File.write(run.config_yml, File.read(cp))
    File.write(run.resolved_json, cfg.to_json)

    started = Time.utc
    corpus_sha = sha256_file(cfg.corpus.path)
    init_sha = sha256_file(cfg.model.init_from)
    init_header = read_model_header(cfg.model.init_from) || ({} of String => Int32 | UInt32)
    git = git_info

    meta = {
      "config_sha256"  => cfg_hash,
      "started_utc"    => started.to_rfc3339,
      "git_sha"        => git["git_sha"],
      "git_branch"     => git["git_branch"],
      "git_dirty"      => git["git_dirty"],
      "host"           => `hostname`.strip,
      "command"        => (["bin/agpt_experiment"] + ARGV).join(" "),
      "corpus"         => {
        "path"       => cfg.corpus.path,
        "sha256"     => corpus_sha,
        "byte_count" => File.size(cfg.corpus.path),
      },
      "model_init" => {
        "path"   => cfg.model.init_from,
        "sha256" => init_sha,
        "header" => init_header,
      },
      "environment" => environment_info,
    }
    File.write(run.meta_json, meta.to_json)

    # ---- carve corpus (no-op when train_frac == 1.0) ----
    split_offset = carve_corpus(cfg.corpus, run)
    train_corpus_path = split_offset ? run.train_corpus : cfg.corpus.path

    # ---- train ----
    trainer_args = build_trainer_args(cfg, run, train_corpus_path)
    full_cmd = "#{cfg.train.tool} #{trainer_args.join(" ")}"
    STDERR.puts "agpt_experiment: trainer cmd: #{full_cmd}"
    train_status = spawn_tee(cfg.train.tool, trainer_args, run.train_log)
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
    convert_status = spawn_tee("python3", convert_args, run.eval_log)
    unless convert_status.success?
      STDERR.puts "HF convert failed; tail of #{run.eval_log}:"
      STDERR.puts File.read(run.eval_log).split('\n').last(20).join('\n')
      exit 4
    end

    # ---- eval ----
    # Resolve eval text source from the split.
    eval_source_path : String? = nil
    eval_args = [
      cfg.eval.tool,
      "--hf-dir", run.hf_dir,
      "--task-name", cfg.eval.task_name,
      "--batch-size", cfg.eval.batch_size.to_s,
      "--device", cfg.eval.device,
      "--out-json", run.eval_raw_json,
    ]
    case cfg.eval.split
    when "train"
      eval_source_path = train_corpus_path
      eval_args << "--text-file" << eval_source_path
    when "tail-heldout"
      eval_source_path = run.heldout_corpus
      eval_args << "--text-file" << eval_source_path
    when "external-heldout"
      eval_source_path = cfg.eval.external_file.not_nil!
      eval_args << "--text-file" << eval_source_path
    when "benchmark"
      eval_args << "--builtin-task" << cfg.eval.benchmark.not_nil!
    end
    if v = cfg.eval.limit
      eval_args << "--limit" << v.to_s
    end
    # Append to the existing eval.log (which has the HF-convert output).
    File.open(run.eval_log, "a") do |io|
      io.puts "\n--- lm-eval run ---"
    end
    eval_status = spawn_tee_append("python3", eval_args, run.eval_log)
    unless eval_status.success?
      STDERR.puts "lm-eval failed; tail of #{run.eval_log}:"
      STDERR.puts File.read(run.eval_log).split('\n').last(20).join('\n')
      exit 4
    end

    # ---- write result.json ----
    metrics = parse_eval_json(run.eval_raw_json, cfg.eval.task_name) || {} of String => Float64
    ended = Time.utc
    wall = (ended - started).total_seconds

    eval_record = {
      "split"               => cfg.eval.split,
      "source_path"         => eval_source_path,
      "source_sha256"       => eval_source_path ? sha256_file(eval_source_path.not_nil!) : nil,
      "benchmark"           => cfg.eval.benchmark,
      "corpus_train_frac"   => cfg.corpus.train_frac,
      "corpus_split_offset" => split_offset,
    }

    result = {
      "run_id"          => run_id,
      "experiment"      => cfg.meta.experiment,
      "wall_seconds"    => wall,
      "started_utc"     => started.to_rfc3339,
      "ended_utc"       => ended.to_rfc3339,
      "evaluator"       => "lm-evaluation-harness via agpt_lm_eval.py",
      "eval"            => eval_record,
      "metrics"         => metrics,
      "checkpoint_sha"  => sha256_file(run.checkpoint),
    }
    File.write(run.result_json, JSON.parse(result.to_json).to_pretty_json)

    # Update meta with end-time fields too.
    updated_meta = JSON.parse(File.read(run.meta_json)).as_h
    updated_meta["ended_utc"] = JSON::Any.new(ended.to_rfc3339)
    updated_meta["wall_seconds"] = JSON::Any.new(wall)
    File.write(run.meta_json, JSON.parse(updated_meta.to_json).to_pretty_json)

    # ---- aggregate ----
    update_runs_json(rnd_root, cfg.meta.experiment)
    regenerate_readme(rnd_root, cfg.meta.experiment)

    STDERR.puts "agpt_experiment: done."
    STDERR.puts "  run dir : #{run.path}"
    metrics.each do |k, v|
      STDERR.puts "  #{k} = #{v.round(4)}"
    end
    STDERR.puts "  wall    : #{wall.round(1)}s"
  end

  # Variant of spawn_tee that appends to the log instead of truncating.
  def self.spawn_tee_append(cmd : String, args : Array(String), log_path : String) : Process::Status
    File.open(log_path, "a") do |log_io|
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
end

AgptExperiment.main(ARGV)
