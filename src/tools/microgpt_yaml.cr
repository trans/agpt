# bin/microgpt_yaml
#
# YAML-config adapter for bin/microgpt.
#
# Reads a YAML config conforming to docs/yaml-schema.md, validates the
# subset of fields microgpt consumes (strict-reject anything microgpt
# can't honor), translates to microgpt's existing CLI flags, and exec's
# bin/microgpt with the resulting argv.
#
# Zero changes to microgpt itself. Microgpt's legacy CLI still works
# unchanged via the bare `bin/microgpt` binary; this adapter is what
# you use when you want to drive microgpt from the new canonical YAML
# schema (directly or via bin/agpt_experiment --trainer microgpt).
#
# Usage:
#   bin/microgpt_yaml --config <yaml-path> [--seed <int>]

require "option_parser"
require "yaml"

config_path  = ""
seed_override : Int32? = nil
microgpt_bin = "bin/microgpt"

OptionParser.parse do |p|
  p.banner = "Usage: microgpt_yaml --config <yaml-path> [--seed <int>]"
  p.on("--config PATH", "YAML config file") { |v| config_path = v }
  p.on("--seed N", "Override train.seed") { |v| seed_override = v.to_i }
  p.on("--microgpt-bin PATH", "Path to microgpt binary (default: bin/microgpt)") { |v| microgpt_bin = v }
  p.on("-h", "--help", "Show this help") { puts p; exit 0 }
end

if config_path.empty?
  STDERR.puts "Error: --config is required"
  exit 1
end
unless File.exists?(config_path)
  STDERR.puts "Error: config file not found: #{config_path}"
  exit 1
end
unless File.exists?(microgpt_bin)
  STDERR.puts "Error: microgpt binary not found at #{microgpt_bin}"
  exit 1
end

# --- Field consumption registry per docs/yaml-schema.md ---
#
# Microgpt's consumed fields (what the adapter knows how to translate
# into microgpt CLI flags). Anything within the trainer-domain sections
# (corpus, model, train) that's NOT in this set is a strict-reject
# error. Metadata sections (description/experiment/run_slug/eval) and
# orchestrator-only sections (corpus.heldout, corpus.carve) are ignored.
MG_CORPUS_FIELDS = Set{"path", "vocab_source", "heldout", "carve"}
MG_MODEL_FIELDS  = Set{"d_model", "n_layers", "n_heads", "init_file", "init_seed", "save_file"}
MG_TRAIN_FIELDS  = Set{"budget", "seed", "quiet", "optimizer", "lr_schedule", "seq_len", "backend", "heads", "lookahead"}
MG_TRAIN_OPTIMIZER_FIELDS = Set{"name", "lr"}
MG_TRAIN_LR_SCHEDULE_FIELDS = Set{"name", "warmup_epochs"}
MG_TRAIN_BUDGET_FIELDS = Set{"unit", "value"}

# Top-level sections microgpt explicitly ignores (no error).
MG_IGNORED_SECTIONS = Set{"description", "experiment", "run_slug", "eval", "trie"}

# Within the trainer-domain sections, fields microgpt ignores (no error).
# Examples: corpus.heldout/carve/vocab_source — eval/orchestrator concerns.
MG_IGNORED_CORPUS_FIELDS = Set{"heldout", "carve", "vocab_source"}

cfg = YAML.parse(File.read(config_path))

unless cfg.as_h?
  STDERR.puts "Error: config root must be a mapping (#{config_path})"
  exit 1
end

# --- Top-level section validation ---
allowed_top = Set{"corpus", "model", "train"} | MG_IGNORED_SECTIONS
cfg.as_h.each_key do |k|
  ks = k.as_s
  unless allowed_top.includes?(ks)
    STDERR.puts "Error: unknown top-level section '#{ks}' in #{config_path}"
    exit 1
  end
end

# --- corpus ---
corpus = cfg["corpus"]?
if corpus.nil? || !corpus.as_h?
  STDERR.puts "Error: corpus block is required"
  exit 1
end

corpus_path : String? = nil
corpus.as_h.each do |k, v|
  ks = k.as_s
  if MG_IGNORED_CORPUS_FIELDS.includes?(ks)
    next  # ignored — not microgpt's concern
  end
  unless MG_CORPUS_FIELDS.includes?(ks)
    STDERR.puts "Error: unknown corpus field '#{ks}'"
    exit 1
  end
  case ks
  when "path"          then corpus_path = v.as_s
  when "vocab_source"  then next  # ignored; microgpt derives from corpus.path
  end
end
if corpus_path.nil? || corpus_path.try(&.empty?)
  STDERR.puts "Error: corpus.path is required"
  exit 1
end
unless File.exists?(corpus_path.not_nil!)
  STDERR.puts "Error: corpus.path file not found: #{corpus_path}"
  exit 1
end

# --- model ---
model = cfg["model"]?
d_model     = 64
n_layers    = 2
n_heads     = 4
init_file : String? = nil
save_file : String? = nil
init_seed   = 0

if model && model.as_h?
  model.as_h.each do |k, v|
    ks = k.as_s
    unless MG_MODEL_FIELDS.includes?(ks) || ks == "d_ff" || ks == "head_dim"
      STDERR.puts "Error: unknown model field '#{ks}'"
      exit 1
    end
    case ks
    when "d_model"   then d_model   = v.as_i
    when "n_layers"  then n_layers  = v.as_i
    when "n_heads"   then n_heads   = v.as_i
    when "init_file" then init_file = v.as_s
    when "init_seed" then init_seed = v.as_i
    when "save_file" then save_file = v.as_s
    when "d_ff", "head_dim"
      # microgpt has its own defaults; accept silently if specified
      next
    end
  end
end

# --- train ---
train = cfg["train"]?
if train.nil? || !train.as_h?
  STDERR.puts "Error: train block is required"
  exit 1
end

seed_val     = 42
quiet        = false
seq_len      = 128
lookahead    = 0
backend      = "crystal"
heads        = "uniform"
budget_unit  = ""
budget_value : Float64 = 0.0
lr            : Float64 = 0.0003
lr_schedule   = "constant"
warmup_epochs = 0
opt_name      = "adam"

train.as_h.each do |k, v|
  ks = k.as_s
  unless MG_TRAIN_FIELDS.includes?(ks) || ks == "max_depth" || ks == "lr"
    STDERR.puts "Error: train field '#{ks}' is not consumed by microgpt"
    exit 1
  end
  case ks
  when "budget"
    v.as_h.each do |bk, bv|
      bks = bk.as_s
      unless MG_TRAIN_BUDGET_FIELDS.includes?(bks)
        STDERR.puts "Error: unknown train.budget field '#{bks}'"
        exit 1
      end
      case bks
      when "unit"  then budget_unit  = bv.as_s
      when "value" then budget_value = bv.raw.is_a?(Int64) ? bv.as_i.to_f64 : bv.as_f
      end
    end
  when "seed"        then seed_val = v.as_i
  when "quiet"       then quiet    = v.as_bool
  when "seq_len"     then seq_len  = v.as_i
  when "lookahead"   then lookahead = v.as_i
  when "backend"     then backend  = v.as_s
  when "heads"       then heads    = v.as_s
  when "max_depth"
    # Cross-check rule: if both seq_len and max_depth set, must match.
    md = v.as_i
    if md != seq_len
      STDERR.puts "Error: train.seq_len (#{seq_len}) and train.max_depth (#{md}) must match"
      exit 1
    end
  when "optimizer"
    v.as_h.each do |ok, ov|
      oks = ok.as_s
      unless MG_TRAIN_OPTIMIZER_FIELDS.includes?(oks)
        STDERR.puts "Error: train.optimizer field '#{oks}' is not consumed by microgpt"
        exit 1
      end
      case oks
      when "name"
        on = ov.as_s
        unless on == "adam"
          STDERR.puts "Error: microgpt only supports train.optimizer.name=adam (got #{on.inspect})"
          exit 1
        end
        opt_name = on
      when "lr" then lr = ov.raw.is_a?(Int64) ? ov.as_i.to_f64 : ov.as_f
      end
    end
  when "lr_schedule"
    v.as_h.each do |sk, sv|
      sks = sk.as_s
      unless MG_TRAIN_LR_SCHEDULE_FIELDS.includes?(sks)
        STDERR.puts "Error: unknown train.lr_schedule field '#{sks}'"
        exit 1
      end
      case sks
      when "name"           then lr_schedule   = sv.as_s
      when "warmup_epochs"  then warmup_epochs = sv.as_i
      end
    end
  end
end

# Required: budget.unit + value
if budget_unit.empty?
  STDERR.puts "Error: train.budget.unit is required"
  exit 1
end
if budget_value <= 0
  STDERR.puts "Error: train.budget.value is required and must be > 0"
  exit 1
end

# Translate budget → microgpt's --steps
corpus_size = File.size(corpus_path.not_nil!).to_i
steps : Int32 = case budget_unit
                when "steps"
                  budget_value.to_i
                when "epochs"
                  # Convert epochs → steps. Microgpt does sampling-with-replacement
                  # over the corpus. Approximate: steps_per_epoch = corpus_size / seq_len.
                  steps_per_epoch = (corpus_size / seq_len).to_i
                  (budget_value * steps_per_epoch).to_i
                when "wall_seconds"
                  STDERR.puts "Error: train.budget.unit=wall_seconds is not supported by microgpt adapter (microgpt has no wall-time stopping flag)"
                  exit 1
                else
                  STDERR.puts "Error: train.budget.unit must be 'epochs', 'steps', or 'wall_seconds' (got #{budget_unit.inspect})"
                  exit 1
                end

# Convert warmup_epochs → warmup-steps if lr_schedule has warmup
warmup_steps = warmup_epochs > 0 ? (warmup_epochs * (corpus_size / seq_len)).to_i : 0

# Apply --seed override if given
final_seed = seed_override || seed_val

# Resolve model checkpoint path. Microgpt's --model is overloaded
# (load if exists, save at end). Map init_file/save_file:
#   - both specified, different paths → error (microgpt can't do this)
#   - both specified, same path        → use that path
#   - only init_file                    → load + save back (overwrite)
#   - only save_file                    → fresh init, save here
#   - neither                           → microgpt's default (derived from corpus name)
model_path : String? = nil
if init_file && save_file
  if init_file != save_file
    STDERR.puts "Error: microgpt cannot use distinct init_file (#{init_file}) and save_file (#{save_file}) — they must be the same path"
    exit 1
  end
  model_path = init_file
elsif init_file
  model_path = init_file
elsif save_file
  model_path = save_file
end

# Build microgpt argv
argv = [corpus_path.not_nil!]
argv << "--steps" << steps.to_s
argv << "--seed" << final_seed.to_s
argv << "--d-model" << d_model.to_s
argv << "--n-layers" << n_layers.to_s
argv << "--heads" << heads
argv << "--backend" << backend
argv << "--seq-len" << seq_len.to_s
argv << "--lookahead" << lookahead.to_s if lookahead != 0
argv << "--lr" << lr.to_s
argv << "--lr-schedule" << lr_schedule if lr_schedule != "constant"
argv << "--warmup-steps" << warmup_steps.to_s if warmup_steps > 0
if mp = model_path
  argv << "--model" << mp
end
# Microgpt's --no-save: if user gave init_file only (no save_file), they
# probably want to load + save back (overwriting). If they gave only
# save_file, they want to save. Neither + no defaults → microgpt's
# auto-derived path. No-save is set when no model_path resolves.

unless quiet
  STDERR.puts "microgpt_yaml: invoking #{microgpt_bin} with:"
  STDERR.puts "  argv: #{argv.inspect}"
end

# Exec microgpt
exit Process.run(microgpt_bin, args: argv, output: STDOUT, error: STDERR, input: STDIN).exit_code
