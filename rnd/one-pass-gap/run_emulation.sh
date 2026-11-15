#!/bin/sh
# Follow-up 5 (exact emulation of v2 by the PyTorch replay) and follow-up 4
# (trie targets at longer windows), in one GPU slot. Run only inside a slot
# (sh /home/trans/Projects/air/tools/slots add @agpt ...).
#
# 1. Frozen per-unit gradient dumps from v2 (AGPT_GRAD_DUMP_DIR: every unit's
#    event-mean gradient at the same weights, no step taken) for the pd=1 plan
#    (`agpt`) and the shuffled pd=3 + crown plan (`pd3cs`), at the seed weights
#    and after 16 conventional epochs, with the default build (bf16 K/V cache,
#    TF32 matmuls) and the fp32-cache build with TF32 off.
#    src/tools/agpt_unit_grad_compare.py compares them with PyTorch afterwards.
# 2. The seed-weight default dumps' units.tsv become the exact replays' unit
#    orders (v2_units/), checked against the original runs' unit sequences.
# 3. Orchestrator runs: tp1x, the 36 longer-window configs, tp3x. A failed run
#    is reported and skipped.
# Usage: sh rnd/one-pass-gap/run_emulation.sh DUMP_DIR
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/one-pass-gap
OUT=${1:?usage: run_emulation.sh DUMP_DIR}
SEED=rnd/cudax-d16-linear-mass-rerun/seeds/shake-d64L2-h4-dff256-d16-seed42.model
CV16=$D/20261115T015547-cv-e16-lr0-006/checkpoint.model
mkdir -p $OUT
just build-agpt-train-v2 build-agpt-train-v2-kvfp32 build-agpt-experiment > $D/build.log 2>&1 \
  || { echo "build failed, see $D/build.log"; exit 1; }

# dump NAME BASE_CONFIG INIT BINARY [fp32]
dump() {
  mkdir -p $OUT/$1
  sed -e "s|init_file: .*|init_file: $3|" $2 > $OUT/$1/cfg.yml
  if [ "$5" = fp32 ]; then
    AGPT_GRAD_DUMP_DIR=$OUT/$1 NVIDIA_TF32_OVERRIDE=0 $4 --config $OUT/$1/cfg.yml > $OUT/$1/train.log 2>&1
  else
    AGPT_GRAD_DUMP_DIR=$OUT/$1 $4 --config $OUT/$1/cfg.yml > $OUT/$1/train.log 2>&1
  fi
  s=$?
  echo "dump $1: exit $s, $(($(wc -l < $OUT/$1/units.tsv) - 1)) units"
}
for pd in pd1:$D/configs/agpt-e1-lr0.012.yml pd3s:$D/configs/pd3cs-e1-lr0.0015.yml; do
  name=${pd%%:*}; cfg=${pd#*:}
  for w in seed:$SEED cv16:$CV16; do
    dump ${name}_${w%%:*}_bf16 $cfg ${w#*:} bin/agpt_train_v2
    dump ${name}_${w%%:*}_fp32 $cfg ${w#*:} bin/agpt_train_v2_kvfp32 fp32
  done
done
mkdir -p $D/v2_units
cp $OUT/pd1_seed_bf16/units.tsv $D/v2_units/pd1.tsv
cp $OUT/pd3s_seed_bf16/units.tsv $D/v2_units/pd3s.tsv
# The dumps' unit order against the order the original runs trained in.
python3 - "$D" <<'EOF'
import csv, re, sys
D = sys.argv[1]
for tsv, run in (("pd1", "20261114T083411-agpt-e1-lr0-012"), ("pd3s", "20261114T175616-pd3cs-e1-lr0-0015")):
    rows = list(csv.DictReader(open(f"{D}/v2_units/{tsv}.tsv"), delimiter="\t"))
    dump = [(int(r["unit_index"]) + 1, int(float(r["trained_events"]))) for r in rows]
    log = [(int(a), int(b)) for a, b in re.findall(r"^    unit (\d+)/\d+ .*?trained_events=(\d+)", open(f"{D}/{run}/train.log").read(), re.M)]
    print(f"order {tsv}: {len(dump)} dump rows, {len(log)} logged units, same unit sequence: {dump == log}")
EOF

run() {
  for c in "$@"; do
    bin/agpt_experiment --config $c --trainer src/tools/agpt_minibatch_train.py > $D/last_run.log 2>&1
    s=$?
    case $s in
      0) echo "$(basename $c .yml): $(grep -m1 'lm_eval_rolling_byte_perplexity' $D/last_run.log | tr -s ' ')" ;;
      5) echo "$(basename $c .yml): already run, skipped" ;;
      *) echo "$(basename $c .yml): FAILED (exit $s), log in $OUT/failed-$(basename $c .yml).log"
         cp $D/last_run.log $OUT/failed-$(basename $c .yml).log ;;
    esac
  done
}
run $D/configs/tp1x-*.yml
for e in 1 4; do
  for T in 32 64 128; do
    run $D/configs/cv$T-e$e-*.yml $D/configs/cvt$T-e$e-*.yml
  done
done
run $D/configs/tp3x-*.yml
echo "all done"
