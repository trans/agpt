#!/bin/sh
# Check the switch to full precision (the user, 2026-11-18: "I agree, fp32").
# Before rebuilding, keep the two binaries the grid timed (run.sh): the old
# default (bf16 cache, TF32) and the old fp32-cache build. Then rebuild v2 (now
# fp32 cache and fp32 cuBLAS math) and the reduced build (the old default), and:
#  1. frozen-weight gradient dumps at a trained point (pd1-fp32-fp32-r1's
#     checkpoint, one pd=1 pass, no step taken) from each build, compared:
#     new v2 against the old fp32 cache with NVIDIA_TF32_OVERRIDE=0, the
#     reduced build against the old default, each beside a repeat of itself
#     (the floor GPU atomics set), and new v2 against the reduced build;
#  2. the finite-difference check (src/tools/agpt_grad_check.py) on new v2
#     with no env var set;
#  3. one run of each case through bin/agpt_experiment with each new build.
# Run only inside a slot. Usage: sh rnd/exact-build-speed/switch_check.sh
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
unset NVIDIA_TF32_OVERRIDE AGPT_V2_CUBLAS_MATH
D=rnd/exact-build-speed
W=/home/trans/agpt-tries/switch-check
T=$D/20261118T231717-pd1-fp32-fp32-r1/checkpoint.model
C=rnd/one-pass-gap/configs/agpt-e1-lr0.012.yml
mkdir -p $W
[ -f $W/old_default ] || cp bin/agpt_train_v2 $W/old_default
[ -f $W/old_kvfp32 ] || cp bin/agpt_train_v2_kvfp32 $W/old_kvfp32
sha256sum $W/old_default $W/old_kvfp32
just build-agpt-train-v2 build-agpt-train-v2-reduced build-agpt-experiment > $D/build.log 2>&1 \
  || { echo "build failed, see $D/build.log"; exit 1; }
sha256sum bin/agpt_train_v2 bin/agpt_train_v2_reduced

echo "$(date -u +%H:%M) dumps"
sed -e "s|init_file: .*|init_file: $T|" $C > $W/cfg.yml
dump() {  # dump NAME BINARY [ENV...]
  n=$1; b=$2; shift 2
  rm -rf $W/$n; mkdir -p $W/$n
  env "$@" AGPT_GRAD_DUMP_DIR=$W/$n $b --config $W/cfg.yml > $W/$n.log 2>&1
  echo "  $n: exit $?, $(($(wc -l < $W/$n/units.tsv) - 1)) units"
}
dump new bin/agpt_train_v2
dump new2 bin/agpt_train_v2
dump oldexact $W/old_kvfp32 NVIDIA_TF32_OVERRIDE=0
dump reduced bin/agpt_train_v2_reduced
dump reduced2 bin/agpt_train_v2_reduced
dump olddefault $W/old_default
python3 $D/compare_dumps.py $W $D/switch_dumps.json new2:new new:oldexact reduced2:reduced reduced:olddefault reduced:new

echo "$(date -u +%H:%M) finite differences"
python3 src/tools/agpt_grad_check.py --ckpt $T --base-config $C --out $W/fd --eps 1e-2 3e-3 1e-3 --dirs anc \
  > $D/switch_fd.out 2>&1; echo "  exit $?"; tail -12 $D/switch_fd.out

echo "$(date -u +%H:%M) runs"
for c in pd1 pd3; do
  for t in v2 v2-reduced; do
    case $t in v2) n=$c-v2fp32-switch ;; *) n=$c-v2reduced-switch ;; esac
    bin/agpt_experiment --config $D/configs/$n.yml --trainer $t > $D/last_run.log 2>&1
    st=$?
    if [ $st -eq 0 ]; then
      j=$(ls -d $D/2026*-$n | tail -1)/result.json
      echo "$(date -u +%H:%M) $n: $(python3 -c "import json; r=json.load(open('$j')); m=r['metrics']; print(f\"train {r['train_wall_seconds']:.1f} s, rolling {m['lm_eval_rolling_byte_perplexity']:.4f}, fixed {m['agpt_fixed_token_perplexity']:.4f}\")")"
    else
      echo "$(date -u +%H:%M) $n: FAILED (exit $st)"; cp $D/last_run.log $D/failed-$n.log
    fi
  done
done
echo "$(date -u +%H:%M) all done"
