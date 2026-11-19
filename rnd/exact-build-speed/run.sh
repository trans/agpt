#!/bin/sh
# Exact-build speed (the user, 2026-11-18: "want to do that"): v2's training
# seconds in its normal build and in full precision, on two one-pass-gap runs
# (configs/, written by make_configs.py). Repeat 1 of every run, then repeat 2,
# so drift in the GPU's clocks falls on all settings alike. About 30 minutes.
# Run only inside a slot (sh /home/trans/Projects/air/tools/slots add @agpt ...).
# Usage: sh rnd/exact-build-speed/run.sh
# Since fe3a4c8 (full precision the default) this no longer runs as written:
# build-agpt-train-v2-kvfp32 and --trainer v2-kvfp32 are gone. Check out
# 18c104c to rerun the grid as recorded.
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/exact-build-speed
just build-agpt-train-v2 build-agpt-train-v2-kvfp32 build-agpt-experiment > $D/build.log 2>&1 \
  || { echo "build failed, see $D/build.log"; exit 1; }
sha256sum bin/agpt_train_v2 bin/agpt_train_v2_kvfp32

echo "$(date -u +%H:%M) start"
for r in 1 2; do
  for c in pd1 pd3; do
    for s in bf16-tf32 bf16-fp32 fp32-tf32 fp32-fp32; do
      n=$c-$s-r$r
      case $s in fp32-*) t=v2-kvfp32 ;; *) t=v2 ;; esac
      case $s in *-fp32) e="env NVIDIA_TF32_OVERRIDE=0" ;; *) e="env -u NVIDIA_TF32_OVERRIDE" ;; esac
      $e bin/agpt_experiment --config $D/configs/$n.yml --trainer $t > $D/last_run.log 2>&1
      st=$?
      if [ $st -eq 0 ]; then
        j=$(ls -d $D/2026*-$n | tail -1)/result.json
        echo "$(date -u +%H:%M) $n: $(python3 -c "import json; r=json.load(open('$j')); m=r['metrics']; print(f\"train {r['train_wall_seconds']:.1f} s, rolling {m['lm_eval_rolling_byte_perplexity']:.4f}, fixed {m['agpt_fixed_token_perplexity']:.4f}\")")"
      else
        echo "$(date -u +%H:%M) $n: FAILED (exit $st)"; cp $D/last_run.log $D/failed-$n.log
      fi
    done
  done
done
echo "$(date -u +%H:%M) all done"
