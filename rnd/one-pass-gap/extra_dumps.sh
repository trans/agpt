#!/bin/sh
# Follow-up 5, run in slot 45 after run_emulation.sh: separate the bf16 K/V cache from TF32 at the
# trained point (the default build with TF32 off, the fp32-cache build with TF32 on). Usage: sh rnd/one-pass-gap/extra_dumps.sh DUMP_DIR
renice -n 19 -p $$ > /dev/null; ionice -c3 -p $$; export OMP_NUM_THREADS=2
cd /home/trans/Projects/air/agpt
OUT=$1; D=rnd/one-pass-gap; CV16=$D/20261115T015547-cv-e16-lr0-006/checkpoint.model
for pd in pd1:$D/configs/agpt-e1-lr0.012.yml pd3s:$D/configs/pd3cs-e1-lr0.0015.yml; do
  name=${pd%%:*}; cfg=${pd#*:}
  for v in bf16_notf32:bin/agpt_train_v2:0 fp32_tf32:bin/agpt_train_v2_kvfp32:1; do
    tag=${v%%:*}; rest=${v#*:}; bin=${rest%%:*}; tf=${rest#*:}
    d=$OUT/${name}_cv16_$tag; mkdir -p $d
    sed -e "s|init_file: .*|init_file: $CV16|" $cfg > $d/cfg.yml
    if [ "$tf" = 0 ]; then AGPT_GRAD_DUMP_DIR=$d NVIDIA_TF32_OVERRIDE=0 $bin --config $d/cfg.yml > $d/train.log 2>&1
    else AGPT_GRAD_DUMP_DIR=$d $bin --config $d/cfg.yml > $d/train.log 2>&1; fi
    echo "dump ${name}_cv16_$tag: exit $?, $(($(wc -l < $d/units.tsv) - 1)) units"
  done
done
