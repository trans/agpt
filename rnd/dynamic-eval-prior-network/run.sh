#!/bin/sh
# The slot: dynamic evaluation of mobius's prior + stream networks on its 50 whole books (dyneval_prior.py).
# Two runs at a time (each ~1 GB host, well under 1 GB GPU at batch 1). Settings are chosen on the tune books
# (even indices) and read on the report books (odd). A finished run (its .json exists) is not run again.
# The chain releases the slot on exit however it ends.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=2
trap 'sh /home/trans/Projects/air/tools/slots done @agpt' EXIT
F=/home/trans/Projects/air/mobius/runs/frozen/runs
N10=$F/pd_s10.8m_stream_svd32_b1024.pt
N25=$F/pd_s25m_stream_svd32_b1024.pt
O=out
mkdir -p $O
log() { echo "[$(date -u +%H:%M:%S)] $*" >> chain.log; }
one() {  # one NAME CHECKPOINT ARGS...
  name=$1; ck=$2; shift 2
  [ -f $O/$name.json ] && return 0
  nice -n 19 ionice -c3 python3 -I dyneval_prior.py "$ck" $O/$name "$@" > $O/$name.log 2>&1 || { log "$name FAILED: $(tail -1 $O/$name.log)"; return 1; }
  log "$name: $(tail -1 $O/$name.log)"
}
best() {  # best PREFIX: the setting with the lowest bits among out/PREFIX*.json
  python3 -I -c "
import json, sys, glob
r = [json.load(open(p)) for p in glob.glob('$O/$1*.json')]
r = [x for x in r if x['steps'] > 0]
b = min(r, key=lambda x: x['bits'])
print(b['setting']['name'])"
}
log "start; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# sizing: 40 adaptive windows
one size10 $N10 setting=full-adam-1e-4 books=0-1 limit=40 || exit 1
log "  sizing: $(grep -o '[0-9.]* s;.*' $O/size10.log | head -1)"

# static, every book, both networks
one static10 $N10 setting=static & one static25 $N25 setting=static & wait

# the 10.8M on the tune books
one t10-full-adam-1e-5 $N10 setting=full-adam-1e-5 books=tune & one t10-full-adam-3e-5 $N10 setting=full-adam-3e-5 books=tune & wait
one t10-full-adam-1e-4 $N10 setting=full-adam-1e-4 books=tune & one t10-full-adam-3e-4 $N10 setting=full-adam-3e-4 books=tune & wait
one t10-lora16-adam-1e-3 $N10 setting=lora16-adam-1e-3 books=tune & one t10-lora16-adam-3e-3 $N10 setting=lora16-adam-3e-3 books=tune & wait
B=$(best t10-full) || exit 1
L=$(best t10-lora) || exit 1
log "tune choice: $B, $L"

# the 10.8M on the report books: own past and the other-book control at the chosen setting
one r10-own $N10 setting=$B books=report & one r10-other $N10 setting=$B books=report past=other & wait
# apart: everything adapting (inp and gate too) at the same rate, and the best LoRA
A=$(echo $B | sed 's/^full/all/')
one r10-all $N10 setting=$A books=report & one r10-lora $N10 setting=$L books=report & wait

# the 25.8M: its rate on the tune books (the 10.8M's and a third of it), then the report books
LR=${B##*-adam-}
LR3=$(python3 -I -c "print(f'{float(\"$LR\") / 3:.0e}')")
one t25-full-adam-$LR $N25 setting=full-adam-$LR books=tune & one t25-full-adam-$LR3 $N25 setting=full-adam-$LR3 books=tune & wait
B25=$(best t25-full) || exit 1
log "tune choice 25.8M: $B25"
one r25-own $N25 setting=$B25 books=report & one r25-other $N25 setting=$B25 books=report past=other & wait
log "CHAIN DONE"
