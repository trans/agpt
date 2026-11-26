#!/bin/sh
# The second slot (the user via @air, 2026-11-26): dynamic evaluation of mobius's 100M-trained networks on the
# same 50 books and the 20M books' stream rows (an approximate figure: the rows' counts are not the networks' own;
# mobius 2026-11-26, ~0.0003 bits and < 0.0001 bits). One run at a time: an adaptive run measures ~1.6 GB of
# anonymous host memory on the CPU before the CUDA context, and mobius asked for under 3 GB in the slot.
# The 25.8M's rate is tuned on the tune books (1e-4, the 20M networks' choice, and 2e-4, since the 20M 25.8M was
# still improving at 1e-4); the 10.8M takes 1e-4, the 20M 10.8M's tuned rate. Releases the slot on exit.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=2
trap 'sh /home/trans/Projects/air/tools/slots done @agpt' EXIT
F=/home/trans/Projects/air/mobius/runs/frozen/runs
H25=$F/pd_s25m_stream100m_svd32_b1024.pt
H10=$F/pd_s10m8_stream100m_svd32_b1024.pt
O=out
mkdir -p $O
log() { echo "[$(date -u +%H:%M:%S)] $*" >> chain100m.log; }
one() {  # one NAME CHECKPOINT ARGS...
  name=$1; ck=$2; shift 2
  [ -f $O/$name.json ] && return 0
  nice -n 19 ionice -c3 python3 -I dyneval_prior.py "$ck" $O/$name "$@" > $O/$name.log 2>&1 || { log "$name FAILED: $(tail -1 $O/$name.log)"; return 1; }
  log "$name: $(tail -1 $O/$name.log)"
}
best() {  # best PREFIX: the setting with the lowest bits among out/PREFIX*.json
  python3 -I -c "
import json, glob
r = [json.load(open(p)) for p in glob.glob('$O/$1*.json')]
r = [x for x in r if x['steps'] > 0]
print(min(r, key=lambda x: x['bits'])['setting']['name'])"
}
log "start; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# the 25.8M on 100M: static, its rate on the tune books, then the report books
one static25h $H25 setting=static || exit 1
one t25h-full-adam-1e-4 $H25 setting=full-adam-1e-4 books=tune || exit 1
one t25h-full-adam-2e-4 $H25 setting=full-adam-2e-4 books=tune || exit 1
B=$(best t25h-full) || exit 1
log "tune choice: $B"
one r25h-own $H25 setting=$B books=report || exit 1
one r25h-other $H25 setting=$B books=report past=other || exit 1

# the 10.8M on 100M, for the pair: static, then the report books at the 20M 10.8M's rate
one static10h $H10 setting=static || exit 1
one r10h-own $H10 setting=full-adam-1e-4 books=report || exit 1
one r10h-other $H10 setting=full-adam-1e-4 books=report past=other || exit 1
log "CHAIN DONE"
