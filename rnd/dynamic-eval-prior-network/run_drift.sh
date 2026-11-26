#!/bin/sh
# The third slot (the user via @air, 2026-11-26: "Sure"): limit the drift the other-book control measures, on
# mobius's 100M-trained 25.8M (the headline), then the 100M 10.8M if the clock allows (mobius's GPU run from ~23:00).
# Two arms, each chosen on the tune books and read on the report books with its other-book control:
#   rate:  Adam 3e-5 against the 1e-4 and 2e-4 already run (slot 220)
#   decay: Adam 1e-4 with decay toward the trained weights after each step, lambda 0.003 and 0.01
# One run at a time (host <= 1.9 GB). Releases the slot on exit.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=2
trap 'sh /home/trans/Projects/air/tools/slots done @agpt' EXIT
F=/home/trans/Projects/air/mobius/runs/frozen/runs
H25=$F/pd_s25m_stream100m_svd32_b1024.pt
H10=$F/pd_s10m8_stream100m_svd32_b1024.pt
O=out
log() { echo "[$(date -u +%H:%M:%S)] $*" >> chaindrift.log; }
one() {  # one NAME CHECKPOINT ARGS...
  name=$1; ck=$2; shift 2
  [ -f $O/$name.json ] && return 0
  nice -n 19 ionice -c3 python3 -I dyneval_prior.py "$ck" $O/$name "$@" > $O/$name.log 2>&1 || { log "$name FAILED: $(tail -1 $O/$name.log)"; return 1; }
  log "$name: $(tail -1 $O/$name.log)"
}
best() {  # best REGEX: the setting with the lowest bits among out/t25h-*.json whose setting matches REGEX
  python3 -I -c "
import json, glob, re
r = [json.load(open(p)) for p in glob.glob('$O/t25h-*.json')]
r = [x for x in r if x['steps'] > 0 and re.fullmatch(r'$1', x['setting']['name'])]
print(min(r, key=lambda x: x['bits'])['setting']['name'])"
}
log "start; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

one t25h-full-adam-1e-4-d0.01 $H25 setting=full-adam-1e-4-d0.01 books=tune || exit 1
one t25h-full-adam-1e-4-d0.003 $H25 setting=full-adam-1e-4-d0.003 books=tune || exit 1
one t25h-full-adam-3e-5 $H25 setting=full-adam-3e-5 books=tune || exit 1
D=$(best 'full-adam-1e-4-d.*') || exit 1
R=$(best 'full-adam-[0-9.e-]+') || exit 1
A=$(best '.*') || exit 1
log "tune choice: decay $D, rate $R, overall $A"

one r25h-own-decay $H25 setting=$D books=report || exit 1
one r25h-other-decay $H25 setting=$D books=report past=other || exit 1
if [ "$R" != full-adam-1e-4 ]; then   # 1e-4 is r25h-own / r25h-other already
  one r25h-own-rate $H25 setting=$R books=report || exit 1
  one r25h-other-rate $H25 setting=$R books=report past=other || exit 1
fi
if [ "$(date -u +%H%M)" -lt 2300 ] && [ "$A" != full-adam-1e-4 ]; then
  one r10h-own-best $H10 setting=$A books=report || exit 1
  one r10h-other-best $H10 setting=$A books=report past=other || exit 1
else
  log "10.8M skipped (clock $(date -u +%H:%M) or overall choice $A)"
fi
log "CHAIN DONE"
