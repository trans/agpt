#!/usr/bin/env python3
"""Held-out loss by context length (1..16) for one-pass-gap runs: every window start in every held-out
chunk, each run the best-rate run of its arm and budget. Usage: python3 rnd/one-pass-gap/depth_loss.py ARM:EPOCHS ...
Writes rnd/one-pass-gap/depth_loss.json."""
import sys, glob, json, math, torch, torch.nn.functional as F
sys.path.insert(0, 'src/tools'); sys.path.insert(0, 'rnd/one-pass-gap')
from agpt_ppl import AGPTModel, build_vocab, load_model
import summarize as sm
c2i, V = build_vocab('data/input.txt'); T = 16
chunks = sorted(glob.glob('data/.splits/4fa9aec1db6b3aea/heldout_chunks/*.txt'))
seqs = [torch.tensor([c2i[c] for c in open(p, encoding='utf-8').read() if c in c2i]) for p in chunks]
rows = sm.load(); best = {}
for r in rows:
    k = (r['arm'], r['epochs'])
    if k not in best or r['rolling'] < best[k]['rolling']: best[k] = r
def depth_loss(path):
    mc, sd = load_model(path); mc['seq_len'] = T; m = AGPTModel(mc, sd, device='cpu').eval()
    tot = torch.zeros(T, dtype=torch.float64); n = 0
    with torch.no_grad():
        for s in seqs:
            st = torch.arange(s.numel() - T)
            for j in range(0, st.numel(), 4096):
                ch = s[st[j:j + 4096].view(-1, 1) + torch.arange(T + 1).view(1, -1)]
                ce = F.cross_entropy(m(ch[:, :T]).reshape(-1, V), ch[:, 1:].reshape(-1), reduction='none').view(-1, T)
                tot += ce.double().sum(0); n += ch.shape[0]
    return (tot / n).tolist()
want = sys.argv[1:]
out = {}
for k in want:
    arm, e = k.split(':'); r = best[(arm, int(e))]
    d = sorted(glob.glob(f"rnd/one-pass-gap/{r['run']}/checkpoint.model"))[0]
    out[k] = dict(run=r['run'], lr=r['lr'], rolling=r['rolling'], depth=depth_loss(d))
    print(k, r['run'], 'lr', r['lr'], 'rolling', round(r['rolling'], 3), ' '.join(f'{x:.3f}' for x in out[k]['depth']), flush=True)
json.dump(out, open('rnd/one-pass-gap/depth_loss.json', 'w'), indent=1)
