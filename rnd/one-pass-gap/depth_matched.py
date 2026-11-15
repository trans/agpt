#!/usr/bin/env python3
"""Held-out loss by context length at matched overall loss.

Runs that are behind overall are behind mostly at long contexts, because a model learns short-context
statistics first, so comparing plans at equal steps (depth_loss.py) says little about where a plan
falls behind. This compares each run with random-batch runs that reach the same loss at contexts
5-16: the reference is every one-pass run of the random-batch arms (mb, mb84, mb724, mbL, cv), and
each run's loss at 1-4 characters is compared with the reference interpolated linearly at its own
mean loss over contexts 5-16.

Every one-pass seq_len-16 run with a checkpoint is evaluated on the CPU (every window start in every
held-out chunk), cached by run id in depth_matched.json; only runs missing from it are computed.
Usage: python3 rnd/one-pass-gap/depth_matched.py
"""
import glob, json, os, sys
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, 'src/tools'); sys.path.insert(0, 'rnd/one-pass-gap')
from agpt_ppl import AGPTModel, build_vocab, load_model
import summarize as sm

D = 'rnd/one-pass-gap'; OUT = f'{D}/depth_matched.json'; T = 16
REF = ('mb', 'mb84', 'mb724', 'mbL', 'cv')
ARMS = ('agpt', 'tp1x', 'pd2c', 'pd3c', 'pd3cs', 'pd3css', 'tp3x', 'pd4c', 'tq1338', 'tq335', 'tq84', 'mbu3', 'mbu1', 'pk260833', 'pk65208', 'pk16302', 'pk1338', 'pk84')


def depth_loss(path, seqs, V):
    mc, sd = load_model(path); mc['seq_len'] = T
    m = AGPTModel(mc, sd, device='cpu').eval(); tot = torch.zeros(T, dtype=torch.float64); n = 0
    with torch.no_grad():
        for s in seqs:
            st = torch.arange(s.numel() - T)
            for j in range(0, st.numel(), 4096):
                ch = s[st[j:j + 4096].view(-1, 1) + torch.arange(T + 1).view(1, -1)]
                ce = F.cross_entropy(m(ch[:, :T]).reshape(-1, V), ch[:, 1:].reshape(-1), reduction='none').view(-1, T)
                tot += ce.double().sum(0); n += ch.shape[0]
    return (tot / n).tolist()


def main():
    torch.set_num_threads(2)
    rows = [r for r in sm.load() if r['epochs'] == 1 and r['arm'] in REF + ARMS
            and os.path.exists(f"{D}/{r['run']}/checkpoint.model")]
    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [r for r in rows if r['run'] not in cache]
    if todo:
        c2i, V = build_vocab('data/input.txt')
        chunks = sorted(glob.glob('data/.splits/4fa9aec1db6b3aea/heldout_chunks/*.txt'))
        seqs = [torch.tensor([c2i[c] for c in open(p, encoding='utf-8').read() if c in c2i]) for p in chunks]
        for r in todo:
            cache[r['run']] = depth_loss(f"{D}/{r['run']}/checkpoint.model", seqs, V)
            print('computed', r['run'], flush=True)
        json.dump(dict(sorted(cache.items())), open(OUT, 'w'), indent=1)
    deep = lambda v: float(np.mean(v[4:]))
    ref = sorted((deep(cache[r['run']]), cache[r['run']][:4]) for r in rows if r['arm'] in REF)
    X = np.array([x for x, _ in ref]); Y = np.array([y for _, y in ref])
    print(f"reference: {len(ref)} random-batch runs, loss at contexts 5-16 from {X.min():.3f} to {X.max():.3f}\n")
    print("excess over the reference at 1, 2, 3, 4 characters; rolling = their sum / 16 (nats)")
    best = {}
    for r in rows:
        if r['arm'] in ARMS and (r['arm'] not in best or r['rolling'] < best[r['arm']]['rolling']):
            best[r['arm']] = r
    for arm in ARMS:
        for r in sorted((r for r in rows if r['arm'] == arm), key=lambda r: r['lr']):
            v = cache[r['run']]; x = deep(v)
            star = '*' if r is best[arm] else ' '
            if not X.min() <= x <= X.max():
                print(f"  {arm:6} lr {r['lr']:<8g}{star} 5-16: {x:.3f}  outside the reference")
                continue
            ex = [v[k] - np.interp(x, X, Y[:, k]) for k in range(4)]
            print(f"  {arm:6} lr {r['lr']:<8g}{star} 5-16: {x:.3f}  " + ' '.join(f'{e:+.3f}' for e in ex)
                  + f"   rolling {sum(ex) / T:+.3f}")
    print("\n* best rolling PPL of its arm")


if __name__ == '__main__':
    main()
