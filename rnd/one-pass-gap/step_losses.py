#!/usr/bin/env python3
"""Step-by-step losses of a v2 run (its per-unit mean_loss lines) against its exact replay (log_step_loss lines).
Usage: python3 rnd/one-pass-gap/step_losses.py V2_SLUG:REPLAY_SLUG ...  (run-id slugs, e.g. agpt-e1-lr0-012:tp1x-e1-lr0-012)"""
import re, sys, glob, numpy as np
D = 'rnd/one-pass-gap'
def v2(run): return np.array([float(x) for x in re.findall(r'^    unit \d+/\d+ .*?mean_loss=([0-9.]+)', open(glob.glob(f'{D}/*-{run}')[0] + '/train.log').read(), re.M)])
def rp(run): return np.array([float(x) for x in re.findall(r'^    step \d+ events=\d+ mean_loss=([0-9.]+)', open(sorted(glob.glob(f'{D}/*-{run}'))[-1] + '/train.log').read(), re.M)])
for a, b in [x.split(':') for x in sys.argv[1:]]:
    x, y = v2(a), rp(b)
    if x.size != y.size: print(a, b, 'step counts differ', x.size, y.size); continue
    d = np.abs(x - y); n = x.size
    marks = [i for i in (1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000, n) if i <= n]
    print(f'{a} vs {b}: {n} steps; |loss diff| ' + ', '.join(f'step {i}: {d[i-1]:.1e}' for i in marks)
          + f'; max {d.max():.1e} at step {d.argmax()+1}; mean loss v2 {x.mean():.5f} replay {y.mean():.5f}')
