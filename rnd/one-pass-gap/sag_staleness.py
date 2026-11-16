"""How well does each step's direction point along the full-batch gradient at the current weights?

Plain AGPT pd=1 replay (Adam beta1 0.9, lr 0.006) and SAG (beta1 0, lr 0.0015), on the first 20k
characters of the training split, constant lr, 2 passes. At every step: the full-batch mean gradient
at the current weights (every unit), its cosine with the direction Adam is given (plain: the unit's
event-mean gradient; SAG: the table's mass-weighted mean), and the full-batch loss.
Run from the repo root, on the CPU:
    CUDA_VISIBLE_DEVICES= python3 rnd/one-pass-gap/sag_staleness.py rnd/one-pass-gap/sag_staleness.json
"""
import json, sys
import torch, torch.nn.functional as F
sys.path.insert(0, 'src/tools')
import agpt_minibatch_train as amt
from agpt_ppl import AGPTModel, build_vocab, load_model

torch.set_num_threads(2)
T = 16
c2i, V = build_vocab('data/input.txt')
text = open('data/.splits/4fa9aec1db6b3aea/train_corpus.txt', encoding='utf-8').read()[:20000]
tokens = torch.tensor([c2i.get(c, 0) for c in text])
n_starts = tokens.numel() - T - 1
planner = amt.TriePlanner({'trie_plan': 'pd', 'trie_plan_depth': 1}, tokens, n_starts, T)
pos = torch.arange(T, dtype=torch.int32).view(1, T)
offsets = torch.arange(T + 1).view(1, T + 1)


def unit_grad(model, plan, s):
    """summed gradient (flat) and summed loss of plan step s"""
    model.zero_grad(set_to_none=True)
    nb, off = len(planner.blens), plan['off']
    tot = 0.0
    for b, Lb in enumerate(planner.blens):
        a, z = off[s * nb + b], off[s * nb + b + 1]
        if z <= a:
            continue
        ii = torch.from_numpy(plan['sw_i'][a:z])
        ch = tokens[ii.view(-1, 1) + offsets[:, :Lb + 1]]
        ce = F.cross_entropy(model(ch[:, :Lb]).reshape(-1, V), ch[:, 1:].reshape(-1), reduction='none').view(-1, Lb)
        m = ((torch.from_numpy(plan['sw_bits'][a:z]).view(-1, 1) >> pos[:, :Lb]) & 1).to(ce.dtype)
        ls = (ce * m).sum()
        ls.backward()
        tot += ls.item()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()])
    return g.double(), tot


def run(kind, lr, beta1, passes=2):
    import numpy as np
    planner.rng = np.random.default_rng(1)  # same unit orders in both runs
    mc, sd = load_model('rnd/cudax-d16-linear-mass-rerun/seeds/shake-d64L2-h4-dff256-d16-seed42.model')
    mc['seq_len'] = T
    model = AGPTModel(mc, sd, device='cpu')
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, betas=(beta1, 0.999))
    P = sum(p.numel() for p in params)
    U = planner.unit_sizes.size
    table, mass = torch.zeros(U, P, dtype=torch.float64), [0.0] * U
    D, M = torch.zeros(P, dtype=torch.float64), 0.0
    rows = []
    for ep in range(passes):
        plan = planner.epoch(with_cost=False)
        for s in range(plan['n_steps']):
            # full-batch gradient and loss at the current weights
            Gf, Lf, Ef = torch.zeros(P, dtype=torch.float64), 0.0, 0.0
            for t in range(plan['n_steps']):
                g, l = unit_grad(model, plan, t)
                Gf += g; Lf += l; Ef += plan['step_events'][t]
            gfull = Gf / Ef
            g, _ = unit_grad(model, plan, s)
            E = plan['step_events'][s]
            if kind == 'sag':
                u = int(plan['unit_of_step'][s])
                D += g - table[u]; table[u] = g; M += E - mass[u]; mass[u] = E
                d = D / M
            else:
                d = g / E
            cos = float(torch.dot(d, gfull) / (d.norm() * gfull.norm()))
            cos_unit = float(torch.dot(g, gfull) / (g.norm() * gfull.norm()))
            rows.append(dict(pass_=ep + 1, step=s + 1, full_loss=Lf / Ef, cos_dir_full=cos, cos_unit_full=cos_unit,
                             full_norm=float(gfull.norm()), dir_norm=float(d.norm())))
            o = 0
            for p in params:
                n = p.numel()
                if p.grad is None:
                    p.grad = torch.empty_like(p)
                p.grad.copy_(d[o:o + n].view_as(p).float()); o += n
            opt.step()
            print(kind, ep + 1, s + 1, f"loss {Lf / Ef:.4f} cos(dir,full) {cos:+.3f} cos(unit,full) {cos_unit:+.3f}", flush=True)
    return rows


out = {'plain': run('plain', 0.006, 0.9), 'sag': run('sag', 0.0015, 0.0)}
json.dump(out, open(sys.argv[1], 'w'), indent=0)
