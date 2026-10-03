#!/usr/bin/env python3
"""Prefix trie vs recency-ordered context tree: aggregation coherence
(rnd/context-tree-orientation, todo/context-tree-orientation.md).

One node table, two trees. Nodes are every context string s with
1 <= |s| <= DEPTH seen in the training slice, each with its next-char counts
n(s, x). Both trees use exactly these nodes and counts, so the loss
sum_s sum_x n(s,x) * -log p(x|s) is identical; only the edges differ:

  fwd     prefix trie:  parent(s) = s[:-1], node state reads s's NEWEST char
          (left-to-right GRU, today's AGPT orientation)
  rev     suffix tree:  parent(s) = s[1:],  node state reads s's OLDEST char
          (newest-first GRU)
  rev-res suffix tree with residual readout z_s = z_parent(s) + W h_s + b

All three start from the same initial weights and train by full-batch Adam on
the whole tree (no partitioning, so the update sequence is not confounded by
unit choice). At each checkpoint:

  * train loss overall and by depth; held-out NLL with context <= DEPTH
    (diagnostic, not canonical)
  * node coherence (the Jacobian reduction, paper section 4-5): at node p the
    parts are p's own readout adjoint at h_p and each child's contribution
    J^T G_child; rho_p = ||sum parts|| / sum ||parts||. Reported per depth as
    the kept fraction sum_p ||G_p|| / sum_p sum ||parts||. rev-res also gets
    the logit-path version (parts e_p and children's E_c).
  * unit coherence in parameter space, as in rnd/gradient-population
    Experiment 1: units are depth-2 subtrees (their depth-1 ancestor is
    context only). rho global = what a pd=0 step keeps of them, rho per-root
    = what a pd=1 step keeps; plus pd=1 units (whole root subtrees) globally.

Usage:
  python3 src/tools/agpt_context_tree_coherence.py --out DIR \
      [--depth 8] [--d-model 64] [--steps 1000] [--lr 3e-3] \
      [--checkpoints 0,30,100,300,1000] [--seed 1] [--models fwd,rev,rev-res]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPLIT = "data/.splits/4fa9aec1db6b3aea"


# --------------------------------------------------------------------------
# node table


def build_levels(ids, V, D):
    """Per depth k (1..D): unique context codes (lexicographic, oldest char
    most significant), dense next-char counts, parent index in each tree,
    and the char each tree's node state reads."""
    N = len(ids)
    levels = []
    prev = None
    for k in range(1, D + 1):
        n = N - k
        code = np.zeros(n, dtype=np.int64)
        for j in range(k):
            code = code * V + ids[j:j + n]
        tgt = ids[k:k + n]
        uniq, inv = np.unique(code, return_inverse=True)
        cnt = np.bincount(inv * V + tgt, minlength=len(uniq) * V).reshape(-1, V).astype(np.float32)
        if k == 1:
            par_pre = np.zeros(len(uniq), dtype=np.int64)
            par_suf = np.zeros(len(uniq), dtype=np.int64)
        else:
            pp, ps = uniq // V, uniq % (V ** (k - 1))
            par_pre = np.minimum(np.searchsorted(prev, pp), len(prev) - 1)
            par_suf = np.minimum(np.searchsorted(prev, ps), len(prev) - 1)
            assert (prev[par_pre] == pp).all() and (prev[par_suf] == ps).all()
        levels.append(dict(code=uniq, cnt=cnt, par_pre=par_pre, par_suf=par_suf,
                           newest=uniq % V, oldest=uniq // (V ** (k - 1))))
        prev = uniq
    return levels


def unit_index(levels, pre, V, dev, A):
    """Group nodes at depth >= A by their depth-A ancestor (the first A chars
    in the prefix trie, the last A chars in the suffix tree). Per level k
    (0-indexed, k >= A-1): node order with each unit contiguous, per-unit
    [start, end) and each node's parent position local to its unit's
    previous-level slice (0 at the anchor level)."""
    D = len(levels)
    codeA = levels[A - 1]["code"]
    n_units = len(codeA)
    order, starts, ends, local_par = [None] * D, [None] * D, [None] * D, [None] * D
    posinv_prev = None
    for k in range(A - 1, D):
        c = levels[k]["code"]
        key_code = (c // (V ** (k + 1 - A))) if pre else (c % (V ** A))
        key = np.searchsorted(codeA, key_code)
        assert (codeA[key] == key_code).all()
        o = np.argsort(key, kind="stable")
        skey = key[o]
        st = np.searchsorted(skey, np.arange(n_units), "left")
        en = np.searchsorted(skey, np.arange(n_units), "right")
        posinv = np.empty_like(o)
        posinv[o] = np.arange(len(o))
        if k == A - 1:
            lp = np.zeros(len(o), dtype=np.int64)
        else:
            parent = levels[k]["par_pre" if pre else "par_suf"][o]
            lp = posinv_prev[parent] - starts[k - 1][skey]
        order[k] = torch.from_numpy(o).to(dev)
        starts[k], ends[k] = st, en
        local_par[k] = torch.from_numpy(lp).to(dev)
        posinv_prev = posinv
    return dict(A=A, n_units=n_units, order=order, starts=starts, ends=ends, local_par=local_par)


class Tree:
    """One orientation over the shared node table, on device."""

    def __init__(self, levels, orient, V, dev):
        self.orient = orient
        self.D = len(levels)
        pre = orient == "fwd"
        self.par = [torch.from_numpy(L["par_pre" if pre else "par_suf"]).to(dev) for L in levels]
        self.inp = [torch.from_numpy(L["newest" if pre else "oldest"]).to(dev) for L in levels]
        self.n = [len(L["code"]) for L in levels]
        # depth-2 units (coherence, as in gradient-population Exp 1)
        u2 = unit_index(levels, pre, V, dev, 2)
        self.n_units = u2["n_units"]
        self.order, self.starts, self.ends, self.local_par = u2["order"], u2["starts"], u2["ends"], u2["local_par"]
        code2 = levels[1]["code"]
        self.unit_root = np.searchsorted(levels[0]["code"], (code2 // V) if pre else (code2 % V))
        # depth-1 units (pd=1 training): whole root-child subtrees
        self.roots = unit_index(levels, pre, V, dev, 1)


class TreeGRU(nn.Module):
    def __init__(self, V, d, residual):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.cell = nn.GRUCell(d, d)
        self.out = nn.Linear(d, V)
        self.residual = residual
        self.d = d


def level_loss(z, c):
    """sum_rows -sum_x c_x log softmax(z)_x, without materializing log_softmax."""
    return (c.sum(1) * torch.logsumexp(z, -1) - (c * z).sum(1)).sum()


def run_tree(model, T, cnt):
    """Plain-autograd full-tree loss. Only for small depths (verification)."""
    dev = cnt[0].device
    h_prev, z_prev, total = torch.zeros(1, model.d, device=dev, dtype=cnt[0].dtype), None, 0.0
    for k in range(T.D):
        h = model.cell(model.emb(T.inp[k]), h_prev[T.par[k]])
        z = model.out(h)
        if model.residual and k > 0:
            z = z + z_prev[T.par[k]]
        total = total + level_loss(z, cnt[k])
        h_prev, z_prev = h, z
    return total


def forward_states(model, T, cnt):
    """No-grad forward over the whole tree: node states (and logits for the
    residual readout), plus the loss per depth."""
    dev = cnt[0].device
    hs, zs, per_depth = [], [], []
    with torch.no_grad():
        h_prev, z_prev = torch.zeros(1, model.d, device=dev, dtype=cnt[0].dtype), None
        for k in range(T.D):
            h = model.cell(model.emb(T.inp[k]), h_prev[T.par[k]])
            z = model.out(h)
            if model.residual and k > 0:
                z = z + z_prev[T.par[k]]
            per_depth.append(float(level_loss(z, cnt[k])))
            hs.append(h)
            zs.append(z if model.residual else None)
            h_prev, z_prev = h, z
    return hs, zs, per_depth


def tree_backward(model, T, cnt, hs, zs, scale=1.0, coherence=False):
    """Exact gradient of scale * total loss, accumulated into .grad, one depth
    at a time from the leaves rootward. Each depth is recomputed from its
    parents' stored states, and the children's summed adjoint is injected as
    sum(h * A_h) (+ sum(z * A_z) for the residual readout): the trie
    recursion G_p = g_p + sum_x J^T G_px, with autograd doing each J^T.

    With coherence=True also returns, per depth, the node coherence of that
    sum: parts are the readout-path adjoint at h_p and each child's GRU-path
    contribution (and, for the residual readout, e_p and each child's
    logit-path contribution)."""
    dev = cnt[0].device
    W = model.out.weight
    A_h = A_z = ch_rows = zch_rows = None
    node = []
    for k in reversed(range(T.D)):
        if k > 0:
            hp = hs[k - 1][T.par[k]].detach().requires_grad_()
        else:
            hp = torch.zeros(T.n[0], model.d, device=dev, dtype=cnt[0].dtype)
        h = model.cell(model.emb(T.inp[k]), hp)
        z = model.out(h)
        zp = None
        if model.residual and k > 0:
            zp = zs[k - 1][T.par[k]].detach().requires_grad_()
            z = z + zp
        sur = level_loss(z, cnt[k]) * scale
        if A_h is not None:
            sur = sur + (h * A_h).sum()
        if A_z is not None:
            sur = sur + (z * A_z).sum()
        sur.backward()
        if coherence and A_h is not None:
            with torch.no_grad():
                par = T.par[k + 1]
                e_own = (F.softmax(z, -1) * cnt[k].sum(1, keepdim=True) - cnt[k]) * scale
                read = e_own + A_z if A_z is not None else e_own
                own_h = read @ W
                Gn = (own_h + A_h).norm(dim=1)
                parts = own_h.norm(dim=1).index_add_(0, par, ch_rows.norm(dim=1))
                nchild = torch.zeros(T.n[k], device=dev, dtype=cnt[0].dtype).index_add_(0, par, torch.ones(len(par), device=dev, dtype=cnt[0].dtype))
                rho = Gn / parts.clamp_min(1e-30)
                rec = dict(depth=k + 1, kept_h=float(Gn.sum() / parts.sum()),
                           median_rho_h=float(rho[nchild >= 1].median()))
                if A_z is not None:
                    En = (e_own + A_z).norm(dim=1)
                    zparts = e_own.norm(dim=1).index_add_(0, par, zch_rows.norm(dim=1))
                    rec["kept_z"] = float(En.sum() / zparts.sum())
                node.append(rec)
        if k > 0:
            ch_rows = hp.grad
            A_h = torch.zeros(T.n[k - 1], model.d, device=dev, dtype=cnt[0].dtype).index_add_(0, T.par[k], hp.grad)
            if zp is not None:
                zch_rows = zp.grad
                A_z = torch.zeros(T.n[k - 1], z.shape[1], device=dev, dtype=cnt[0].dtype).index_add_(0, T.par[k], zp.grad)
    node.reverse()
    return node


def subtree_backward(model, T, cnt, U, u, scale=1.0):
    """Exact gradient of scale * (loss of unit u's subtree), accumulated into
    .grad: the same leaves-first level sweep as tree_backward, restricted to
    one depth-1 unit (a pd=1 training unit). Returns the unit's event count."""
    assert U["A"] == 1
    dev, dt = cnt[0].device, cnt[0].dtype
    hs, zs, idxs, lps = [], [], [], []
    with torch.no_grad():
        h_prev, z_prev = torch.zeros(1, model.d, device=dev, dtype=dt), None
        for k in range(T.D):
            s, e = int(U["starts"][k][u]), int(U["ends"][k][u])
            if s == e:
                break
            idx, lp = U["order"][k][s:e], U["local_par"][k][s:e]
            h = model.cell(model.emb(T.inp[k][idx]), h_prev[lp])
            z = model.out(h)
            if model.residual and k > 0:
                z = z + z_prev[lp]
            hs.append(h)
            zs.append(z if model.residual else None)
            idxs.append(idx)
            lps.append(lp)
            h_prev, z_prev = h, z
    A_h = A_z = None
    events = 0.0
    for k in reversed(range(len(idxs))):
        idx, lp = idxs[k], lps[k]
        if k > 0:
            hp = hs[k - 1][lp].detach().requires_grad_()
        else:
            hp = torch.zeros(len(idx), model.d, device=dev, dtype=dt)
        h = model.cell(model.emb(T.inp[k][idx]), hp)
        z = model.out(h)
        zp = None
        if model.residual and k > 0:
            zp = zs[k - 1][lp].detach().requires_grad_()
            z = z + zp
        c = cnt[k][idx]
        events += float(c.sum())
        sur = level_loss(z, c) * scale
        if A_h is not None:
            sur = sur + (h * A_h).sum()
        if A_z is not None:
            sur = sur + (z * A_z).sum()
        sur.backward()
        if k > 0:
            A_h = torch.zeros(len(idxs[k - 1]), model.d, device=dev, dtype=dt).index_add_(0, lp, hp.grad)
            if zp is not None:
                A_z = torch.zeros(len(idxs[k - 1]), z.shape[1], device=dev, dtype=dt).index_add_(0, lp, zp.grad)
    return events


def node_coherence(model, T, cnt):
    hs, zs, _ = forward_states(model, T, cnt)
    model.zero_grad()
    node = tree_backward(model, T, cnt, hs, zs, coherence=True)
    model.zero_grad()
    return node


def flat_grad(model, loss):
    gs = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, p in zip(gs, model.parameters())])


def unit_loss(model, T, cnt, u):
    dev = cnt[0].device
    r = int(T.unit_root[u])
    h = model.cell(model.emb(T.inp[0][r:r + 1]), torch.zeros(1, model.d, device=dev, dtype=cnt[0].dtype))
    z = model.out(h)
    loss = 0.0
    events = 0.0
    for k in range(1, T.D):
        s, e = int(T.starts[k][u]), int(T.ends[k][u])
        if s == e:
            break
        idx = T.order[k][s:e]
        lp = T.local_par[k][s:e]
        hn = model.cell(model.emb(T.inp[k][idx]), h[lp])
        zn = model.out(hn)
        if model.residual:
            zn = zn + z[lp]
        c = cnt[k][idx]
        loss = loss + level_loss(zn, c)
        events += float(c.sum())
        h, z = hn, zn
    return loss, events


def root_own_loss(model, T, cnt, r):
    dev = cnt[0].device
    h = model.cell(model.emb(T.inp[0][r:r + 1]), torch.zeros(1, model.d, device=dev, dtype=cnt[0].dtype))
    return level_loss(model.out(h), cnt[0][r:r + 1]), float(cnt[0][r].sum())


def rho(rows):
    return float(rows.sum(0).norm() / rows.norm(dim=1).sum().clamp_min(1e-30))


def unit_coherence(model, T, cnt):
    G = []
    ev = []
    for u in range(T.n_units):
        loss, e = unit_loss(model, T, cnt, u)
        if e == 0:
            continue
        G.append(flat_grad(model, loss))
        ev.append((u, e))
    units = np.array([u for u, _ in ev])
    evs = torch.tensor([e for _, e in ev], device=G[0].device)
    G = torch.stack(G)
    g = G / evs[:, None]
    roots = T.unit_root[units]
    res = dict(n_units=len(units), rho_global_w=rho(G), rho_global_unw=rho(g))
    per_w, per_u, wts = [], [], []
    Gr = []
    for r in np.unique(roots):
        m = torch.from_numpy(roots == r).to(G.device)
        per_w.append(rho(G[m]))
        per_u.append(rho(g[m]))
        wts.append(float(evs[m].sum()))
        own, _ = root_own_loss(model, T, cnt, int(r))
        Gr.append(G[m].sum(0) + flat_grad(model, own))
    wts = np.array(wts)
    res["rho_per_root_w"] = float(np.dot(per_w, wts) / wts.sum())
    res["rho_per_root_unw"] = float(np.mean(per_u))
    res["rho_pd1_units_global_w"] = rho(torch.stack(Gr))
    gn = g / g.norm(dim=1, keepdim=True).clamp_min(1e-30)
    C = gn @ gn.T
    rt = torch.from_numpy(roots).to(G.device)
    same = rt[:, None] == rt[None, :]
    off = ~torch.eye(len(units), dtype=torch.bool, device=G.device)
    res["cos_within_root"] = float(C[same & off].mean())
    res["cos_across_root"] = float(C[~same].mean())
    res["frac_pairs_neg"] = float((C[off] < 0).float().mean())
    return res


def heldout_nll(model, chunks, D, rev, dev):
    total, n = 0.0, 0
    with torch.no_grad():
        for L in range(1, D + 1):
            X, Y = [], []
            for ch in chunks:
                js = [L] if L < D else range(D, len(ch))
                for j in js:
                    if j < len(ch):
                        X.append(ch[j - L:j])
                        Y.append(ch[j])
            if not X:
                continue
            X = torch.tensor(np.array(X), device=dev)
            Y = torch.tensor(np.array(Y), device=dev)
            if rev:
                X = X.flip(1)
            h = torch.zeros(len(X), model.d, device=dev)
            zsum = 0.0
            for t in range(L):
                h = model.cell(model.emb(X[:, t]), h)
                if model.residual:
                    zsum = zsum + model.out(h)
            z = zsum if model.residual else model.out(h)
            total += float(F.cross_entropy(z, Y, reduction="sum"))
            n += len(Y)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--pd", type=int, default=0, choices=(0, 1),
                    help="0: one full-batch Adam step per epoch; 1: one step per root-child subtree "
                         "(event-mean gradient, unit order reshuffled each epoch, same order for every model)")
    ap.add_argument("--steps", type=int, default=1000, help="epochs (at pd=0 an epoch is one step)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--checkpoints", default="0,30,100,300,1000")
    ap.add_argument("--unit-checkpoints", default=None,
                    help="checkpoints that also get unit coherence (default: all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--models", default="fwd,rev,rev-res")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    vocab = sorted(set(open("data/input.txt", encoding="utf-8").read()))
    c2i = {c: i for i, c in enumerate(vocab)}
    V, D = len(vocab), args.depth
    train = np.array([c2i[c] for c in open(f"{SPLIT}/train_corpus.txt", encoding="utf-8").read()], dtype=np.int64)
    chunk_dir = f"{SPLIT}/heldout_chunks"
    chunks = [np.array([c2i[c] for c in open(os.path.join(chunk_dir, f), encoding="utf-8").read()], dtype=np.int64)
              for f in sorted(os.listdir(chunk_dir))]

    t0 = time.time()
    levels = build_levels(train, V, D)
    cnt = [torch.from_numpy(L["cnt"]).to(dev) for L in levels]
    events = [float(c.sum()) for c in cnt]
    total_events = sum(events)
    trees = {}
    for o in ("fwd", "rev"):
        trees[o] = Tree(levels, o, V, dev)
    n_nodes = [len(L["code"]) for L in levels]
    print(f"table: {sum(n_nodes):,} nodes {n_nodes}, {total_events:,.0f} events, "
          f"{trees['fwd'].n_units} depth-2 units, built in {time.time() - t0:.1f}s", flush=True)

    ckpts = sorted(int(x) for x in args.checkpoints.split(","))
    uckpts = set(ckpts if args.unit_checkpoints is None else (int(x) for x in args.unit_checkpoints.split(",")))
    summary = dict(args=vars(args), split=SPLIT, V=V, nodes_per_depth=n_nodes,
                   events_per_depth=events, models={})

    for name in args.models.split(","):
        T = trees["fwd" if name == "fwd" else "rev"]
        torch.manual_seed(args.seed)
        model = TreeGRU(V, args.d_model, residual=(name == "rev-res")).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        recs = []
        t_train = 0.0
        rng = np.random.default_rng(args.seed)
        if args.pd == 1:
            R = T.roots
            unit_events = [float(sum(cnt[k][R["order"][k][int(R["starts"][k][u]):int(R["ends"][k][u])]].sum()
                                     for k in range(T.D))) for u in range(R["n_units"])]
        for step in range(args.steps + 1):
            if step in ckpts:
                _, _, per_depth = forward_states(model, T, cnt)
                rec = dict(step=step, pd=args.pd, train_loss=sum(per_depth) / total_events,
                           train_loss_by_depth=[float(l) / e for l, e in zip(per_depth, events)],
                           heldout_nll=heldout_nll(model, chunks, D, name != "fwd", dev),
                           node=node_coherence(model, T, cnt))
                if step in uckpts:
                    tu = time.time()
                    rec["unit"] = unit_coherence(model, T, cnt)
                    rec["unit_seconds"] = time.time() - tu
                recs.append(rec)
                nd = rec["node"]
                line = (f"[{name:7}] pd{args.pd} ep {step:5d} train {rec['train_loss']:.4f} "
                        f"heldout {rec['heldout_nll']:.4f} (ppl {math.exp(rec['heldout_nll']):.3f}) "
                        f"node kept_h d1 {nd[0]['kept_h']:.3f} d2 {nd[1]['kept_h']:.3f} d3 {nd[2]['kept_h']:.3f}")
                if "kept_z" in nd[0]:
                    line += f" kept_z d1 {nd[0]['kept_z']:.3f}"
                if "unit" in rec:
                    un = rec["unit"]
                    line += (f" | unit rho_g_w {un['rho_global_w']:.3f} rho_root_w {un['rho_per_root_w']:.3f} "
                             f"pd1 {un['rho_pd1_units_global_w']:.3f} cos in/out {un['cos_within_root']:.3f}/"
                             f"{un['cos_across_root']:.3f} neg {un['frac_pairs_neg']:.2f}")
                print(line, flush=True)
            if step == args.steps:
                break
            ts = time.time()
            if args.pd == 0:
                hs, zs, _ = forward_states(model, T, cnt)
                opt.zero_grad()
                tree_backward(model, T, cnt, hs, zs, scale=1.0 / total_events)
                del hs, zs
                opt.step()
            else:
                for u in rng.permutation(T.roots["n_units"]):
                    if unit_events[u] == 0:
                        continue
                    opt.zero_grad()
                    subtree_backward(model, T, cnt, T.roots, int(u), scale=1.0 / unit_events[u])
                    opt.step()
            torch.cuda.synchronize()
            t_train += time.time() - ts
        summary["models"][name] = dict(checkpoints=recs, train_seconds=t_train)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    print(f"wrote {args.out}/summary.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
