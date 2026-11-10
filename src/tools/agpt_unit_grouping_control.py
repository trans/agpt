#!/usr/bin/env python3
"""No-trie control for rnd/context-tree-orientation: does the pd=1 penalty of
the suffix tree come from how its units group contexts, or from the
newest-first reading direction?

The orientation runs (agpt_context_tree_coherence.py) confounded two things.
A suffix-tree pd=1 unit is "all contexts ending in c" AND its model reads
newest-first; a prefix-trie unit is "all contexts starting with c" AND its
model reads oldest-first. Here the tree is gone: every context s (1 <= |s| <=
DEPTH, with its next-char counts) is a training example whose state is
recomputed from its own characters, so any grouping works with any reading
direction. One Adam step per group, event-mean gradient, 65 groups per epoch
in a fresh random order each epoch, as at pd=1.

  reading  fwd   oldest char first (the prefix-trie GRU)
           rev   newest char first (the suffix-tree GRU)
  grouping first   by oldest char  (= prefix-trie root-child units)
           last    by newest char  (= suffix-tree root-child units)
           random  each NODE to a random group every epoch. Lumpy: the 65
                   depth-1 nodes carry an eighth of all events between them,
                   so a small group can be one heavy context. Not a
                   minibatch control; kept for the record.
  Event groupings split a node's events across groups, so they work on
  corpus windows (context train[i:i+L], target train[i+L]) instead of nodes:
           window  each window to a random group every epoch: shuffled
                   minibatch SGD, the "random samples" control
           start   windows grouped by a random label of their start position
                   i, so the up-to-DEPTH windows that start at i stay together
                   (a prefix-trie unit's clumping, without the char sort)
           target  windows grouped by a random label of their target position
                   (a suffix-tree unit's clumping: one target's backoff chain)
           start-char, target-char   label = the char at the start / the
                   newest char: the same groups as first / last, through the
                   event code path (a check of that path)
  Random group probabilities match the first-char units' event shares.

fwd/first and rev/last recompute exactly the pd=1 runs of the orientation
tool (same nodes per step, same unit order), so they check this code path.

Also reports, per grouping, how far each group's target distribution is from
the corpus's: the event-weighted mean KL(p_group || p_all) over groups.

Usage:
  python3 src/tools/agpt_unit_grouping_control.py --out DIR \
      [--cells fwd/first,fwd/last,fwd/window,fwd/start,rev/first,rev/last] \
      [--epochs 500] [--checkpoints 0,3,10,30,100,300,500] [--seed 1] [--chunk 65536]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agpt_context_tree_coherence import (  # noqa: E402
    SPLIT, TreeGRU, build_levels, heldout_nll, level_loss)

CHUNK = 65536
EVENT_GROUPINGS = ("window", "start", "target", "start-char", "target-char")


def decode(code, k, V):
    """Context codes (oldest char most significant) -> [n, k] char ids."""
    out = np.empty((len(code), k), dtype=np.int64)
    c = code.copy()
    for j in range(k - 1, -1, -1):
        out[:, j] = c % V
        c //= V
    return out


def gru_logits(model, X):
    h = torch.zeros(len(X), model.d, device=X.device)
    for t in range(X.shape[1]):
        h = model.cell(model.emb(X[:, t]), h)
    return model.out(h)


def batch_backward(model, Xs, cnt, idxs, scale):
    """Accumulate scale * grad of the loss over the given nodes, level by
    level and in row chunks (the losses are independent, so this is exact)."""
    for k, idx in enumerate(idxs):
        for s in range(0, len(idx), CHUNK):
            i = idx[s:s + CHUNK]
            (level_loss(gru_logits(model, Xs[k][i]), cnt[k][i]) * scale).backward()


def train_loss(model, Xs, cnt, events):
    per = []
    with torch.no_grad():
        for k in range(len(Xs)):
            tot = 0.0
            for s in range(0, len(Xs[k]), CHUNK):
                tot += float(level_loss(gru_logits(model, Xs[k][s:s + CHUNK]), cnt[k][s:s + CHUNK]))
            per.append(tot)
    return sum(per) / sum(events), [p / e for p, e in zip(per, events)]


def fixed_groups(Xs, col, n_groups, dev):
    """Per group, per level: indices of the nodes whose char at column col
    (0 = oldest, -1 = newest) is the group's char."""
    groups = [[] for _ in range(n_groups)]
    for X in Xs:
        key = X[:, col]
        o = torch.argsort(key, stable=True)
        st = torch.searchsorted(key[o], torch.arange(n_groups, device=dev))
        en = torch.searchsorted(key[o], torch.arange(n_groups, device=dev), right=True)
        for g in range(n_groups):
            groups[g].append(o[int(st[g]):int(en[g])])
    return groups


def random_groups(Xs, probs, gen, dev):
    n_groups = len(probs)
    p = torch.tensor(probs, device=dev, dtype=torch.float32)
    groups = [[] for _ in range(n_groups)]
    for X in Xs:
        key = torch.multinomial(p, len(X), replacement=True, generator=gen)
        o = torch.argsort(key, stable=True)
        st = torch.searchsorted(key[o], torch.arange(n_groups, device=dev))
        en = torch.searchsorted(key[o], torch.arange(n_groups, device=dev), right=True)
        for g in range(n_groups):
            groups[g].append(o[int(st[g]):int(en[g])])
    return groups


def window_index(train, levels, dev):
    """Per level k (context length k+1): for every corpus window i (context
    train[i:i+k+1], target train[i+k+1]), its node index in that level."""
    N, out = len(train), []
    V = int(train.max()) + 1
    for k, L in enumerate(levels):
        n = N - (k + 1)
        code = np.zeros(n, dtype=np.int64)
        for j in range(k + 1):
            code = code * V + train[j:j + n]
        nidx = np.searchsorted(L["code"], code)
        assert (L["code"][nidx] == code).all()
        out.append(dict(nidx=torch.from_numpy(nidx).to(dev),
                        tgt=torch.from_numpy(train[k + 1:k + 1 + n].copy()).to(dev)))
    return out


def event_groups(W, labels, n_groups):
    """Group corpus windows (events) instead of nodes. labels[k] gives each
    level-k window its group. Per group, per level: the distinct nodes it
    touches with their event counts, and its events as (pair, target)."""
    groups = [[] for _ in range(n_groups)]
    ar = torch.arange(n_groups, device=labels[0].device)
    for k, w in enumerate(W):
        n_k = int(w["nidx"].max()) + 1
        key = labels[k] * n_k + w["nidx"]
        o = torch.argsort(key)
        uniq, inv, counts = torch.unique_consecutive(key[o], return_inverse=True, return_counts=True)
        off = torch.zeros(len(uniq) + 1, dtype=torch.int64, device=key.device)
        off[1:] = torch.cumsum(counts, 0)
        pg, node, tgt = uniq // n_k, uniq % n_k, w["tgt"][o]
        ps = torch.searchsorted(pg, ar).tolist()
        pe = torch.searchsorted(pg, ar, right=True).tolist()
        offl = off.tolist() if len(off) < 4096 else None
        for g in range(n_groups):
            a, b = ps[g], pe[g]
            ea = offl[a] if offl else int(off[a])
            eb = offl[b] if offl else int(off[b])
            groups[g].append(dict(node=node[a:b], n=counts[a:b].float(), off=off[a:b + 1] - ea,
                                  inv=inv[ea:eb] - a, tgt=tgt[ea:eb]))
    return groups


def event_labels(kind, train_t, W, probs_t, gen):
    if kind == "window":
        return [torch.multinomial(probs_t, len(w["nidx"]), replacement=True, generator=gen) for w in W]
    if kind in ("start", "target"):
        pos = torch.multinomial(probs_t, len(train_t), replacement=True, generator=gen)
    elif kind in ("start-char", "target-char"):
        pos = train_t
    else:
        raise ValueError(kind)
    if kind.startswith("start"):
        return [pos[:len(w["nidx"])] for w in W]
    # target position i+k+1; "target-char" keys on the context's newest char, at i+k
    sh = 0 if kind == "target" else 1
    return [pos[k + 1 - sh:k + 1 - sh + len(w["nidx"])] for k, w in enumerate(W)]


def event_backward(model, Xs, egroup, scale):
    events = 0.0
    for k, gk in enumerate(egroup):
        m = len(gk["node"])
        for s in range(0, m, CHUNK):
            e = min(s + CHUNK, m)
            ea, eb = int(gk["off"][s]), int(gk["off"][e])
            z = gru_logits(model, Xs[k][gk["node"][s:e]])
            loss = (gk["n"][s:e] * torch.logsumexp(z, -1)).sum() - z[gk["inv"][ea:eb] - s, gk["tgt"][ea:eb]].sum()
            (loss * scale).backward()
        events += float(gk["n"].sum()) if m else 0.0
    return events


def group_events(groups, cnt):
    return [float(sum(cnt[k][idx].sum() for k, idx in enumerate(g))) for g in groups]


def target_kl(groups, cnt):
    """Event-weighted mean over groups of KL(group target dist || corpus target dist)."""
    tot = sum(c.sum(0) for c in cnt)
    p_all = tot / tot.sum()
    kls, ws = [], []
    for g in groups:
        t = sum(cnt[k][idx].sum(0) for k, idx in enumerate(g))
        w = float(t.sum())
        if w == 0:
            continue
        p = t / w
        m = p > 0
        kls.append(float((p[m] * (p[m] / p_all[m]).log()).sum()))
        ws.append(w)
    ws = np.array(ws)
    return float(np.dot(kls, ws) / ws.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--checkpoints", default="0,3,10,30,100,300,500")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cells", default="fwd/first,fwd/last,fwd/random,rev/first,rev/last,rev/random")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="rows per forward/backward chunk (memory only)")
    ap.add_argument("--schedule", default="constant", choices=("constant", "cosine"),
                    help="cosine: linear warmup over --warmup-epochs, then cosine to 0 at the last step")
    ap.add_argument("--warmup-epochs", type=float, default=0.0)
    args = ap.parse_args()
    globals()["CHUNK"] = args.chunk
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
    for L in levels:
        del L["cnt"]  # the host copy is ~400 MB at depth 8 and never read again
    events = [float(c.sum()) for c in cnt]
    X_old = [torch.from_numpy(decode(L["code"], k + 1, V)).to(dev) for k, L in enumerate(levels)]
    X_by_read = {"fwd": X_old, "rev": [X.flip(1) for X in X_old]}
    n_groups = len(levels[0]["code"])
    assert n_groups == V
    fixed = {"first": fixed_groups(X_old, 0, n_groups, dev), "last": fixed_groups(X_old, -1, n_groups, dev)}
    fixed_ev = {g: group_events(fixed[g], cnt) for g in fixed}
    probs = np.array(fixed_ev["first"]) / sum(fixed_ev["first"])
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1000)
    kl = {g: target_kl(fixed[g], cnt) for g in fixed}
    kl["random"] = target_kl(random_groups(X_old, probs, gen, dev), cnt)
    W = window_index(train, levels, dev)
    train_t = torch.from_numpy(train).to(dev)
    probs_t = torch.tensor(probs, device=dev, dtype=torch.float32)
    for kind in ("window", "start", "target"):
        eg = event_groups(W, event_labels(kind, train_t, W, probs_t, gen), n_groups)
        tot = sum(torch.bincount(gk["tgt"], minlength=V).float() for g in eg for gk in g)
        p_all = tot / tot.sum()
        kls, ws = [], []
        for g in eg:
            t = sum(torch.bincount(gk["tgt"], minlength=V).float() for gk in g)
            if float(t.sum()) == 0:
                continue
            p = t / t.sum()
            m = p > 0
            kls.append(float((p[m] * (p[m] / p_all[m]).log()).sum()))
            ws.append(float(t.sum()))
        kl[kind] = float(np.dot(kls, ws) / sum(ws))
        del eg
    print(f"table: {sum(len(X) for X in X_old):,} nodes, {sum(events):,.0f} events, {n_groups} groups, "
          f"built in {time.time() - t0:.1f}s", flush=True)
    print("mean KL(group targets || corpus targets), nats: "
          + ", ".join(f"{g} {v:.4f}" for g, v in kl.items()), flush=True)

    ckpts = sorted(int(x) for x in args.checkpoints.split(","))
    summary = dict(args=vars(args), split=SPLIT, V=V, nodes_per_depth=[len(X) for X in X_old],
                   events_per_depth=events, target_kl=kl, cells={})

    for cell in args.cells.split(","):
        read, grouping = cell.split("/")
        Xs = X_by_read[read]
        torch.manual_seed(args.seed)
        model = TreeGRU(V, args.d_model, residual=False).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        rng = np.random.default_rng(args.seed)
        gen = torch.Generator(device=dev).manual_seed(args.seed + 1000)
        recs, t_train = [], 0.0
        total_steps, warm = args.epochs * n_groups, args.warmup_epochs * n_groups

        def set_lr(t):
            """t = epoch * n_groups + position in the epoch's group order."""
            if args.schedule == "constant":
                return
            if t < warm:
                f = (t + 1) / warm
            else:
                f = 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1.0, total_steps - warm)))
            for pg in opt.param_groups:
                pg["lr"] = args.lr * f

        for ep in range(args.epochs + 1):
            if ep in ckpts:
                tl, tl_d = train_loss(model, Xs, cnt, events)
                hn = heldout_nll(model, chunks, D, read == "rev", dev)
                recs.append(dict(epoch=ep, train_loss=tl, train_loss_by_depth=tl_d, heldout_nll=hn))
                print(f"[{cell:10}] ep {ep:4d} train {tl:.4f} heldout {hn:.4f} (ppl {math.exp(hn):.3f})", flush=True)
            if ep == args.epochs:
                break
            ts = time.time()
            if grouping in EVENT_GROUPINGS:
                eg = event_groups(W, event_labels(grouping, train_t, W, probs_t, gen), n_groups)
                gev = [sum(float(gk["n"].sum()) for gk in g) for g in eg]
                for j, g in enumerate(rng.permutation(n_groups)):
                    if gev[g] == 0:
                        continue
                    opt.zero_grad()
                    event_backward(model, Xs, eg[g], 1.0 / gev[g])
                    set_lr(ep * n_groups + j)
                    opt.step()
                del eg
            else:
                if grouping == "random":
                    groups = random_groups(X_old, probs, gen, dev)
                    gev = group_events(groups, cnt)
                else:
                    groups, gev = fixed[grouping], fixed_ev[grouping]
                for j, g in enumerate(rng.permutation(n_groups)):
                    if gev[g] == 0:
                        continue
                    opt.zero_grad()
                    batch_backward(model, Xs, cnt, groups[g], 1.0 / gev[g])
                    set_lr(ep * n_groups + j)
                    opt.step()
            torch.cuda.synchronize()
            t_train += time.time() - ts
        summary["cells"][cell] = dict(checkpoints=recs, train_seconds=t_train)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    print(f"wrote {args.out}/summary.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
