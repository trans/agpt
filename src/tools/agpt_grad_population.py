#!/usr/bin/env python3
"""Analyze a gradient-population dump (rnd/gradient-population).

Input dir (written by bin/agpt_train_v2 with AGPT_GRAD_DUMP_DIR set):
  grads.f32   N rows x P float32, one event-mean gradient per training unit
  units.tsv   per-row metadata (anchor, root child, context tokens, events, ...)
  layout.json parameter section offsets

Reports, for the frozen fan-out population {delta_i}:
  * gradient coherence rho(S) = ||sum_{i in S} delta_i|| / sum_{i in S} ||delta_i||
    globally and per root-child subtree (event-weighted and unweighted)
  * cosine similarity within vs across root-child subtrees (trie distance)
  * covariance spectrum of the population (participation ratio, top-k variance)
  * per-parameter-section coherence

Usage:
  python3 src/tools/agpt_grad_population.py DUMP_DIR [--vocab data/input.txt] [--out summary.json]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return sorted(set(f.read()))


def decode(tokens, vocab):
    out = []
    for t in tokens.split(","):
        if t == "":
            continue
        c = vocab[int(t)]
        out.append(c if c != "\n" else "\\n")
    return "".join(out)


def rho(vectors, weights=None):
    """Coherence of a set of vectors: ||sum w_i v_i|| / sum ||w_i v_i||."""
    if weights is None:
        weights = np.ones(vectors.shape[0], dtype=np.float64)
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1) * weights
    s = np.linalg.norm((vectors.astype(np.float64) * weights[:, None]).sum(axis=0))
    denom = norms.sum()
    return float(s / denom) if denom > 0 else float("nan")


def rho_from_gram(G, norms, idx, weights=None):
    """Same as rho() but from a precomputed Gram matrix (dot products)."""
    idx = np.asarray(idx)
    w = np.ones(len(idx)) if weights is None else np.asarray(weights, dtype=np.float64)
    sub = G[np.ix_(idx, idx)]
    s2 = float(w @ sub @ w)
    denom = float((norms[idx] * w).sum())
    return (np.sqrt(max(s2, 0.0)) / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--vocab", default="data/input.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    d = args.dump_dir
    layout = json.load(open(os.path.join(d, "layout.json")))
    P = layout["total_floats"]
    vocab = load_vocab(args.vocab)

    rows = []
    with open(os.path.join(d, "units.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    N = len(rows)
    X = np.fromfile(os.path.join(d, "grads.f32"), dtype=np.float32)
    assert X.size == N * P, f"grads.f32 has {X.size} floats, expected {N}x{P}"
    X = X.reshape(N, P)

    events = np.array([float(r["trained_events"]) for r in rows])
    root = np.array([int(r["root_child_id"]) for r in rows])
    anchor = np.array([int(r["anchor_id"]) for r in rows])
    ctx = [decode(r["context_tokens"], vocab) for r in rows]
    loss = np.array([float(r["mean_loss"]) for r in rows])

    norms = np.sqrt(np.einsum('ij,ij->i', X, X, dtype=np.float64))
    # Gram matrix of the raw mean-gradients (float64 accumulate via float32 matmul chunks)
    # Gram matrix accumulated in float64 from float32 column chunks (peak memory ~ X + one chunk).
    G = np.zeros((N, N), dtype=np.float64)
    step = 8192
    for c0 in range(0, P, step):
        blk = X[:, c0:c0 + step].astype(np.float64)
        G += blk @ blk.T
    del blk
    Gn = G / np.outer(norms, norms)  # cosine matrix

    summary = {"dump_dir": d, "N": N, "P": P, "partition_depth": layout["partition_depth"],
               "apply_step": layout["apply_step"], "total_events": float(events.sum())}

    # ---- global coherence -------------------------------------------------
    all_idx = np.arange(N)
    summary["rho_global_unweighted"] = rho_from_gram(G, norms, all_idx)
    summary["rho_global_event_weighted"] = rho_from_gram(G, norms, all_idx, events)

    # ---- per root-child coherence -----------------------------------------
    per_root = []
    for rc in np.unique(root):
        idx = np.where(root == rc)[0]
        if len(idx) == 0:
            continue
        first_char = ctx[idx[0]][0] if ctx[idx[0]] else "?"
        entry = {
            "root_child_id": int(rc),
            "char": first_char,
            "units": int(len(idx)),
            "events": float(events[idx].sum()),
            "rho_unweighted": rho_from_gram(G, norms, idx) if len(idx) > 1 else 1.0,
            "rho_event_weighted": rho_from_gram(G, norms, idx, events[idx]) if len(idx) > 1 else 1.0,
        }
        if len(idx) > 1:
            sub = Gn[np.ix_(idx, idx)]
            off = sub[~np.eye(len(idx), dtype=bool)]
            entry["mean_pairwise_cos"] = float(off.mean())
        else:
            entry["mean_pairwise_cos"] = float("nan")
        per_root.append(entry)
    per_root.sort(key=lambda e: -e["events"])
    summary["per_root_child"] = per_root

    # Event-weighted average of per-root rho (what pd=1 aggregation "sees" on average)
    w = np.array([e["events"] for e in per_root])
    rw = np.array([e["rho_event_weighted"] for e in per_root])
    ru = np.array([e["rho_unweighted"] for e in per_root])
    summary["mean_root_rho_event_weighted_by_events"] = float((w * rw).sum() / w.sum())
    summary["mean_root_rho_unweighted_by_events"] = float((w * ru).sum() / w.sum())
    summary["median_root_rho_unweighted"] = float(np.median(ru))

    # ---- cosine vs trie distance -----------------------------------------
    same = root[:, None] == root[None, :]
    offdiag = ~np.eye(N, dtype=bool)
    within = Gn[same & offdiag]
    across = Gn[~same]
    summary["cos_within_root_mean"] = float(within.mean())
    summary["cos_within_root_median"] = float(np.median(within))
    summary["cos_across_root_mean"] = float(across.mean())
    summary["cos_across_root_median"] = float(np.median(across))
    summary["cos_all_pairs_frac_negative"] = float((Gn[offdiag] < 0).mean())

    # cosine bucketed by unit size (events), to separate noise from disagreement
    qs = np.quantile(events, [0.0, 0.25, 0.5, 0.75, 1.0])
    buckets = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        idx = np.where((events >= lo) & (events <= hi))[0]
        if len(idx) < 2:
            continue
        sub = Gn[np.ix_(idx, idx)]
        off = sub[~np.eye(len(idx), dtype=bool)]
        buckets.append({"events_min": float(lo), "events_max": float(hi), "units": int(len(idx)),
                        "mean_cos": float(off.mean()), "mean_norm": float(norms[idx].mean()),
                        "rho": rho_from_gram(G, norms, idx)})
    summary["cos_by_event_quartile"] = buckets

    # cosine of each unit with the event-weighted global mean direction
    mean_dir = np.einsum('ij,i->j', X, events, dtype=np.float64)
    mean_dir /= np.linalg.norm(mean_dir)
    cos_mean = np.einsum('ij,j->i', X, mean_dir, dtype=np.float64) / norms
    summary["cos_to_global_mean"] = {
        "mean": float(cos_mean.mean()), "median": float(np.median(cos_mean)),
        "frac_negative": float((cos_mean < 0).mean()),
        "event_weighted_mean": float((cos_mean * events).sum() / events.sum()),
    }

    # ---- covariance spectrum ---------------------------------------------
    # centered (unweighted) population; eigenvalues of the centered Gram
    # H G H (H = I - 11^T/N) equal the squared singular values of X - mean.
    H = np.eye(N) - np.full((N, N), 1.0 / N)
    Gc = H @ G @ H
    ev = np.linalg.eigvalsh(Gc)[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum()
    cum = np.cumsum(ev) / tot
    summary["spectrum"] = {
        "participation_ratio": float(tot ** 2 / (ev ** 2).sum()),
        "top1_frac": float(cum[0]),
        "top5_frac": float(cum[min(4, N - 1)]),
        "top10_frac": float(cum[min(9, N - 1)]),
        "top50_frac": float(cum[min(49, N - 1)]),
        "top100_frac": float(cum[min(99, N - 1)]),
        "rank_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "rank_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "top_eigs": [float(x) for x in ev[:10]],
    }
    # event-weighted, uncentered (rows scaled by sqrt(events)): Fisher-like second moment
    sw = np.sqrt(events)
    Gw = G * np.outer(sw, sw)
    evw = np.clip(np.linalg.eigvalsh(Gw)[::-1], 0, None)
    cumw = np.cumsum(evw) / evw.sum()
    summary["spectrum_event_weighted_uncentered"] = {
        "participation_ratio": float(evw.sum() ** 2 / (evw ** 2).sum()),
        "top1_frac": float(cumw[0]),
        "top10_frac": float(cumw[min(9, N - 1)]),
        "rank_for_90pct": int(np.searchsorted(cumw, 0.90) + 1),
    }

    # ---- per-section coherence -------------------------------------------
    sections = []
    for name, off, ln in layout["sections"]:
        S = X[:, off:off + ln].astype(np.float64)
        n = np.linalg.norm(S, axis=1)
        s_unw = np.linalg.norm(S.sum(axis=0)) / max(n.sum(), 1e-30)
        s_w = np.linalg.norm((S * events[:, None]).sum(axis=0)) / max((n * events).sum(), 1e-30)
        sections.append({"section": name, "floats": ln,
                         "norm_share": float((n ** 2).sum() / (norms ** 2).sum()),
                         "rho_unweighted": float(s_unw), "rho_event_weighted": float(s_w)})
    summary["sections"] = sections

    # ---- print ------------------------------------------------------------
    print(f"dump: {d}  units={N}  params={P}  pd={layout['partition_depth']}  events={events.sum():.0f}")
    print(f"global rho: unweighted={summary['rho_global_unweighted']:.4f}  event-weighted={summary['rho_global_event_weighted']:.4f}")
    print(f"per-root rho (event-weighted avg over roots): weighted={summary['mean_root_rho_event_weighted_by_events']:.4f}  "
          f"unweighted={summary['mean_root_rho_unweighted_by_events']:.4f}  median(unw)={summary['median_root_rho_unweighted']:.4f}")
    print(f"cosine within-root mean={summary['cos_within_root_mean']:.4f} median={summary['cos_within_root_median']:.4f} | "
          f"across-root mean={summary['cos_across_root_mean']:.4f} median={summary['cos_across_root_median']:.4f} | "
          f"frac pairs negative={summary['cos_all_pairs_frac_negative']:.3f}")
    cm = summary["cos_to_global_mean"]
    print(f"cos to global mean dir: mean={cm['mean']:.4f} median={cm['median']:.4f} event-weighted={cm['event_weighted_mean']:.4f} frac<0={cm['frac_negative']:.3f}")
    print("cos by event quartile:")
    for b in buckets:
        print(f"  events [{b['events_min']:.0f}, {b['events_max']:.0f}]  units={b['units']:4d}  mean_cos={b['mean_cos']:.4f}  rho={b['rho']:.4f}  mean_norm={b['mean_norm']:.3f}")
    sp = summary["spectrum"]
    print(f"spectrum (centered): PR={sp['participation_ratio']:.1f}  top1={sp['top1_frac']:.3f} top10={sp['top10_frac']:.3f} "
          f"top50={sp['top50_frac']:.3f} top100={sp['top100_frac']:.3f}  rank90={sp['rank_for_90pct']} rank99={sp['rank_for_99pct']}")
    spw = summary["spectrum_event_weighted_uncentered"]
    print(f"spectrum (event-weighted, uncentered): PR={spw['participation_ratio']:.1f}  top1={spw['top1_frac']:.3f} top10={spw['top10_frac']:.3f} rank90={spw['rank_for_90pct']}")
    print(f"top {args.top} root children by events:")
    print("  char  units   events        rho_w   rho_unw  mean_cos")
    for e in per_root[:args.top]:
        print(f"  {e['char']!r:5} {e['units']:5d} {e['events']:10.0f}  {e['rho_event_weighted']:.4f}  {e['rho_unweighted']:.4f}  {e['mean_pairwise_cos']:.4f}")
    lo = sorted([e for e in per_root if e["units"] >= 5], key=lambda e: e["rho_unweighted"])[:6]
    hi = sorted([e for e in per_root if e["units"] >= 5], key=lambda e: -e["rho_unweighted"])[:6]
    print("lowest-rho roots (>=5 units): " + ", ".join(f"{e['char']!r}={e['rho_unweighted']:.3f}" for e in lo))
    print("highest-rho roots (>=5 units): " + ", ".join(f"{e['char']!r}={e['rho_unweighted']:.3f}" for e in hi))
    print("sections (norm share / rho_unw / rho_w):")
    for s in sections:
        if s["norm_share"] >= 0.01:
            print(f"  {s['section']:14} {s['norm_share']:.3f}  {s['rho_unweighted']:.4f}  {s['rho_event_weighted']:.4f}")

    out = args.out or os.path.join(d, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
