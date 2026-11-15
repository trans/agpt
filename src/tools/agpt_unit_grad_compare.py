#!/usr/bin/env python3
"""Compare v2's per-unit gradients with a standard transformer's, unit by unit.

bin/agpt_train_v2 with AGPT_GRAD_DUMP_DIR set writes, at frozen weights, every
training unit's event-mean gradient (grads.f32, one row per unit), its loss
and event count (units.tsv) and the parameter layout (layout.json). This tool
recomputes each selected unit's loss and gradient with the PyTorch reference
model (agpt_ppl.AGPTModel, the model the minibatch trainer trains) over the
same events as plain windows, and reports, per unit and per parameter tensor,
the relative error |g_v2 - g_ref| / |g_ref| and the cosine. The reference is
float64; a float32 PyTorch gradient is compared to it too, as the rounding
floor of an ordinary fp32 implementation.

A unit is named in units.tsv by its anchor's token path (cut at the
partition depth k) and the anchor edge's depths [a, e]. Its events are the
windows whose first len(path) tokens are the path: a crown unit (e < k) at
depths a..e, a piece at depths a..T. Windows start at every character and
read past the end from the front, the cached tries' boundary mode
(tail-wrap-front-v1). Each unit's event count is checked against the dump.

Usage:
  python3 src/tools/agpt_unit_grad_compare.py --dump DIR --model CKPT.model \
      --corpus TRAIN.txt --vocab data/input.txt --out RESULT.json \
      [--rows all | --sample N] [--cache DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from agpt_ppl import AGPTModel, build_vocab, load_model  # noqa: E402

WINDOWS_PER_CHUNK = 2048


def sections(model, n_layers):
    """(name, tensor in v2's storage order) per layout.json section: Linear
    weights are stored (in, out) row-major, as the .model file holds them."""
    out = [("token_emb", model.tok_emb.weight)]
    for l, layer in enumerate(model.layers[:n_layers]):
        for w in ("wq", "wk", "wv", "wo"):
            lin = getattr(layer, w)
            out += [(f"l{l}.{w}_w", lin.weight), (f"l{l}.{w}_b", lin.bias)]
        out += [(f"l{l}.ln1_gamma", layer.ln1.weight), (f"l{l}.ln1_beta", layer.ln1.bias),
                (f"l{l}.l1_w", layer.l1.weight), (f"l{l}.l1_b", layer.l1.bias),
                (f"l{l}.l2_w", layer.l2.weight), (f"l{l}.l2_b", layer.l2.bias),
                (f"l{l}.ln2_gamma", layer.ln2.weight), (f"l{l}.ln2_beta", layer.ln2.bias)]
    out += [("final_gamma", model.final_ln.weight), ("final_beta", model.final_ln.bias),
            ("out_w", model.out_head.weight), ("out_b", model.out_head.bias)]
    return out


def flat_grad(secs):
    """Gradient per section in v2's layout (nn.Linear holds (out, in))."""
    parts = {}
    for name, p in secs:
        g = p.grad
        if name.endswith("_w") and name != "token_emb":
            g = g.t()
        parts[name] = g.detach().contiguous().reshape(-1).double().numpy().copy()
    return parts


def unit_reference(model, secs, ext, n, T, path, a, lmax, dtype):
    """Event-mean loss and gradient of one unit: windows starting with `path`,
    events at depths a..lmax."""
    P = len(path)
    pt = torch.tensor(path, dtype=torch.long)
    match = torch.ones(n, dtype=torch.bool)
    for d in range(P):
        match &= ext[d:d + n] == pt[d]
    W = torch.nonzero(match).squeeze(1)
    E = W.numel() * (lmax - a + 1)
    model.zero_grad(set_to_none=False)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    offs = torch.arange(lmax + 1).view(1, -1)
    mask = torch.zeros(lmax, dtype=dtype)
    mask[a - 1:lmax] = 1
    loss_sum = 0.0
    for j in range(0, W.numel(), WINDOWS_PER_CHUNK):
        ch = ext[W[j:j + WINDOWS_PER_CHUNK].view(-1, 1) + offs]
        logits = model(ch[:, :lmax])
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ch[:, 1:].reshape(-1),
                             reduction="none").view(-1, lmax)
        ls = (ce * mask).sum()
        (ls / E).backward()
        loss_sum += float(ls.detach())
    return loss_sum / E, flat_grad(secs), int(W.numel()), E


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rows", default=None, help="'all' (default) or comma-separated row numbers")
    g.add_argument("--sample", type=int, help="N rows: the 5 largest units, then a size-stratified random sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", help="directory for reference gradients, reused across calls")
    ap.add_argument("--no-fp32", action="store_true", help="skip the float32 PyTorch comparison")
    args = ap.parse_args()

    dump = Path(args.dump)
    lay = json.loads((dump / "layout.json").read_text())
    rows = list(csv.DictReader((dump / "units.tsv").open(), delimiter="\t"))
    P_all, k = lay["total_floats"], lay["partition_depth"]
    shape = lay["shape"]
    T = shape["seq_len"]
    grads = np.memmap(dump / "grads.f32", dtype=np.float32, mode="r", shape=(len(rows), P_all))
    sec_off = {name: (off, size) for name, off, size in lay["sections"]}

    if args.sample:
        ev = np.array([float(r["trained_events"]) for r in rows])
        big = list(np.argsort(-ev)[:5])
        rng = np.random.default_rng(args.seed)
        rest = np.setdiff1d(np.arange(len(rows)), big)
        q = np.quantile(np.log(ev[rest]), np.linspace(0, 1, 6))
        strata = np.clip(np.searchsorted(q, np.log(ev[rest]), side="right") - 1, 0, 4)
        per = max(1, (args.sample - len(big)) // 5)
        pick = []
        for s in range(5):
            cand = rest[strata == s]
            pick += list(rng.choice(cand, size=min(per, cand.size), replace=False))
        sel = sorted(int(x) for x in big + pick)
    elif args.rows and args.rows != "all":
        sel = [int(x) for x in args.rows.split(",")]
    else:
        sel = list(range(len(rows)))

    char_to_id, V = build_vocab(args.vocab)
    text = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    tok = torch.tensor([char_to_id.get(c, 0) for c in text], dtype=torch.long)
    n = tok.numel()
    ext = torch.cat([tok, tok[:T]])
    mc, sd = load_model(args.model)
    mc["seq_len"] = T
    ck_sha = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    models = {}
    for dt in ([torch.float64] if args.no_fp32 else [torch.float64, torch.float32]):
        m = AGPTModel(mc, sd, device="cpu").to(dt)
        models[dt] = (m, sections(m, mc["n_layers"]))
    secs64 = models[torch.float64][1]
    if sum(p.numel() for _, p in secs64) != P_all:
        sys.exit(f"model has {sum(p.numel() for _, p in secs64)} floats, layout {P_all}")
    for name, p in secs64:
        off, size = sec_off[name]
        if size != p.numel():
            sys.exit(f"section {name}: layout {size} floats, model {p.numel()}")
    cache = Path(args.cache) if args.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    def ref(path, a, lmax, dt):
        key = hashlib.sha1(f"{ck_sha}|{','.join(map(str, path))}|{a}|{lmax}|{dt}|wrap".encode()).hexdigest()
        if cache and (cache / f"{key}.npz").exists():
            z = np.load(cache / f"{key}.npz")
            return float(z["loss"]), {nm: z[nm] for nm, _ in secs64}, int(z["W"]), int(z["E"])
        m, secs = models[dt]
        loss, parts, W, E = unit_reference(m, secs, ext, n, T, path, a, lmax, dt)
        if cache:
            np.savez(cache / f"{key}.npz", loss=loss, W=W, E=E, **parts)
        return loss, parts, W, E

    def vec(parts):
        return np.concatenate([parts[nm] for nm, _ in secs64])

    def v2vec(r):
        row = np.asarray(grads[r], dtype=np.float64)
        return np.concatenate([row[sec_off[nm][0]:sec_off[nm][0] + sec_off[nm][1]] for nm, _ in secs64]), row

    res = {"dump": str(dump), "model": args.model, "model_sha256": ck_sha, "partition_depth": k,
           "units_in_dump": len(rows), "units_compared": len(sel), "layout": {kk: lay[kk] for kk in ("anc_grad",)},
           "units": []}
    names = [nm for nm, _ in secs64]
    full_v2 = full_ref = None
    ev_tot = 0.0
    t0 = time.time()
    for c, r in enumerate(sel):
        row = rows[r]
        path = [int(x) for x in row["context_tokens"].split(",")]
        a, e = int(row["anchor_depth"]), int(row["anchor_endpoint_depth"])
        lmax = e if e < k else T
        loss64, p64, W, E = ref(path, a, lmax, torch.float64)
        if E != round(float(row["trained_events"])):
            sys.exit(f"row {r}: {E} events here, {row['trained_events']} in the dump (path {path}, a={a}, e={e})")
        g_ref = vec(p64)
        g_v2, raw = v2vec(r)
        if abs(np.linalg.norm(raw) - float(row["grad_l2"])) > 1e-4 * float(row["grad_l2"]) + 1e-12:
            sys.exit(f"row {r}: grads.f32 norm {np.linalg.norm(raw)} != units.tsv grad_l2 {row['grad_l2']}")
        nr = np.linalg.norm(g_ref)
        u = {"row": r, "path": path, "anchor_depth": a, "endpoint_depth": e, "windows": W, "events": E,
             "loss_v2": float(row["mean_loss"]), "loss_ref": loss64, "grad_norm_ref": float(nr),
             "rel_err_v2": float(np.linalg.norm(g_v2 - g_ref) / nr),
             "cos_v2": float(g_v2 @ g_ref / (np.linalg.norm(g_v2) * nr))}
        per = {}
        o = 0
        for nm in names:
            sz = p64[nm].size
            gr, gv = g_ref[o:o + sz], g_v2[o:o + sz]
            per[nm] = float(np.linalg.norm(gv - gr) / max(np.linalg.norm(gr), 1e-300))
            o += sz
        u["rel_err_v2_by_tensor"] = per
        if torch.float32 in models:
            loss32, p32, _, _ = ref(path, a, lmax, torch.float32)
            g32 = vec(p32)
            u["rel_err_fp32"] = float(np.linalg.norm(g32 - g_ref) / nr)
            u["cos_fp32"] = float(g32 @ g_ref / (np.linalg.norm(g32) * nr))
            u["loss_fp32"] = loss32
        res["units"].append(u)
        w = float(E)
        full_v2 = g_v2 * w if full_v2 is None else full_v2 + g_v2 * w
        full_ref = g_ref * w if full_ref is None else full_ref + g_ref * w
        ev_tot += w
        worst = max(per, key=per.get)
        print(f"[{c + 1}/{len(sel)} {time.time() - t0:.0f}s] row {r} path {path} a={a} e={e} events {E}: "
              f"v2 rel err {u['rel_err_v2']:.2e} cos {u['cos_v2']:.8f}"
              + (f" | fp32 rel err {u['rel_err_fp32']:.2e}" if "rel_err_fp32" in u else "")
              + f" | loss v2 {u['loss_v2']:.6f} ref {loss64:.6f} | worst tensor {worst} {per[worst]:.2e}", flush=True)
    if len(sel) == len(rows):
        fv, fr = full_v2 / ev_tot, full_ref / ev_tot
        res["full_pass"] = {"rel_err_v2": float(np.linalg.norm(fv - fr) / np.linalg.norm(fr)),
                            "cos_v2": float(fv @ fr / (np.linalg.norm(fv) * np.linalg.norm(fr))),
                            "grad_norm_ref": float(np.linalg.norm(fr))}
        print(f"full pass (event-weighted over all {len(rows)} units): v2 rel err "
              f"{res['full_pass']['rel_err_v2']:.2e} cos {res['full_pass']['cos_v2']:.8f}", flush=True)
    re_v2 = np.array([u["rel_err_v2"] for u in res["units"]])
    res["summary"] = {"rel_err_v2_median": float(np.median(re_v2)), "rel_err_v2_max": float(re_v2.max()),
                      "cos_v2_min": float(min(u["cos_v2"] for u in res["units"]))}
    if torch.float32 in models:
        re32 = np.array([u["rel_err_fp32"] for u in res["units"]])
        res["summary"].update(rel_err_fp32_median=float(np.median(re32)), rel_err_fp32_max=float(re32.max()))
    print("summary:", json.dumps(res["summary"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
