#!/usr/bin/env python3
"""Dynamic evaluation of mobius's prior + stream network on its 50 whole held-out books.

agpt's dynamic evaluation (rnd/dynamic-eval: score a block, take one optimizer
step on the block just scored, go on; reset at each document) applied to
mobius's 10.8M prior-code network (or another checkpoint of the same trainer),
over the books of mobius's data/pg19/imm-split-20m-books with their book-length
stream rows. Agreed with @mobius (agpt .ai/mail/received/20261123-161956Z-*,
20261126-185534Z-*).

Protocol. Each book is scored exactly as mobius's score_books.py scores it:
imm's transformer_baseline.score_windows over the book (windows of `block`
characters at a stride of block / 2, each target scored once, from the window
where it has the most context; windows never cross books), the rows of the
window from the rows prefix, the checkpoint's gate standardisation. A static
setting reproduces score_books.py. An adaptive setting walks the same windows
in order: score the window's targets with the current weights, then take one
optimizer step on the mean NLL of those same targets (text already scored), so
the scored positions are the static ones, one for one. The model stays in
eval mode throughout (no dropout, no prior dropout). Weights and optimizer
state are reset to the checkpoint at every book.

Settings (settings=, comma-separated):
  static
  full-adam-LR   the transformer adapts (embeddings, position table, blocks, final
                 norm; the head is tied to the embeddings); the prior's input layer
                 `inp` and the output `gate` stay frozen (mobius: they price the
                 prior, and the stream rows already adapt to the book)
  all-adam-LR    every parameter adapts, inp and gate too
  loraR-adam-LR  a rank-R delta B @ A on every block's qkv, proj and both MLP maps
                 (B starts at zero, A at N(0, 1/in) from a fixed seed); nothing else
                 adapts
  (sgd in place of adam: plain SGD)
  ...-dLAMBDA    after each step, decay the adapting weights toward the checkpoint's:
                 w += LAMBDA * (w0 - w) (Krause et al. 2018's decay, without their RMS
                 step scaling), e.g. full-adam-1e-4-d0.01

Control (past=other): the same steps, but book i's k-th step trains on the k-th
window of the other books' text (books i+1, i+2, ... in order, each walked by its
own windows with its own rows), so the weights see as much held-out text as
with past=own, none of it this book's. Own gain minus the control's gain is
what the book's own past is worth.

Output: OUT.npy, float32 nats per held-out character in the layout of
score_books.py's (nan where not scored: the books' first characters and the
books this run did not score), and OUT.json (setting, books, timings,
provenance). summarize.py pairs them. Diagnostic: mobius's protocol, not
agpt's canonical lm-eval.

    python3 -I dyneval_prior.py CHECKPOINT OUT_PREFIX setting=S [past=own|other] [books=tune|report|all|A-B]
        [shard=I/K] [rows=PREFIX] [device=cuda] [limit=N]
  (limit=N: only the first N windows of each book, for sizing and checks)
"""
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

MOBIUS = Path("/home/trans/Projects/air/mobius")
FROZEN = MOBIUS / "runs/frozen"
SPLIT = MOBIUS / "data/pg19/imm-split-20m-books"
ROWS = MOBIUS / "runs/imm/pg19split20m_books_stream_d24_pg10g"
sys.path.insert(0, str(FROZEN))
import prior_input_only as pt  # noqa: E402  (imports imm's scripts/transformer_baseline.py and imm/data.py)

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git_head(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


class LowRank(nn.Module):
    def __init__(self, out_f, in_f, r, gen):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, in_f, generator=gen) / math.sqrt(in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, W):
        return W + (self.B @ self.A).to(W.dtype)


def parse_setting(s):
    if s == "static":
        return {"name": s, "kind": "static"}
    m = re.fullmatch(r"(\w+?)-(adam|sgd)-([0-9.e+-]+?)(?:-d([0-9.e+-]+))?", s)
    if not m:
        raise SystemExit(f"unknown setting {s!r}")
    kind, opt, lr, decay = m.group(1), m.group(2), float(m.group(3)), float(m.group(4) or 0)
    if kind in ("full", "all"):
        return {"name": s, "kind": kind, "opt": opt, "lr": lr, "decay": decay}
    if kind.startswith("lora") and kind[4:].isdigit():
        return {"name": s, "kind": "lora", "rank": int(kind[4:]), "opt": opt, "lr": lr, "decay": decay}
    raise SystemExit(f"unknown setting {s!r}")


def main():
    ck_path, out_prefix = sys.argv[1:3]
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    st = parse_setting(opts.get("setting", "static"))
    past = opts.get("past", "own")
    if past not in ("own", "other"):
        raise SystemExit("past= must be own or other")
    if past == "other" and st["kind"] == "static":
        raise SystemExit("past=other needs an adaptive setting")
    device = torch.device(opts.get("device", "cuda"))
    rows_prefix = opts.get("rows", str(ROWS))
    limit = int(opts["limit"]) if "limit" in opts else None

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    cfg = json.loads(Path(a["config"]).read_text())
    vocab = pt.CharVocab.from_text(pt.read_text(pt.IMM / cfg["vocab_input"]))
    base = pt.tb.CharGPT(vocab.size, a["block"], a["layers"], a["heads"], a["width"], a["dropout"])
    # vocab.encode through a byte table (the text is ASCII): the same ids without an 18.7M-item list
    raw = np.frombuffer((SPLIT / "heldout_corpus.txt").read_bytes(), dtype=np.uint8)
    lut = np.full(256, -1, dtype=np.int64)
    for ch, i in vocab.stoi.items():
        if len(ch.encode()) == 1:
            lut[ch.encode()[0]] = i
    held = torch.from_numpy(lut[raw])
    if (held < 0).any():
        raise SystemExit("held-out text has a byte outside the vocabulary")
    del raw
    n = len(held)
    manifest = json.loads((SPLIT / "manifest.json").read_text())
    sizes = np.array([r["size"] for r in manifest["chunk_records"]])
    assert sizes.sum() == n
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    nb = len(sizes)
    rows = pt.PriorRows(rows_prefix, "eval", vocab.size, n)
    rows.mean = np.array(ck["gate_feature_mean"], dtype=np.float32)
    rows.std = np.array(ck["gate_feature_std"], dtype=np.float32)
    model = pt.PriorGPT(base, rows.features, a["prior_input"], a["input_only"], a["output_mix"], a["prior_input_form"],
                        None, a["prior_input_k"], a["gate_reads_code"])
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    block = model.block
    dtype = torch.bfloat16 if device.type == "cuda" else None

    # which books: tune = even indices, report = odd (the draw's groups spread over both)
    sel = opts.get("books", "all")
    if sel == "tune":
        books = list(range(0, nb, 2))
    elif sel == "report":
        books = list(range(1, nb, 2))
    elif sel == "all":
        books = list(range(nb))
    else:
        b0, b1 = (int(x) for x in sel.split("-"))
        books = list(range(b0, b1))
    if "shard" in opts:
        i, k = (int(x) for x in opts["shard"].split("/"))
        books = books[i::k]

    def windows(bi):
        a0 = int(bounds[bi])
        return [(s + a0, first + a0, end + a0) for s, first, end in pt.tb.score_windows(int(sizes[bi]), block)]

    # what adapts
    params = []
    if st["kind"] == "full":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.base.parameters():
            p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    elif st["kind"] == "all":
        params = list(model.parameters())
    elif st["kind"] == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        gen = torch.Generator().manual_seed(0)
        for blk in model.base.blocks:
            for lin in (blk.qkv, blk.proj, blk.mlp[0], blk.mlp[2]):
                o, i_ = lin.weight.shape
                P.register_parametrization(lin, "weight", LowRank(o, i_, st["rank"], gen).to(device))
                params += [lin.parametrizations.weight[0].A, lin.parametrizations.weight[0].B]
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    init = [p.detach().clone() for p in params]
    n_adapt = sum(p.numel() for p in params)

    def forward(w, grad):
        s, first, end = w
        length = end - 1 - s
        x = held[s : s + length].unsqueeze(0).to(device)
        y = held[s + 1 : s + 1 + length].unsqueeze(0).to(device)
        lp, gf = rows.window([s], length)
        with torch.set_grad_enabled(grad), torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits, _, _ = model(x, lp.to(device), gf.to(device))
        nll = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")[0]
        return nll[first - s - 1 : end - s - 1]

    say(f"{Path(ck_path).name} (block {block}, {a['layers']}x{a['width']}) setting {st['name']} past {past}; "
        f"{len(books)} books; {n_adapt:,} parameters adapt; {device}")
    loss = np.full(n, np.nan, dtype=np.float32)
    steps = 0
    per_book = {}
    for bi in books:
        tb0 = time.time()
        with torch.no_grad():
            for p, q in zip(params, init):
                p.copy_(q)
        opt = None
        if params:
            opt = (torch.optim.Adam(params, lr=st["lr"]) if st["opt"] == "adam" else torch.optim.SGD(params, lr=st["lr"]))
        wins = windows(bi)
        if limit:
            wins = wins[:limit]
        ctrl = []
        if past == "other":
            j = bi
            while len(ctrl) < len(wins):
                j = (j + 1) % nb
                assert j != bi
                ctrl += windows(j)
            ctrl = ctrl[: len(wins)]
        for k, w in enumerate(wins):
            if opt is None:
                ev = forward(w, False)
                loss[w[1] : w[2]] = ev.cpu().numpy()
                continue
            if past == "own":
                ev = forward(w, True)
                loss[w[1] : w[2]] = ev.detach().cpu().numpy()   # scored before the step below
            else:
                loss[w[1] : w[2]] = forward(w, False).cpu().numpy()
                ev = forward(ctrl[k], True)
            opt.zero_grad(set_to_none=True)
            ev.mean().backward()
            opt.step()
            if st["decay"]:
                with torch.no_grad():
                    for p, q in zip(params, init):
                        p.lerp_(q, st["decay"])
            steps += 1
        got = loss[wins[0][1] : wins[-1][2]]
        per_book[bi] = {"windows": len(wins), "seconds": round(time.time() - tb0, 2),
                        "bits": float(np.nanmean(got) / math.log(2))}
        say(f"  book {bi}: {len(wins)} windows, {per_book[bi]['bits']:.4f} bits/char, {per_book[bi]['seconds']:.1f} s; "
            f"peak host {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB"
            + (f", GPU {torch.cuda.max_memory_allocated() // 2**20} MB" if device.type == "cuda" else ""))
        if not np.isfinite(per_book[bi]["bits"]):
            raise SystemExit(f"book {bi}: non-finite loss; stopped")
    np.save(f"{out_prefix}.npy", loss)
    scored = ~np.isnan(loss)
    bits = float(loss[scored].mean() / math.log(2))
    res = {
        "checkpoint": str(Path(ck_path).resolve()), "checkpoint_sha256": sha256(ck_path),
        "setting": st, "past": past, "books": books, "limit": limit, "rows": rows_prefix, "split": str(SPLIT),
        "block": block, "adapting_parameters": n_adapt, "steps": steps, "scored": int(scored.sum()), "bits": bits,
        "per_book": per_book, "seconds": round(time.time() - t0, 1), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_host_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else None,
        "torch": torch.__version__, "argv": sys.argv,
        "code": {"dyneval_prior.py": sha256(__file__), "prior_input_only.py": sha256(FROZEN / "prior_input_only.py"),
                 "transformer_baseline.py": sha256(pt.IMM / "scripts/transformer_baseline.py"),
                 "imm/data.py": sha256(pt.IMM / "imm/data.py"),
                 "agpt": git_head(Path(__file__).resolve().parents[2]), "imm": git_head(pt.IMM), "mobius": git_head(MOBIUS)},
    }
    Path(f"{out_prefix}.json").write_text(json.dumps(res, indent=1) + "\n")
    say(f"{st['name']} past {past}: {bits:.4f} bits/char over {int(scored.sum())} characters, {steps} steps, "
        f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
