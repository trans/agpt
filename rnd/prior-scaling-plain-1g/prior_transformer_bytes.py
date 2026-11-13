"""Does a count prior's gain shrink as the transformer grows?

The character transformer of imm's scripts/transformer_baseline.py, with
mobius's prior rows added to its output logits:

    logits = head(h) + g * prior_log_probs
    g      = softplus(w . [h, gate_features] + b)

`g` starts at 1 (w = 0), so training starts from the prior at full strength,
as imm's does; the network learns where to trust it. mobius's rows are
cross-fitted: a training row's prior did not count the text it predicts.

With --prior none, the run is transformer_baseline.py's exactly: the same
model init, batches and dropout draws.

Training, scoring and the result file follow transformer_baseline.py (best
and last held-out step both reported; the held-out text is the only test set).
At every held-out evaluation the same weights are also scored with the prior
taken away (ablated_heldout_ppl; see heldout_prior): how much the network has
learned on its own, and whether it now does better without the prior.

    python prior_transformer.py --config <imm config>.json --output <out>.json \
        --prior /home/trans/Projects/air/mobius/runs/imm/<rows prefix> [model args]
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMM = Path("/home/trans/Projects/air/imm")  # agpt copy: air/imm by absolute path
sys.path.insert(0, str(IMM))
sys.path.insert(0, str(IMM / "scripts"))

import transformer_baseline as tb  # noqa: E402
from imm.data import CharVocab, read_text  # noqa: E402


class PriorRows:
    """mobius_prior_rows_v2: row r holds natural-log probabilities over the
    vocabulary for ids[r + 1] given ids[0..r] (float16), and the gate features
    of the longest context ending at r (float32)."""

    def __init__(self, prefix: str, split: str, vocab: int, n_text: int) -> None:
        lp_path = Path(f"{prefix}.{split}_log_probs.dat")
        gf_path = Path(f"{prefix}.{split}_gate_features.dat")
        n = n_text - 1
        if lp_path.stat().st_size != n * vocab * 2:
            raise SystemExit(f"{lp_path}: {lp_path.stat().st_size} bytes, expected {n} rows x {vocab} float16")
        nf, rem = divmod(gf_path.stat().st_size, n * 4)
        if rem:
            raise SystemExit(f"{gf_path}: size is not {n} rows of float32")
        self.lp = np.memmap(lp_path, dtype=np.float16, mode="r", shape=(n, vocab))
        self.gf = np.memmap(gf_path, dtype=np.float32, mode="r", shape=(n, nf))
        self.n, self.features = n, nf
        # A file still being built can be the right size and all zeros (sparse):
        # every row must be a distribution.
        probe = np.asarray(self.lp[:: max(1, n // 2000)], dtype=np.float32)
        lse = np.log(np.exp(probe).sum(1))
        if not np.all(np.abs(lse) < 0.05):
            raise SystemExit(f"{lp_path}: rows do not sum to 1 (logsumexp up to {np.abs(lse).max():.3f}); not built yet?")
        self.mean = np.zeros(nf, dtype=np.float32)
        self.std = np.ones(nf, dtype=np.float32)

    def standardize_from(self, other: "PriorRows") -> None:
        every = max(1, other.n // 50000)
        sample = np.asarray(other.gf[::every], dtype=np.float32)
        mean, std = sample.mean(0), sample.std(0)
        const = std < 1e-6  # the bias column (and any other constant) passes through
        self.mean = np.where(const, 0.0, mean).astype(np.float32)
        self.std = np.where(const, 1.0, std).astype(np.float32)

    def window(self, starts, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        lp = np.stack([self.lp[s : s + length] for s in starts]).astype(np.float32)
        gf = (np.stack([self.gf[s : s + length] for s in starts]) - self.mean) / self.std
        return torch.from_numpy(np.maximum(lp, -30.0)), torch.from_numpy(gf.astype(np.float32))

    def nll(self, ids: torch.Tensor) -> float:
        """Mean natural-log loss of the prior alone on targets 1..n-1."""
        tot = 0.0
        for a in range(0, self.n, 1 << 20):
            b = min(self.n, a + (1 << 20))
            lp = np.asarray(self.lp[a:b], dtype=np.float32)
            tot -= float(lp[np.arange(b - a), ids[a + 1 : b + 1].numpy()].sum())
        return tot / self.n


class PriorGPT(nn.Module):
    def __init__(self, base: tb.CharGPT, features: int, prior_input: bool = False) -> None:
        super().__init__()
        self.base = base
        width = base.ln.weight.shape[0]
        self.gate = nn.Linear(width + features, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(math.e - 1.0))  # softplus(b) = 1
        # --prior-input: the network also sees the prior's prediction at every
        # position, as imm's residual does (--prior-input log-probs): the log
        # probabilities (clamped at -18, as imm) and the gate features, projected
        # into the input. Zero at the start, so step 0 is the output-only model.
        self.inp = None
        if prior_input:
            self.inp = nn.Linear(base.tok.num_embeddings + features, width, bias=False)
            nn.init.zeros_(self.inp.weight)
        self.block = base.block

    def hidden(self, ids: torch.Tensor, lp: torch.Tensor, gf: torch.Tensor, use_input: bool = True,
               keep: torch.Tensor | None = None) -> torch.Tensor:
        """The final hidden state h (after the last layer norm). keep (one 0/1
        per sequence) takes the prior out of the input where 0 (--prior-dropout)."""
        b = self.base
        t = ids.shape[1]
        e = b.tok(ids) + b.pos(torch.arange(t, device=ids.device))
        if self.inp is not None and use_input:
            pin = self.inp(torch.cat([lp.clamp(min=-18.0), gf], dim=-1))
            e = e + (pin if keep is None else pin * keep.to(pin.dtype)[:, None, None])
        x = b.drop(e)
        for blk in b.blocks:
            x = blk(x)
        return b.ln(x)

    def forward(self, ids: torch.Tensor, lp: torch.Tensor, gf: torch.Tensor,
                use_input: bool = True, keep: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits with the prior, gate, the network's own logits head(h)).
        use_input=False leaves the prior out of the input (an ablation; training
        always has it in when --prior-input). keep (one 0/1 per sequence, training
        with --prior-dropout) takes the prior out of both input and output where 0."""
        b = self.base
        h = self.hidden(ids, lp, gf, use_input, keep)
        g = F.softplus(self.gate(torch.cat([h.float(), gf], dim=-1)).float())
        own = b.head(h).float()
        prior = g * lp if keep is None else g * lp * keep.float()[:, None, None]
        return own + prior, g, own


@torch.no_grad()
def heldout_prior(model: PriorGPT, ids: torch.Tensor, rows: PriorRows, batch: int, device, dtype
                  ) -> tuple[float, int, float, dict[str, float]]:
    """Held-out loss of the trained model, and of the same weights with the prior
    taken away (the ablations, each a total like the first):
      own          the network's own logits head(h), the prior out of the output
                   (and, for --prior-input, still in the input);
      in_only      --prior-input only: the prior in the input, not the output (= own);
      out_only     --prior-input only: the prior in the output, not the input;
      neither      --prior-input only: the prior in neither (no prior at all)."""
    model.eval()
    total, count, gsum = 0.0, 0, 0.0
    abl: dict[str, float] = {"own": 0.0}
    if model.inp is not None:
        abl.update(out_only=0.0, neither=0.0)
    by_length: dict[int, list] = {}
    for w in tb.score_windows(len(ids), model.block):
        by_length.setdefault(w[2] - 1 - w[0], []).append(w)

    def nll(logits, y):
        return F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")

    for length, group in by_length.items():
        for i in range(0, len(group), batch):
            part = group[i : i + batch]
            x = torch.stack([ids[s : s + length] for s, _, _ in part]).to(device)
            y = torch.stack([ids[s + 1 : s + 1 + length] for s, _, _ in part]).to(device)
            lp, gf = rows.window([s for s, _, _ in part], length)
            lp, gf = lp.to(device), gf.to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits, g, own = model(x, lp, gf)
                losses = {"full": nll(logits, y), "own": nll(own, y)}
                if model.inp is not None:
                    logits0, _, own0 = model(x, lp, gf, use_input=False)
                    losses.update(out_only=nll(logits0, y), neither=nll(own0, y))
            for r, (s, first, end) in enumerate(part):
                total += float(losses["full"][r, first - s - 1 :].sum())
                for k in abl:
                    abl[k] += float(losses[k][r, first - s - 1 :].sum())
                gsum += float(g[r, first - s - 1 :, 0].sum())
                count += end - first
    model.train()
    return total, count, gsum / max(1, count), abl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior", default="none", help="mobius rows prefix, or none")
    p.add_argument("--prior-input", action="store_true",
                   help="also feed the prior's log-probs and gate features into the input, as imm does")
    p.add_argument("--prior-dropout", type=float, default=0.0,
                   help="in training, take the prior away (input and output) from this fraction of the "
                        "sequences, so the network must also learn the text itself; scoring always has it")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=Path, default=None, help="write the final weights here (torch.save)")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    vocab = CharVocab.from_text(read_text(IMM / cfg["vocab_input"]))
    # agpt copy: the training text is held as one byte per character (ids < 256), not a
    # Python list and an int64 tensor (about 16 GB at 1G characters); batches are
    # widened to int64 below, so their values are those of the original.
    lut = np.full(256, 255, dtype=np.uint8)
    for ch in vocab.chars:
        lut[ord(ch)] = vocab.stoi[ch]
    raw = np.fromfile(IMM / cfg["train_input"], dtype=np.uint8)
    train = torch.from_numpy(lut[raw])
    del raw
    if int(train.max()) >= vocab.size:
        raise SystemExit("training text has a character outside the vocabulary")
    held_text = read_text(IMM / cfg["eval_input"])
    if cfg.get("eval_max_chars"):
        held_text = held_text[: cfg["eval_max_chars"]]
    held = torch.tensor(vocab.encode(held_text), dtype=torch.long)

    use_prior = args.prior != "none"
    info: dict = {}
    if use_prior:
        tr_rows = PriorRows(args.prior, "train", vocab.size, len(train))
        ev_rows = PriorRows(args.prior, "eval", vocab.size, len(held))
        tr_rows.standardize_from(tr_rows)
        ev_rows.mean, ev_rows.std = tr_rows.mean, tr_rows.std
        meta = Path(f"{args.prior}.json")
        if meta.exists():
            chars = json.loads(meta.read_text()).get("vocab_chars")
            if chars is not None and tuple(chars) != vocab.chars:
                raise SystemExit("prior rows' vocabulary differs from the config's")
        info["prior_only_heldout_ppl"] = math.exp(ev_rows.nll(held))
        info["prior_gate_features"] = tr_rows.features
        print(f"prior alone: heldout_ppl={info['prior_only_heldout_ppl']:.4f} "
              f"bits/char={math.log2(info['prior_only_heldout_ppl']):.4f}", flush=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    base = tb.CharGPT(vocab.size, args.block, args.layers, args.heads, args.width, args.dropout)
    model = (PriorGPT(base, tr_rows.features, args.prior_input) if use_prior else base).to(device)
    n_base = sum(q.numel() for q in base.parameters())
    n_params = sum(q.numel() for q in model.parameters())
    gate_params = list(model.gate.parameters()) if use_prior else []
    gate_ids = {id(q) for q in gate_params}
    decay = [q for q in model.parameters() if q.dim() >= 2 and id(q) not in gate_ids]
    no_decay = [q for q in model.parameters() if q.dim() < 2 or id(q) in gate_ids]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / (args.warmup + 1)
        frac = min(1.0, (step - args.warmup) / max(1, args.iters - args.warmup))
        return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * frac)) * (args.lr - args.min_lr)

    # Batches come from a thread, in the baseline's order (one generator, used only here).
    gen = torch.Generator().manual_seed(args.seed)
    batches: queue.Queue = queue.Queue(maxsize=8)

    # --prior-dropout draws come from their own generator, so the batches stay
    # those of the runs without it.
    mgen = torch.Generator().manual_seed(args.seed + 1)

    def produce() -> None:
        for _ in range(args.iters):
            starts = torch.randint(0, len(train) - args.block - 1, (args.batch,), generator=gen)
            x = torch.stack([train[s : s + args.block] for s in starts]).long()
            y = torch.stack([train[s + 1 : s + 1 + args.block] for s in starts]).long()
            extra = tr_rows.window(starts.tolist(), args.block) if use_prior else None
            if use_prior and args.prior_dropout > 0:
                keep = (torch.rand(args.batch, generator=mgen) >= args.prior_dropout).float()
                extra = (*extra, keep)
            batches.put((x, y, extra))

    threading.Thread(target=produce, daemon=True).start()

    print(f"params={n_params} (transformer {n_base}) train_chars={len(train)} heldout_chars={len(held)} "
          f"device={device} dtype={dtype} prior={args.prior}", flush=True)
    curve, started = [], time.perf_counter()
    for step in range(args.iters + 1):
        if step % args.eval_every == 0 or step == args.iters:
            abl = None
            if use_prior:
                total, count, gmean, abl = heldout_prior(model, held, ev_rows, args.batch, device, dtype)
            else:
                (total, count), gmean = tb.heldout_nll(model, held, args.batch, device, dtype), None
            curve.append({"step": step, "heldout_ppl": math.exp(total / count), "mean_gate": gmean,
                          "sec": time.perf_counter() - started})
            gtxt = f" mean_gate={gmean:.3f}" if gmean is not None else ""
            if abl is not None:
                curve[-1]["ablated_heldout_ppl"] = {k: math.exp(v / count) for k, v in abl.items()}
                gtxt += "".join(f" {k}={v:.4f}" for k, v in curve[-1]["ablated_heldout_ppl"].items())
            print(f"step={step} heldout_ppl={curve[-1]['heldout_ppl']:.4f}{gtxt} scored={count} "
                  f"sec={curve[-1]['sec']:.0f}", flush=True)
        if step == args.iters:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y, extra = batches.get()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            if use_prior:
                keep = extra[2].to(device) if len(extra) > 2 else None
                logits, _, _ = model(x, extra[0].to(device), extra[1].to(device), keep=keep)
            else:
                logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, vocab.size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step={step} train_loss={loss.item():.4f} lr={lr_at(step):.2e}", flush=True)

    best = min(curve, key=lambda c: c["heldout_ppl"])
    commit = subprocess.run(["git", "-C", str(IMM), "log", "-1", "--format=%h", "--", "scripts/transformer_baseline.py"],
                            capture_output=True, text=True).stdout.strip()
    result = {
        "model": "char transformer (imm transformer_baseline.CharGPT) + gated count prior on the logits"
                 + (", and the prior's log-probs and features in its input" if args.prior_input else "")
                 + (f"; in training the prior is taken away from {args.prior_dropout:.0%} of the sequences"
                    if args.prior_dropout > 0 else ""),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "transformer_baseline_commit": commit,
        "params": n_params,
        "transformer_params": n_base,
        "split": {"train_input": cfg["train_input"], "eval_input": cfg["eval_input"], "eval_chars": len(held)},
        **info,
        "heldout_ppl": curve[-1]["heldout_ppl"],
        "heldout_bits_per_char": math.log2(curve[-1]["heldout_ppl"]),
        "best_heldout_ppl": best["heldout_ppl"],
        "best_step": best["step"],
        "protocol": "every held-out character after the first, up to block characters of context",
        "curve": curve,
    }
    if use_prior:
        result["ablated_heldout_ppl"] = curve[-1]["ablated_heldout_ppl"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": result["config"],
                    "gate_feature_mean": tr_rows.mean.tolist() if use_prior else None,
                    "gate_feature_std": tr_rows.std.tolist() if use_prior else None}, args.save)
    print(f"result: heldout_ppl={result['heldout_ppl']:.4f} best={best['heldout_ppl']:.4f}@{best['step']} -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
