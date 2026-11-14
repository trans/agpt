#!/usr/bin/env python3
"""Ordinary minibatch Adam training of the AGPT attention model, as a custom
trainer for bin/agpt_experiment.

The model is the one the AGPT trainers use (RoPE + MHA + LN + FFN, the
`agpt_ppl.AGPTModel` reference). It is trained the conventional way, with
no trie and no gradient aggregation: every window start of the training
corpus once per epoch, in a fresh random order, `seq_len` targets per
window, mean cross-entropy, one Adam step per minibatch. A depth-16 trie
holds the same events (every prefix of length 1..16 with its next
character), so one epoch here and one trie traversal cover the same
training objective; they differ in how many optimizer steps they take.

Ported from the June baseline trainer (`agpt_vanilla_attn_train.py`, tag
`exp/linear-recurrence-final`), changed to read the orchestrator's
resolved YAML, to use agpt_train_v2's learning-rate schedule (per
optimizer step: linear warmup over `warmup_epochs`, then cosine down to
`min_lr_ratio`), and to write epoch checkpoints the orchestrator evaluates.

Usage (through the orchestrator):

    bin/agpt_experiment --config FOO.yml --trainer src/tools/agpt_minibatch_train.py

Config keys read:
    corpus.path, corpus.vocab_source
    model.init_file  (or d_model/n_layers/n_heads/d_ff with init_seed), model.save_file
    train.budget {unit: epochs, value}, train.seed, train.seq_len, train.checkpoint_epochs
    train.optimizer {name: adam, lr, beta (2nd moment), momentum_beta (1st), eps}
    train.lr_schedule {name: constant | warmup-cosine, warmup_epochs, min_lr_ratio}
    experimental.minibatch_size  (windows per step, default 128)
    experimental.micro_batch     (windows per forward/backward, default minibatch_size;
                                  smaller values accumulate the same mean gradient)
    experimental.device          (default: cuda if available)
    experimental.max_corpus_chars (smoke tests only: train on a prefix)
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from agpt_ppl import AGPTModel, MGPT_MAGIC, build_vocab, load_model  # noqa: E402

KNOWN_EXPERIMENTAL = {"minibatch_size", "micro_batch", "device", "max_corpus_chars"}


def random_state_dict(d_model, n_heads, n_layers, d_ff, vocab_size, seed):
    """N(0, 0.02) weights, zero biases, unit LayerNorm gains (the June trainer's init)."""
    g = torch.Generator().manual_seed(seed)
    D, Fd, V = d_model, d_ff, vocab_size

    def randn(*shape):
        return torch.randn(*shape, generator=g) * 0.02

    sd = {"token_emb": randn(V, D)}
    for l in range(n_layers):
        for w in ("wq", "wk", "wv", "wo"):
            sd[f"l{l}.{w}_w"] = randn(D, D)
            sd[f"l{l}.{w}_b"] = torch.zeros(1, D)
        sd[f"l{l}.ln1_g"] = torch.ones(1, D)
        sd[f"l{l}.ln1_b"] = torch.zeros(1, D)
        sd[f"l{l}.l1_w"] = randn(D, Fd)
        sd[f"l{l}.l1_b"] = torch.zeros(1, Fd)
        sd[f"l{l}.l2_w"] = randn(Fd, D)
        sd[f"l{l}.l2_b"] = torch.zeros(1, D)
        sd[f"l{l}.ln2_g"] = torch.ones(1, D)
        sd[f"l{l}.ln2_b"] = torch.zeros(1, D)
    sd["final_g"] = torch.ones(1, D)
    sd["final_b"] = torch.zeros(1, D)
    sd["out_w"] = randn(D, V)
    sd["out_b"] = torch.zeros(1, V)
    return sd


def save_model(path, cfg, model):
    """Write the MGPT .model format that agpt_ppl.load_model and agpt_hf.py read.
    .model stores Linear weights as (in, out); nn.Linear holds (out, in)."""
    D, Fd, V, L, H, S = (cfg["d_model"], cfg["d_ff"], cfg["vocab_size"],
                         cfg["n_layers"], cfg["n_heads"], cfg["seq_len"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def mat(f, t, rows, cols):
        t = t.detach().float().cpu().contiguous()
        assert t.shape == (rows, cols), f"shape {tuple(t.shape)} != ({rows}, {cols})"
        f.write(struct.pack("<2i", rows, cols))
        f.write(t.numpy().astype("<f4").tobytes())

    with open(path, "wb") as f:
        f.write(struct.pack("<I", MGPT_MAGIC))
        f.write(struct.pack("<6i", D, H, L, Fd, V, S))
        mat(f, model.tok_emb.weight, V, D)
        for layer in model.layers:
            for w in ("wq", "wk", "wv", "wo"):
                lin = getattr(layer, w)
                mat(f, lin.weight.T, D, D)
                mat(f, lin.bias.view(1, -1), 1, D)
            mat(f, layer.ln1.weight.view(1, -1), 1, D)
            mat(f, layer.ln1.bias.view(1, -1), 1, D)
            mat(f, layer.l1.weight.T, D, Fd)
            mat(f, layer.l1.bias.view(1, -1), 1, Fd)
            mat(f, layer.l2.weight.T, Fd, D)
            mat(f, layer.l2.bias.view(1, -1), 1, D)
            mat(f, layer.ln2.weight.view(1, -1), 1, D)
            mat(f, layer.ln2.bias.view(1, -1), 1, D)
        mat(f, model.final_ln.weight.view(1, -1), 1, D)
        mat(f, model.final_ln.bias.view(1, -1), 1, D)
        mat(f, model.out_head.weight.T, D, V)
        mat(f, model.out_head.bias.view(1, -1), 1, V)


def epoch_checkpoint_path(save_path, epoch):
    """The orchestrator's naming (agpt_experiment.cr epoch_checkpoint_path)."""
    suffix = f".epoch_{epoch:06d}.model"
    return save_path[: -len(".model")] + suffix if save_path.endswith(".model") else save_path + suffix


def scheduled_lr(lr, name, min_ratio, step, total, warmup):
    """agpt_train_v2.cu scheduled_lr, step for step."""
    if name == "constant" or total <= 1:
        return lr
    warmup = max(0, min(warmup, total))
    if warmup > 0 and step < warmup:
        return lr * min(1.0, max(0.0, (step + 1) / warmup))
    decay = total - warmup
    if decay <= 0:
        return lr
    progress = min(1.0, max(0.0, (step - warmup) / decay))
    min_ratio = min(1.0, max(0.0, min_ratio))
    return lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg_y = yaml.safe_load(Path(args.config).read_text())
    corpus = cfg_y["corpus"]
    mcfg = cfg_y.get("model") or {}
    tcfg = cfg_y["train"]
    exp = cfg_y.get("experimental") or {}
    for k in exp:
        if k not in KNOWN_EXPERIMENTAL:
            print(f"agpt_minibatch_train: warning: unknown experimental key {k!r} ignored", flush=True)

    budget = tcfg["budget"]
    if budget["unit"] != "epochs":
        sys.exit(f"agpt_minibatch_train: budget unit {budget['unit']!r} not supported (epochs only)")
    epochs = int(budget["value"])
    seed = args.seed if args.seed is not None else int(tcfg.get("seed", 42))
    T = int(tcfg["seq_len"])
    opt_cfg = tcfg["optimizer"]
    if opt_cfg["name"] != "adam":
        sys.exit(f"agpt_minibatch_train: optimizer {opt_cfg['name']!r} not supported (adam only)")
    lr = float(opt_cfg["lr"])
    beta1 = float(opt_cfg.get("momentum_beta") or 0.9)
    beta2 = float(opt_cfg.get("beta") or 0.999)
    eps = float(opt_cfg.get("eps") or 1e-8)
    sched = tcfg.get("lr_schedule") or {}
    sched_name = sched.get("name", "constant")
    if sched_name not in ("constant", "warmup-cosine", "warmup_cosine"):
        sys.exit(f"agpt_minibatch_train: lr_schedule {sched_name!r} not supported")
    sched_name = "constant" if sched_name == "constant" else "warmup-cosine"
    warmup_epochs = int(sched.get("warmup_epochs", 0))
    min_ratio = float(sched.get("min_lr_ratio", 0.0))
    checkpoints = sorted(set(tcfg.get("checkpoint_epochs") or []))
    batch = int(exp.get("minibatch_size", 128))
    micro = min(batch, int(exp.get("micro_batch") or batch))
    device = torch.device(exp.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_path = mcfg["save_file"]

    vocab_src = corpus.get("vocab_source") or corpus["path"]
    char_to_id, V = build_vocab(vocab_src)
    text = Path(corpus["path"]).read_text(encoding="utf-8", errors="replace")
    if exp.get("max_corpus_chars"):
        text = text[: int(exp["max_corpus_chars"])]
    tokens = torch.tensor([char_to_id.get(c, 0) for c in text], dtype=torch.long)
    n_starts = tokens.numel() - T - 1
    if n_starts <= 0:
        sys.exit(f"agpt_minibatch_train: corpus too short ({tokens.numel()} chars) for seq_len {T}")

    if mcfg.get("init_file"):
        mc, sd = load_model(mcfg["init_file"])
        if mc["vocab_size"] != V:
            sys.exit(f"agpt_minibatch_train: vocab {V} != init_file's {mc['vocab_size']}")
        mc["seq_len"] = T
        init_desc = f"init_file {mcfg['init_file']}"
    else:
        D, H = int(mcfg["d_model"]), int(mcfg["n_heads"])
        mc = dict(d_model=D, n_heads=H, n_layers=int(mcfg["n_layers"]), d_ff=int(mcfg["d_ff"]),
                  vocab_size=V, seq_len=T, head_dim=D // H)
        init_seed = int(mcfg.get("init_seed") or seed)
        sd = random_state_dict(D, H, mc["n_layers"], mc["d_ff"], V, init_seed)
        init_desc = f"random init, seed {init_seed}"

    torch.manual_seed(seed)
    model = AGPTModel(mc, sd, device="cpu").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    steps_per_epoch = (n_starts + batch - 1) // batch
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    print(f"agpt_minibatch_train: {init_desc}; d={mc['d_model']} L={mc['n_layers']} h={mc['n_heads']} "
          f"ff={mc['d_ff']} V={V} seq_len={T} params={sum(p.numel() for p in model.parameters())}", flush=True)
    print(f"  corpus {corpus['path']} chars={tokens.numel()} window_starts={n_starts} "
          f"targets_per_epoch={n_starts * T}", flush=True)
    print(f"  epochs={epochs} minibatch={batch} steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
          f"micro_batch={micro} lr={lr} schedule={sched_name} warmup_epochs={warmup_epochs} min_lr_ratio={min_ratio} "
          f"betas=({beta1}, {beta2}) eps={eps} seed={seed} device={device}", flush=True)

    tokens_dev = tokens.to(device)
    offsets = torch.arange(T + 1, device=device).view(1, T + 1)
    rng = torch.Generator().manual_seed(seed)
    step = 0
    train_wall = 0.0
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        perm = torch.randperm(n_starts, generator=rng).to(device)
        loss_sum = torch.zeros((), device=device)
        model.train()
        for i in range(0, n_starts, batch):
            starts = perm[i:i + batch]
            nb = starts.numel()
            cur = scheduled_lr(lr, sched_name, min_ratio, step, total_steps, warmup_steps)
            for g in opt.param_groups:
                g["lr"] = cur
            opt.zero_grad(set_to_none=True)
            for j in range(0, nb, micro):
                part = starts[j:j + micro]
                chunks = tokens_dev[part.view(-1, 1) + offsets]
                logits = model(chunks[:, :T])
                loss = F.cross_entropy(logits.reshape(-1, V), chunks[:, 1:].reshape(-1))
                (loss * (part.numel() / nb)).backward()
                loss_sum += loss.detach() * part.numel()
            opt.step()
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        train_wall += dt
        nll = loss_sum.item() / n_starts
        print(f"  epoch {epoch} train_nll={nll:.6f} train_ppl={math.exp(nll):.4f} steps={step} "
              f"lr_end={cur:.6g} wall={dt:.2f}s", flush=True)
        if epoch in checkpoints:
            p = epoch_checkpoint_path(save_path, epoch)
            save_model(p, mc, model)
            print(f"  train-epoch-checkpoint: epoch={epoch} train_wall_seconds={train_wall:.6f} path={p}", flush=True)

    save_model(save_path, mc, model)
    print(f"agpt_minibatch_train: done steps={step} train_wall_seconds={train_wall:.6f} saved {save_path}", flush=True)


if __name__ == "__main__":
    main()
