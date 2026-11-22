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
    experimental.step_events_file (one integer per line, the target events of one
                                  step; each step takes max(1, round(events / seq_len))
                                  random windows, and the steps run in a new random
                                  order each epoch. Replaces minibatch_size: random
                                  batches with an AGPT plan's unit sizes. Windows past
                                  one epoch's worth come from a second permutation.)
    experimental.window_stride   (conventional epochs: windows start every `stride`
                                  characters from a random offset drawn each epoch,
                                  so stride = seq_len trains each target once per
                                  epoch, as ordinary LM training does. Default 1:
                                  every window start.)
    experimental.trie_plan       (replay an AGPT plan, see trie_plan.py: `pd`, `packed`
                                  or `group`. Each step trains exactly its plan step's
                                  events: the step's windows run through the model
                                  (cut to 4, 8 or seq_len tokens by their deepest
                                  event) with a loss mask, mean over the events.)
    experimental.trie_plan_group (trie_plan `group`: 65-ish steps per pass, one per character,
                                  each holding the events whose context has that character
                                  `first` (the oldest: AGPT's prefix-trie pd=1 units, the
                                  same as pd depth 1) or `last` (the newest: the suffix
                                  tree's pd=1 units, target-sorted batches); or `window`:
                                  whole windows assigned at random each pass to groups of
                                  `first`'s sizes, the size-matched shuffled control. The
                                  steps run in a new random order each pass. With wrap_tail
                                  all three have exactly the same step sizes.)
    experimental.trie_plan_depth      (pd: the partition depth)
    experimental.trie_plan_piece_max  (packed: most events in a piece or node chunk)
    experimental.trie_plan_steps      (packed: steps per pass)
    experimental.trie_plan_seed       (unit order / packing / chunking seed, default 1)
    experimental.trie_targets    (true: each position's target is the trie's next-token
                                  distribution for its context, n(c, .) / n(c) over
                                  every event of the corpus, instead of the one token
                                  that follows. Same expected gradient, lower variance;
                                  a context seen once keeps its one-hot target. With
                                  trie_plan, each step's event loss uses the soft
                                  targets; since a plan step holds every event of each
                                  of its contexts, the corpus's own counts give exactly
                                  the hard-target steps. train_nll still reports the
                                  one-hot loss.)
    experimental.trie_targets_corpus (with trie_targets: count each training context's
                                  next tokens over this corpus instead, normally a
                                  larger one containing the training corpus: the
                                  trie keeps its nodes and gets the larger corpus's
                                  child counts. Read as bytes through the vocab; every
                                  byte must be in it. A context it never contains keeps
                                  its one-hot target.)
    experimental.trie_targets_corpus_sha256 (required with trie_targets_corpus: its
                                  SHA-256, checked before counting.)
    experimental.trie_weights_corpus (true, with trie_targets_corpus: weight each event
                                  by n_ref(c) / n(c), its context's count in that corpus
                                  over its count in the training corpus, so each context
                                  carries the larger corpus's weight instead of its own:
                                  the training corpus's events stand for the larger
                                  corpus's loss on the contexts they share. A step's loss
                                  is the weighted sum over its weight (in place of the
                                  event mean), so pd=0 takes the gradient of that loss. A
                                  context the corpus lacks weighs 0. Not with
                                  grad_average.)
    experimental.teacher_targets_file (plain minibatch only: a float16 .npy of shape
                                  (corpus chars, V), row p a teacher's distribution for
                                  character p, e.g. from rnd/forward-backward-coupling/
                                  teacher_preds.py. Each target's loss becomes
                                  (CE(y) + beta CE(teacher, model)) / (1 + beta), the
                                  data term plus beta KL(teacher || model) up to a
                                  constant, with the teacher fixed (stop-gradient).
                                  train_nll still reports the one-hot loss.)
    experimental.teacher_targets_sha256 (required with teacher_targets_file: its SHA-256.)
    experimental.teacher_beta    (with teacher_targets_file: beta, >= 0.)
    experimental.wrap_tail       (true: a window starts at every character and reads past
                                  the end from the front, as the cached tries do, boundary
                                  mode tail-wrap-front-v1. Default: only windows that fit.)
    experimental.trie_plan_order_file (pd only: the units.tsv of a v2 gradient dump
                                  (AGPT_GRAD_DUMP_DIR); every pass takes the units in its
                                  row order instead of a shuffle, so the replay takes v2's
                                  steps in v2's order. Each row must name a unit of the
                                  plan with the same event count.)
    experimental.log_step_loss   (trie_plan only: print every step's event-mean loss, as
                                  v2 prints every unit's.)
    experimental.grad_average    (trie_plan only: what Adam is given instead of the step's
                                  own event-mean gradient.
                                  `sag` (pd plans): stochastic average gradient. A table
                                  keeps every unit's latest summed gradient and its events;
                                  a step recomputes its unit, replaces the unit's row, and
                                  passes Adam the sum of all rows over the sum of their
                                  events: the mass-weighted mean over every unit seen so
                                  far, each at the weights it was computed at. In the first
                                  pass, units not yet reached are left out. The table is
                                  float32 on the host (units x parameters x 4 bytes).
                                  `ema`: no table. D = beta D + G, M = beta M + E, and Adam
                                  gets D / M, where G and E are the step's summed gradient
                                  and events: mass-weighted momentum. With beta 1 and each
                                  unit once, it is the same as sag's first pass.)
    experimental.grad_average_beta (ema only: beta, default 1.)
    experimental.noise_theta     (> 0: agpt_train_v2's split-half noise cadence with a fresh
                                  sample, run_noise_cadence_v2 with growth_noise_resample, on
                                  plain windows with no trie: the adaptive-batch control.
                                  Windows come in random order (a new permutation each pass
                                  through the text); sample j goes to half A if j is even, B if
                                  odd. G and E are each half's summed gradient and events
                                  (seq_len per window). noise^2 = |G_A/E_A - G_B/E_B|^2 n_A n_B /
                                  n^2 (1 - n/N), signal^2 = |g|^2 - noise^2, g = (G_A + G_B) /
                                  (E_A + E_B). Step on g when noise <= theta x signal (or the
                                  budget is spent); otherwise add windows (the norm test's
                                  target, clamped to [1.05 n, max_ratio n]; only the new
                                  windows' gradient is computed, at the same weights). After a
                                  step the next sample has the size the step took. Compute is
                                  counted in positions, seq_len per window per gradient (a
                                  minibatch epoch's meter); the LR schedule runs over the
                                  budget, as v2's does. train.budget is not used.)
    experimental.noise_budget_positions (with noise_theta, required: the run's positions.)
    experimental.noise_start     (with noise_theta: the first sample's windows, default 32.)
    experimental.noise_max_ratio (with noise_theta: the most a growth multiplies the sample
                                  by, default 2.)
                                  micro_batch (default 8192 here) bounds the windows in one
                                  forward/backward.
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

KNOWN_EXPERIMENTAL = {"minibatch_size", "micro_batch", "device", "max_corpus_chars", "step_events_file",
                      "window_stride", "trie_plan", "trie_plan_depth", "trie_plan_piece_max",
                      "trie_plan_steps", "trie_plan_seed", "trie_targets", "wrap_tail",
                      "trie_plan_order_file", "log_step_loss", "grad_average", "grad_average_beta",
                      "trie_plan_group", "trie_targets_corpus", "trie_targets_corpus_sha256",
                      "trie_weights_corpus", "teacher_targets_file", "teacher_targets_sha256",
                      "teacher_beta", "noise_theta", "noise_budget_positions", "noise_start",
                      "noise_max_ratio"}
TOKEN_CAP = 131072  # tokens per forward/backward in trie-plan steps (memory bound only)


class TrieTargets:
    """The trie's next-token distribution for every event's context, on the device.

    dist(starts, nxt) gives, for windows starting at `starts`, a [windows * T, V]
    tensor (window-major, like the logits) whose row for event (i, L) is
    n(c, .) / n(c), c = tokens[i:i+L], counted over every event of the
    corpus. Summed over the occurrences of c, the one-hot targets equal
    n(c) times this row, so for any sample of events the soft loss has the
    hard loss's expected gradient: AGPT's node loss, at minibatch cadence.
    `nxt` is the events' own next tokens (the one-hot targets, flattened the
    same way): a context seen once has its own next token as its row.
    """

    def __init__(self, tokens, n_starts, T, V, device, ref=None, weights=False):
        import numpy as np
        import trie_plan as tp
        t0 = time.perf_counter()
        if ref is None:
            ctx, indptr, child, count, self.n_ctx, self.n_single = tp.context_targets(tokens.numpy(), n_starts, T, V)
        else:
            ctx, indptr, child, count, self.n_ctx, self.n_single = tp.context_targets_from(
                tokens.numpy(), n_starts, T, V, ref)
        self.n_missing = int((ctx < 0).sum()) if ref is not None else None
        self.wtab = None
        if weights:
            # w(c) = n_ref(c) / n(c): the reference corpus's count of the context over the
            # training corpus's (its events here). Per event, [T, n_starts] like ctx.
            n_found = indptr.size - 1
            n_ref = np.add.reduceat(count.astype(np.float64), indptr[:-1]) if n_found else np.zeros(0)
            n_own = np.bincount(ctx[ctx >= 0].astype(np.int64), minlength=n_found).astype(np.float64)
            wc = n_ref / n_own
            wt = np.where(ctx >= 0, wc[np.maximum(ctx, 0)], 0.0)
            self.w_total, self.w_events = float(wt.sum()), int(ctx.size)
            self.w_ctx = wc
            self.wtab = torch.from_numpy(wt.astype(np.float32)).to(device)
        self.n_pairs = child.size + (self.n_single if ref is None else 0)
        self.ctx = torch.from_numpy(ctx).to(device)
        self.indptr = torch.from_numpy(indptr).to(device)
        self.child = torch.from_numpy(child).to(device)
        self.count = torch.from_numpy(count).to(device=device, dtype=torch.float32)
        self.V, self.device = V, device
        self.build_seconds = time.perf_counter() - t0

    def dist(self, starts, nxt, dtype=torch.float32, L=None):
        ids = self.ctx[:L, starts].t().reshape(-1).long()
        n = ids.numel()
        out = torch.zeros(n, self.V, device=self.device, dtype=dtype)
        once = torch.nonzero(ids < 0).squeeze(1)
        out[once, nxt[once]] = 1
        rr = torch.nonzero(ids >= 0).squeeze(1)
        a = self.indptr[ids[rr]]
        lens = self.indptr[ids[rr] + 1] - a
        row = torch.repeat_interleave(rr, lens)
        first = torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
        pidx = torch.repeat_interleave(a, lens) + torch.arange(row.numel(), device=self.device) - first
        out.index_put_((row, self.child[pidx].long()), self.count[pidx].to(dtype))
        return out / out.sum(-1, keepdim=True)

    def weights(self, starts, L):
        """[windows, L] event weights n_ref(c) / n(c) (trie_weights_corpus)."""
        return self.wtab[:L, starts].t()


class TriePlanner:
    """Builds one pass's steps of an AGPT plan over the training windows."""

    def __init__(self, exp, tokens, n_starts, T):
        import numpy as np
        import trie_plan as tp
        self.np, self.tp, self.T, self.n = np, tp, T, n_starts
        self.kind = exp["trie_plan"]
        if self.kind not in ("pd", "packed", "group"):
            sys.exit(f"agpt_minibatch_train: trie_plan {self.kind!r} not supported (pd | packed | group)")
        if (self.kind == "group") != ("trie_plan_group" in exp):
            sys.exit("agpt_minibatch_train: trie_plan_group goes with trie_plan group")
        self.rng = np.random.default_rng(int(exp.get("trie_plan_seed", 1)))
        t0 = time.perf_counter()
        tok = tokens.numpy()
        keys = tp.prefix_keys(tok, n_starts, T)
        self.keys = keys
        self.fixed_order = None
        if exp.get("trie_plan_order_file") and self.kind != "pd":
            sys.exit("agpt_minibatch_train: trie_plan_order_file is for trie_plan pd")
        self.group = None
        if self.kind == "pd":
            self.depth = int(exp["trie_plan_depth"])
            tags = tp.pd_unit_tags(keys, self.depth)
            self.desc = f"pd={self.depth} + crown, one unit per step, shuffled each pass"
        elif self.kind == "group":
            self.group = exp["trie_plan_group"]
            if self.group not in ("first", "last", "window"):
                sys.exit(f"agpt_minibatch_train: trie_plan_group {self.group!r} not supported (first | last | window)")
            first = tok[:n_starts].astype(np.uint64)
            if self.group == "last":  # event (i, L): newest context character tok[i + L - 1]
                tags = tok[np.arange(n_starts)[:, None] + np.arange(T)[None, :]].astype(np.uint64)
                self.desc = "groups by the newest context character (suffix-tree pd=1 units), shuffled each pass"
            else:
                tags = np.broadcast_to(first[:, None], (n_starts, T))
                self.desc = "groups by the oldest context character (prefix-trie pd=1 units), shuffled each pass"
            if self.group == "window":
                # group sizes of `first`, in windows; reassigned at random every pass
                _, self.window_counts = np.unique(first, return_counts=True)
                self.desc = ("whole windows assigned at random each pass to groups of the prefix-trie "
                             "pd=1 units' sizes, shuffled each pass")
        else:
            self.piece_max = int(exp["trie_plan_piece_max"])
            self.n_steps_req = int(exp["trie_plan_steps"])
            occ = tp.occurrences(keys)
            tags, p = tp.packed_unit_tags(keys, occ, self.piece_max, self.rng)
            del occ
            self.desc = (f"packed: pieces and node chunks of <= {self.piece_max} events "
                         f"(piece depth median {int(np.median(p))}), {self.n_steps_req} equal steps, repacked each pass")
        self.unit_of_event, self.unit_sizes = tp.units_from_tags(tags)
        del tags
        if exp.get("trie_plan_order_file"):
            import csv
            import hashlib
            path = Path(exp["trie_plan_order_file"])
            rows = list(csv.DictReader(path.open(), delimiter="\t"))
            self.fixed_order = tp.order_from_dump(keys, self.unit_of_event, self.unit_sizes, rows, self.depth)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.desc = (f"pd={self.depth} + crown, one unit per step, every pass in the order of {path} "
                         f"(sha256 {sha})")
        self.blens = sorted({min(4, T), min(8, T), T})
        self.cache = None
        if self.kind == "group":
            # cut every window at its deepest event (a `last` step's windows are often short)
            self.blens = list(range(1, T + 1))
            if self.group != "window":
                # a fixed partition: sort its (unit, window) pairs once; a pass only reorders the units
                U = self.unit_sizes.size
                sw_u, sw_i, sw_bits = tp.step_windows(self.unit_of_event, np.arange(U, dtype=np.int32), n_starts, T)
                bucket = np.searchsorted(np.array(self.blens), tp.bit_length(sw_bits)).astype(np.int64)
                order = np.lexsort((sw_i, bucket, sw_u))
                sw_u, sw_i, sw_bits, bucket = sw_u[order], sw_i[order], sw_bits[order], bucket[order]
                nb = len(self.blens)
                ub = np.bincount(sw_u.astype(np.int64) * nb + bucket, minlength=U * nb).reshape(U, nb)
                self.cache = (sw_i, sw_bits, ub, np.r_[0, np.cumsum(ub.sum(1))])
                del sw_u, bucket, order
                self.unit_of_event = None  # not needed once the pairs are cached
        self.build_seconds = time.perf_counter() - t0

    @property
    def steps_per_epoch(self):
        return self.n_steps_req if self.kind == "packed" else self.unit_sizes.size

    def epoch(self, with_cost):
        np, tp = self.np, self.tp
        if self.fixed_order is not None:
            step_of_unit = self.fixed_order
            n_steps = self.unit_sizes.size
        elif self.kind in ("pd", "group"):
            if self.group == "window":
                counts = self.window_counts
                gw = np.empty(self.n, dtype=np.int32)
                gw[self.rng.permutation(self.n)] = np.repeat(np.arange(counts.size, dtype=np.int32), counts)
                self.unit_of_event = np.repeat(gw, self.T)
            step_of_unit = tp.shuffled_steps(self.unit_sizes.size, self.rng)
            n_steps = self.unit_sizes.size
        else:
            step_of_unit, n_steps = tp.packed_steps(self.unit_sizes, self.n_steps_req, self.rng)
        if self.cache is not None:
            unit_of_step = np.empty(n_steps, dtype=np.int64)
            unit_of_step[step_of_unit] = np.arange(n_steps)
            c_i, c_bits, ub, start = self.cache
            idx = np.concatenate([np.arange(start[u], start[u + 1]) for u in unit_of_step])
            sw_i, sw_bits = c_i[idx], c_bits[idx]
            per_step = ub[unit_of_step]
            cost = None
            if with_cost:
                sw_step = np.repeat(np.arange(n_steps, dtype=np.int32), per_step.sum(1))
                cost = tp.plan_cost(self.keys, sw_step, sw_i, sw_bits)
                self.keys = None  # costed once, on the first pass
            step_events = self.unit_sizes[unit_of_step].astype(np.float64)
            return dict(n_steps=n_steps, sw_i=sw_i, sw_bits=sw_bits, off=np.r_[0, np.cumsum(per_step.ravel())].tolist(),
                        step_events=step_events.tolist(), cost=cost, pairs=int(sw_i.size), sizes=step_events,
                        unit_of_step=unit_of_step)
        sw_step, sw_i, sw_bits = tp.step_windows(self.unit_of_event, step_of_unit, self.n, self.T)
        cost = tp.plan_cost(self.keys, sw_step, sw_i, sw_bits) if with_cost else None
        lmax = tp.bit_length(sw_bits)
        bucket = np.searchsorted(np.array(self.blens), lmax).astype(np.int64)
        order = np.lexsort((sw_i, bucket, sw_step))
        sw_step, sw_i, sw_bits, bucket = sw_step[order], sw_i[order], sw_bits[order], bucket[order]
        nb = len(self.blens)
        counts = np.bincount(sw_step.astype(np.int64) * nb + bucket, minlength=n_steps * nb)
        off = np.r_[0, np.cumsum(counts)].tolist()
        step_events = np.bincount(sw_step, weights=np.bitwise_count(sw_bits), minlength=n_steps)
        unit_of_step = None
        if self.kind in ("pd", "group"):  # one unit per step: invert unit -> step
            unit_of_step = np.empty(n_steps, dtype=np.int64)
            unit_of_step[step_of_unit] = np.arange(n_steps)
        return dict(n_steps=n_steps, sw_i=sw_i, sw_bits=sw_bits, off=off, step_events=step_events.tolist(),
                    cost=cost, pairs=int(sw_i.size), sizes=step_events, unit_of_step=unit_of_step)


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


def run_noise_cadence(exp, model, opt, tokens_dev, n_starts, T, V, device, seed, lr, sched_name, min_ratio, micro):
    """experimental.noise_theta: run_noise_cadence_v2 (fresh sample) on plain windows. Returns the steps taken."""
    theta = float(exp["noise_theta"])
    budget = int(exp["noise_budget_positions"])
    max_ratio = float(exp.get("noise_max_ratio", 2.0))
    N = n_starts
    n0 = min(int(exp.get("noise_start", 32)), N)
    S = 1_000_000  # v2's kScheduleSteps: the LR is looked up at used / budget of the way
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    offsets = torch.arange(T + 1, device=device).view(1, T + 1)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=rng)
    nxt, data_epoch, drawn = 0, 0, 0

    def take(count):
        """The next `count` window starts of the shuffled order (reshuffled when it runs out)."""
        nonlocal perm, nxt, data_epoch, drawn
        parts = []
        while count > 0:
            if nxt >= N:
                data_epoch += 1
                perm = torch.randperm(N, generator=rng)
                nxt = 0
            k = min(count, N - nxt)
            parts.append(perm[nxt:nxt + k])
            nxt += k
            count -= k
            drawn += k
        return torch.cat(parts).to(device)

    def split(starts, index):
        """Sample j (counted from `index`) to half A if j is even, B if odd."""
        a0 = index % 2
        return starts[a0::2], starts[1 - a0::2]

    def grad_sum(starts):
        """The summed gradient (float64, flat), summed loss and events of these windows at the current weights."""
        g = torch.zeros(n_params, dtype=torch.float64, device=device)
        loss = torch.zeros((), dtype=torch.float64, device=device)
        for j in range(0, starts.numel(), micro):
            part = starts[j:j + micro]
            chunks = tokens_dev[part.view(-1, 1) + offsets]
            logits = model(chunks[:, :T])
            ce = F.cross_entropy(logits.reshape(-1, V), chunks[:, 1:].reshape(-1), reduction="sum")
            opt.zero_grad(set_to_none=True)
            ce.backward()
            g += torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                            for p in params]).double()
            loss += ce.detach().double()
        return g, loss.item(), starts.numel() * T

    print(f"  noise-cadence (no trie): theta={theta:g} budget_positions={budget} start_windows={n0} "
          f"max_ratio={max_ratio:g} windows={N} order=random seed={seed} sample=fresh-each-step "
          f"positions=seq_len per window per gradient micro_batch={micro} lr={lr} schedule={sched_name} "
          f"lr_min_ratio={min_ratio:.3f} (train.budget not used)", flush=True)
    model.train()
    sa, sb = split(take(n0), 0)
    n_a, n_b = sa.numel(), sb.numel()
    used = steps = recomputes = growths = growth_windows = 0
    need_full = True
    while True:
        if need_full:
            if used >= budget:
                break
            g_a, l_a, e_a = grad_sum(sa)
            g_b, l_b, e_b = grad_sum(sb)
            used += e_a + e_b
            recomputes += 1
            need_full = False
        n = n_a + n_b
        e_ab = e_a + e_b
        mean = (g_a + g_b) / e_ab
        d2 = (g_a / e_a - g_b / e_b).square().sum().item()
        m2 = mean.square().sum().item()
        fpc = max(0.0, 1.0 - n / N)
        noise2 = d2 * n_a * n_b / (n * n) * fpc
        signal2 = max(0.0, m2 - noise2)
        ratio = math.sqrt(noise2 / signal2) if signal2 > 0 else math.inf
        spent = used >= budget
        passed = signal2 > 0 and noise2 <= theta * theta * signal2
        loss = (l_a + l_b) / e_ab
        if passed or spent:
            cur = scheduled_lr(lr, sched_name, min_ratio, min(S, int(used / budget * S)), S, 0)
            for grp in opt.param_groups:
                grp["lr"] = cur
            o = 0
            for p in params:
                k = p.numel()
                p.grad = mean[o:o + k].view_as(p).float()
                o += k
            opt.step()
            steps += 1
            print(f"  noise-step {steps}: windows={n} loss={loss:.6f} noise/signal={ratio:.4f} grad={math.sqrt(m2):.4g} "
                  f"lr={cur:.6g} positions={used}{'' if passed else ' (budget spent)'}", flush=True)
            need_full = True
            sa, sb = split(take(n), 0)
            n_a, n_b = sa.numel(), sb.numel()
            continue
        inv_gap = 1.0 / n - 1.0 / N
        if signal2 <= 0 or inv_gap <= 0:
            target = n * max_ratio
        else:
            a = noise2 / inv_gap  # the per-window variance, in g's units
            target = 1.0 / (theta * theta * signal2 / a + 1.0 / N)
        lo = n + max(1, math.ceil(0.05 * n))
        hi = max(lo, math.floor(n * max_ratio))
        want = min(max(math.ceil(target), lo), hi)
        na, nb = split(take(want - n), n)
        ga, la, ea = grad_sum(na)
        gb, lb, eb = grad_sum(nb)
        g_a += ga
        g_b += gb
        l_a, e_a, l_b, e_b = l_a + la, e_a + ea, l_b + lb, e_b + eb
        n_a, n_b = n_a + na.numel(), n_b + nb.numel()
        used += ea + eb
        growths += 1
        growth_windows += na.numel() + nb.numel()
        print(f"  noise-grow {growths}: windows {n} -> {n_a + n_b} (target {target:.0f}) noise/signal={ratio:.4f} "
              f"fresh_loss={(la + lb) / (ea + eb):.6f} sample_loss={loss:.6f} positions={used}", flush=True)
    print(f"  train-growth: completed noise-cadence optimizer_steps={steps} windows={n_a + n_b} positions={used} "
          f"full_recomputes={recomputes} growths={growths} growth_windows={growth_windows} windows_drawn={drawn} "
          f"data_epochs={drawn / N:.3f}", flush=True)
    return steps


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
    # `or` would turn an explicit momentum_beta 0 into 0.9
    beta1 = float(opt_cfg["momentum_beta"]) if opt_cfg.get("momentum_beta") is not None else 0.9
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
    step_sizes = None
    if exp.get("step_events_file"):
        if "minibatch_size" in exp:
            sys.exit("agpt_minibatch_train: give minibatch_size or step_events_file, not both")
        ev = [int(x) for x in Path(exp["step_events_file"]).read_text().split()]
        step_sizes = torch.tensor([max(1, round(e / int(tcfg["seq_len"]))) for e in ev], dtype=torch.long)
        batch = int(step_sizes.max())
    else:
        batch = int(exp.get("minibatch_size", 128))
    stride = int(exp.get("window_stride", 1))
    if stride < 1:
        sys.exit("agpt_minibatch_train: window_stride must be >= 1")
    if exp.get("trie_plan") and (step_sizes is not None or "minibatch_size" in exp or stride != 1):
        sys.exit("agpt_minibatch_train: trie_plan replaces minibatch_size, step_events_file and window_stride")
    if stride != 1 and step_sizes is not None:
        sys.exit("agpt_minibatch_train: give window_stride or step_events_file, not both")
    if exp.get("trie_targets_corpus") and not exp.get("trie_targets"):
        sys.exit("agpt_minibatch_train: trie_targets_corpus goes with trie_targets: true")
    if bool(exp.get("trie_targets_corpus")) != bool(exp.get("trie_targets_corpus_sha256")):
        sys.exit("agpt_minibatch_train: trie_targets_corpus and trie_targets_corpus_sha256 go together")
    weighted = bool(exp.get("trie_weights_corpus"))
    if weighted and not exp.get("trie_targets_corpus"):
        sys.exit("agpt_minibatch_train: trie_weights_corpus goes with trie_targets_corpus")
    if weighted and exp.get("grad_average"):
        sys.exit("agpt_minibatch_train: trie_weights_corpus is not supported with grad_average")
    teach = exp.get("teacher_targets_file")
    if bool(teach) != bool(exp.get("teacher_targets_sha256")) or bool(teach) != ("teacher_beta" in exp):
        sys.exit("agpt_minibatch_train: teacher_targets_file, teacher_targets_sha256 and teacher_beta go together")
    if teach and (exp.get("trie_plan") or exp.get("trie_targets") or exp.get("wrap_tail")):
        sys.exit("agpt_minibatch_train: teacher_targets_file is for plain minibatch runs (no trie_plan, "
                 "trie_targets or wrap_tail)")
    teacher_beta = float(exp.get("teacher_beta", 0.0))
    if teacher_beta < 0:
        sys.exit("agpt_minibatch_train: teacher_beta must be >= 0")
    log_steps = bool(exp.get("log_step_loss"))
    if log_steps and not exp.get("trie_plan"):
        sys.exit("agpt_minibatch_train: log_step_loss is for trie_plan runs")
    avg = exp.get("grad_average")
    if avg is not None:
        if avg not in ("sag", "ema"):
            sys.exit(f"agpt_minibatch_train: grad_average {avg!r} not supported (sag | ema)")
        if not exp.get("trie_plan"):
            sys.exit("agpt_minibatch_train: grad_average is for trie_plan runs")
        if avg == "sag" and exp["trie_plan"] != "pd":
            sys.exit("agpt_minibatch_train: grad_average sag needs trie_plan pd (one unit per step)")
    if "grad_average_beta" in exp and avg != "ema":
        sys.exit("agpt_minibatch_train: grad_average_beta is for grad_average ema")
    avg_beta = float(exp.get("grad_average_beta", 1.0))
    if not 0.0 <= avg_beta <= 1.0:
        sys.exit("agpt_minibatch_train: grad_average_beta must be in [0, 1]")
    noise = "noise_theta" in exp
    if noise:
        if float(exp["noise_theta"]) <= 0 or int(exp.get("noise_budget_positions") or 0) <= 0:
            sys.exit("agpt_minibatch_train: noise_theta must be > 0 and needs noise_budget_positions > 0")
        if int(exp.get("noise_start", 32)) < 2 or float(exp.get("noise_max_ratio", 2.0)) <= 1.0:
            sys.exit("agpt_minibatch_train: noise_start must be >= 2 and noise_max_ratio > 1")
        for k in ("minibatch_size", "step_events_file", "window_stride", "trie_plan", "trie_targets",
                  "teacher_targets_file", "wrap_tail"):
            if k in exp:
                sys.exit(f"agpt_minibatch_train: noise_theta is for plain windows, not with {k}")
    elif any(k in exp for k in ("noise_budget_positions", "noise_start", "noise_max_ratio")):
        sys.exit("agpt_minibatch_train: noise_budget_positions, noise_start and noise_max_ratio go with noise_theta")
    micro = int(exp.get("micro_batch") or 8192) if noise else min(batch, int(exp.get("micro_batch") or batch))
    device = torch.device(exp.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        # torch.optim.Adam's step checks for CUDA graph capture, which opens a CUDA
        # context (~128 MiB of GPU memory) even for CPU tensors; hide the GPU first.
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    save_path = mcfg["save_file"]

    vocab_src = corpus.get("vocab_source") or corpus["path"]
    char_to_id, V = build_vocab(vocab_src)
    text = Path(corpus["path"]).read_text(encoding="utf-8", errors="replace")
    if exp.get("max_corpus_chars"):
        text = text[: int(exp["max_corpus_chars"])]
    tokens = torch.tensor([char_to_id.get(c, 0) for c in text], dtype=torch.long)
    n_chars = tokens.numel()
    if exp.get("wrap_tail"):
        # The trie's boundary_mode tail-wrap-front-v1: a window starts at every
        # character, and past the end it reads on from the front.
        tokens = torch.cat([tokens, tokens[:T]])
        n_starts = n_chars
    else:
        n_starts = n_chars - T - 1
    if n_starts <= 0 or n_chars <= T:
        sys.exit(f"agpt_minibatch_train: corpus too short ({n_chars} chars) for seq_len {T}")

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
    planner = TriePlanner(exp, tokens, n_starts, T) if exp.get("trie_plan") else None
    ref = None
    if exp.get("trie_targets_corpus"):
        import hashlib
        import numpy as np
        raw = Path(exp["trie_targets_corpus"]).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != exp["trie_targets_corpus_sha256"]:
            sys.exit(f"agpt_minibatch_train: trie_targets_corpus sha256 {sha} != the config's")
        lut = np.full(256, -1, dtype=np.int16)
        for ch, i in char_to_id.items():
            b = ch.encode("utf-8")
            if len(b) == 1:
                lut[b[0]] = i
        ref = lut[np.frombuffer(raw, dtype=np.uint8)]
        if (ref < 0).any():
            bad = sorted(set(np.frombuffer(raw, dtype=np.uint8)[ref < 0].tolist()))[:10]
            sys.exit(f"agpt_minibatch_train: trie_targets_corpus has bytes outside the vocab: {bad}")
        ref = ref.astype(np.uint8)
        del raw
        print(f"  trie_targets_corpus {exp['trie_targets_corpus']}: {ref.size} tokens, sha256 {sha}", flush=True)
    targets = TrieTargets(tokens, n_starts, T, V, device, ref, weighted) if exp.get("trie_targets") else None
    del ref
    n_windows = n_starts // stride if stride > 1 else n_starts
    if planner is not None:
        steps_per_epoch = planner.steps_per_epoch
    elif step_sizes is not None:
        steps_per_epoch = len(step_sizes)
    else:
        steps_per_epoch = (n_windows + batch - 1) // batch
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    print(f"agpt_minibatch_train: {init_desc}; d={mc['d_model']} L={mc['n_layers']} h={mc['n_heads']} "
          f"ff={mc['d_ff']} V={V} seq_len={T} params={sum(p.numel() for p in model.parameters())}", flush=True)
    print(f"  corpus {corpus['path']} chars={n_chars} window_starts={n_starts} "
          f"targets_per_epoch={n_starts * T}" + (" (tail wrapped to the front)" if exp.get("wrap_tail") else ""),
          flush=True)
    if noise:
        print(f"  betas=({beta1}, {beta2}) eps={eps} seed={seed} device={device}", flush=True)
    else:
        print(f"  epochs={epochs} minibatch={batch} steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
              f"micro_batch={micro} lr={lr} schedule={sched_name} warmup_epochs={warmup_epochs} "
              f"min_lr_ratio={min_ratio} betas=({beta1}, {beta2}) eps={eps} seed={seed} device={device}", flush=True)
    if step_sizes is not None:
        print(f"  step_events_file {exp['step_events_file']}: {len(step_sizes)} steps per epoch, "
              f"{int(step_sizes.sum())} windows per epoch (median {int(step_sizes.median())}, max {batch}), "
              f"shuffled each epoch", flush=True)
    if stride > 1:
        print(f"  window_stride {stride}: {n_windows} windows per epoch from a random offset each epoch, "
              f"targets_per_epoch={n_windows * T}", flush=True)
    if targets is not None:
        if targets.n_missing is None:
            print(f"  trie_targets: {targets.n_ctx} contexts ({targets.n_single} seen once), "
                  f"{targets.n_pairs} (context, next) pairs, built in {targets.build_seconds:.1f}s", flush=True)
        else:
            print(f"  trie_targets from the corpus above: {targets.n_ctx} training contexts, "
                  f"{targets.n_pairs} (context, next) pairs, {targets.n_single} contexts "
                  f"found once there, {targets.n_missing} events whose context it lacks (one-hot), "
                  f"built in {targets.build_seconds:.1f}s", flush=True)
    if weighted:
        import numpy as np
        q = np.percentile(targets.w_ctx, [0, 50, 90, 99, 100])
        print(f"  trie_weights_corpus: events weighted n_ref(c) / n(c); total weight {targets.w_total:.0f} over "
              f"{targets.w_events} events ({targets.w_total / targets.w_events:.2f} per event); per context "
              f"min {q[0]:.3g} median {q[1]:.3g} p90 {q[2]:.3g} p99 {q[3]:.3g} max {q[4]:.3g}", flush=True)
    if planner is not None:
        print(f"  trie_plan {planner.desc}: {planner.unit_sizes.size} units "
              f"({planner.tp.describe(planner.unit_sizes)}), built in {planner.build_seconds:.1f}s", flush=True)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    if avg is not None:
        # D (float64, on the device) is the sum of the summed unit gradients Adam's
        # direction averages; M is their events. D / M goes to Adam in place of the
        # step's own event-mean gradient.
        avg_D = torch.zeros(n_params, dtype=torch.float64, device=device)
        avg_M = 0.0
        if avg == "sag":
            n_units = planner.unit_sizes.size
            sag_table = torch.zeros((n_units, n_params), dtype=torch.float32,
                                    pin_memory=(device.type == "cuda"))
            sag_mass = [0.0] * n_units  # 0: the unit has no row yet
            print(f"  grad_average sag: table of {n_units} units x {n_params} floats "
                  f"({sag_table.numel() * 4 / 2**20:.1f} MiB, host); Adam gets the mass-weighted mean of every "
                  f"unit's latest summed gradient", flush=True)
        else:
            print(f"  grad_average ema: beta {avg_beta:g}; Adam gets (beta D + G) / (beta M + E), "
                  f"mass-weighted momentum", flush=True)

    tokens_dev = tokens.to(device)
    offsets = torch.arange(T + 1, device=device).view(1, T + 1)
    teacher = None
    if teach:
        import hashlib
        import numpy as np
        raw = Path(teach).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != exp["teacher_targets_sha256"]:
            sys.exit(f"agpt_minibatch_train: teacher_targets_file sha256 {sha} != the config's")
        tarr = np.load(teach)
        if tarr.shape != (n_chars, V) or tarr.dtype != np.float16:
            sys.exit(f"agpt_minibatch_train: teacher_targets_file is {tarr.dtype} {tarr.shape}, "
                     f"want float16 ({n_chars}, {V})")
        teacher = torch.from_numpy(tarr).to(device)
        del raw, tarr
        tnll = -torch.log(teacher.float().gather(1, tokens_dev.view(-1, 1)).clamp_min(1e-12)).mean().item()
        print(f"  teacher_targets {teach}: sha256 {sha}, beta {teacher_beta:g}, teacher NLL of the corpus "
              f"{tnll:.4f}", flush=True)
    if noise:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step = run_noise_cadence(exp, model, opt, tokens_dev, n_starts, T, V, device, seed, lr, sched_name,
                                 min_ratio, micro)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_wall = time.perf_counter() - t0
        save_model(save_path, mc, model)
        print(f"agpt_minibatch_train: done steps={step} train_wall_seconds={train_wall:.6f} saved {save_path}",
              flush=True)
        return
    rng = torch.Generator().manual_seed(seed)
    step = 0
    train_wall = 0.0
    pos = torch.arange(T, device=device, dtype=torch.int32).view(1, T)
    for epoch in range(1, epochs + 1):
        if planner is not None:
            tb = time.perf_counter()
            plan = planner.epoch(with_cost=(epoch == 1))
            sizes_np = plan["sizes"]
            print(f"  trie_plan epoch {epoch}: {plan['n_steps']} steps ({planner.tp.describe(sizes_np.astype(int))}), "
                  f"{plan['pairs']} step-window pairs"
                  + (f", trie positions per pass {plan['cost']}" if plan["cost"] is not None else "")
                  + f", built in {time.perf_counter() - tb:.1f}s", flush=True)
            sw_i = torch.from_numpy(plan["sw_i"]).to(device)
            sw_bits = torch.from_numpy(plan["sw_bits"]).to(device)
            off, step_events, nb = plan["off"], plan["step_events"], len(planner.blens)
            if weighted:
                # each step-window pair's weight: the weights of its events (set bits)
                wp = torch.zeros(sw_i.numel(), device=device, dtype=torch.float64)
                for l in range(T):
                    wp += (((sw_bits >> l) & 1).double() * targets.wtab[l, sw_i.long()].double())
                cs = torch.cat([torch.zeros(1, device=device, dtype=torch.float64), torch.cumsum(wp, 0)])
                bnd = torch.tensor(off[0::nb], device=device, dtype=torch.long)
                step_w = (cs[bnd[1:]] - cs[bnd[:-1]]).tolist()
                if epoch == 1:
                    print(f"  trie_weights_corpus: the pass's step weights sum to {sum(step_w):.0f} "
                          f"(total {targets.w_total:.0f})", flush=True)
                del wp, cs, bnd
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss_sum = torch.zeros((), device=device)
            soft_sum = torch.zeros((), device=device)
            model.train()
            for s in range(plan["n_steps"]):
                cur = scheduled_lr(lr, sched_name, min_ratio, step, total_steps, warmup_steps)
                for g in opt.param_groups:
                    g["lr"] = cur
                opt.zero_grad(set_to_none=True)
                E = step_events[s]
                Wd = step_w[s] if weighted else E  # the step loss's denominator
                step_loss = torch.zeros((), device=device, dtype=torch.float64)
                for b, Lb in enumerate(planner.blens):
                    a, z = off[s * nb + b], off[s * nb + b + 1]
                    cap = max(1, TOKEN_CAP // Lb)
                    for j in range(a, z, cap):
                        ii = sw_i[j:min(z, j + cap)]
                        chunks = tokens_dev[ii.view(-1, 1) + offsets[:, :Lb + 1]]
                        logits = model(chunks[:, :Lb])
                        y = chunks[:, 1:].reshape(-1)
                        if targets is None:
                            ce = F.cross_entropy(logits.reshape(-1, V), y, reduction="none").view(-1, Lb)
                            obj = ce
                        else:
                            logp = F.log_softmax(logits.reshape(-1, V), dim=-1)
                            ce = F.nll_loss(logp.detach(), y, reduction="none").view(-1, Lb)
                            obj = -(targets.dist(ii, y, logp.dtype, Lb) * logp).sum(-1).view(-1, Lb)
                        m = ((sw_bits[j:min(z, j + cap)].view(-1, 1) >> pos[:, :Lb]) & 1).to(ce.dtype)
                        ls = (obj * m).sum()
                        if weighted:
                            ((obj * m * targets.weights(ii.long(), Lb)).sum() / Wd).backward()
                        else:
                            (ls / E).backward()
                        loss_sum += (ce.detach() * m).sum()
                        step_loss += ls.detach().double()
                        if targets is not None:
                            soft_sum += ls.detach()
                avg_note = ""
                if avg is not None:
                    # the step's summed gradient G (p.grad holds G / E)
                    G = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                   for p in params]).float() * E
                    if avg == "ema":
                        avg_D.mul_(avg_beta).add_(G.double())
                        avg_M = avg_beta * avg_M + E
                    else:
                        u = int(plan["unit_of_step"][s])
                        if sag_mass[u] > 0:
                            avg_D.sub_(sag_table[u].to(device).double())
                        avg_D.add_(G.double())
                        sag_table[u].copy_(G)
                        avg_M += E - sag_mass[u]
                        sag_mass[u] = E
                    d = (avg_D / avg_M).float()
                    o = 0
                    for p in params:
                        n = p.numel()
                        if p.grad is None:
                            p.grad = torch.empty_like(p)
                        p.grad.copy_(d[o:o + n].view_as(p))
                        o += n
                    if log_steps:
                        avg_note = f" avg_norm={d.norm().item():.6g} avg_events={avg_M:.0f}"
                opt.step()
                if log_steps:
                    print(f"    step {step + 1} events={E:.0f} mean_loss={step_loss.item() / E:.6f} lr={cur:.6g}"
                          + (f" weight={Wd:.0f}" if weighted else "") + avg_note, flush=True)
                step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            train_wall += dt
            nll = loss_sum.item() / float(sizes_np.sum())
            if avg == "sag":
                # re-sum D from the table each pass so float64 rounding cannot drift
                acc = torch.zeros(n_params, dtype=torch.float64)
                for r in range(0, sag_table.shape[0], 64):
                    acc += sag_table[r:r + 64].double().sum(0)
                avg_D.copy_(acc.to(device))
            del sw_i, sw_bits, plan
            soft_note = f" soft_ce={soft_sum.item() / float(sizes_np.sum()):.6f}" if targets is not None else ""
            print(f"  epoch {epoch} train_nll={nll:.6f} train_ppl={math.exp(nll):.4f}{soft_note} steps={step} "
                  f"lr_end={cur:.6g} wall={dt:.2f}s", flush=True)
            if epoch in checkpoints:
                p = epoch_checkpoint_path(save_path, epoch)
                save_model(p, mc, model)
                print(f"  train-epoch-checkpoint: epoch={epoch} train_wall_seconds={train_wall:.6f} path={p}",
                      flush=True)
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if stride > 1:
            o = int(torch.randint(stride, (1,), generator=rng))
            perm = o + stride * torch.randperm(n_windows, generator=rng)
        else:
            perm = torch.randperm(n_starts, generator=rng)
        if step_sizes is not None:
            sizes = step_sizes[torch.randperm(len(step_sizes), generator=rng)]
            need = int(sizes.sum())
            while perm.numel() < need:
                perm = torch.cat([perm, torch.randperm(n_starts, generator=rng)])
            bounds = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]).tolist()
        else:
            bounds = list(range(0, n_windows, batch)) + [n_windows]
        perm = perm.to(device)
        loss_sum = torch.zeros((), device=device)
        soft_sum = torch.zeros((), device=device)
        model.train()
        for i0, i1 in zip(bounds[:-1], bounds[1:]):
            starts = perm[i0:i1]
            nb = starts.numel()
            cur = scheduled_lr(lr, sched_name, min_ratio, step, total_steps, warmup_steps)
            for g in opt.param_groups:
                g["lr"] = cur
            opt.zero_grad(set_to_none=True)
            if weighted:
                Wb = targets.weights(starts, T).double().sum().item()
            for j in range(0, nb, micro):
                part = starts[j:j + micro]
                chunks = tokens_dev[part.view(-1, 1) + offsets]
                logits = model(chunks[:, :T])
                if targets is None and teacher is None:
                    loss = F.cross_entropy(logits.reshape(-1, V), chunks[:, 1:].reshape(-1))
                    (loss * (part.numel() / nb)).backward()
                elif teacher is not None:
                    logp = F.log_softmax(logits.reshape(-1, V), dim=-1)
                    y = chunks[:, 1:].reshape(-1)
                    tq = teacher[(part.view(-1, 1) + offsets[:, 1:]).reshape(-1)].float()
                    ce = -logp.gather(1, y.view(-1, 1)).squeeze(1)
                    soft = ((ce + teacher_beta * -(tq * logp).sum(-1)) / (1.0 + teacher_beta)).mean()
                    (soft * (part.numel() / nb)).backward()
                    soft_sum += soft.detach() * part.numel()
                    loss = ce.detach().mean()
                else:
                    logp = F.log_softmax(logits.reshape(-1, V), dim=-1)
                    y = chunks[:, 1:].reshape(-1)
                    soft_ev = -(targets.dist(part, y, logp.dtype) * logp).sum(-1)
                    soft = soft_ev.mean()
                    if weighted:
                        ((soft_ev.view(-1, T) * targets.weights(part, T)).sum() / Wb).backward()
                    else:
                        (soft * (part.numel() / nb)).backward()
                    soft_sum += soft.detach() * part.numel()
                    loss = F.nll_loss(logp.detach(), y)
                loss_sum += loss.detach() * part.numel()
            opt.step()
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        train_wall += dt
        nll = loss_sum.item() / bounds[-1]
        soft_note = (f" soft_ce={soft_sum.item() / bounds[-1]:.6f}" if targets is not None
                     else f" coupled={soft_sum.item() / bounds[-1]:.6f}" if teacher is not None else "")
        print(f"  epoch {epoch} train_nll={nll:.6f} train_ppl={math.exp(nll):.4f}{soft_note} steps={step} "
              f"lr_end={cur:.6g} wall={dt:.2f}s", flush=True)
        if epoch in checkpoints:
            p = epoch_checkpoint_path(save_path, epoch)
            save_model(p, mc, model)
            print(f"  train-epoch-checkpoint: epoch={epoch} train_wall_seconds={train_wall:.6f} path={p}", flush=True)

    save_model(save_path, mc, model)
    print(f"agpt_minibatch_train: done steps={step} train_wall_seconds={train_wall:.6f} saved {save_path}", flush=True)


if __name__ == "__main__":
    main()
