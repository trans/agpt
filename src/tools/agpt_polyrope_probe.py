#!/usr/bin/env python3
"""PolyRoPE probe: an exact 16-character softmax window plus a far channel whose
content is only the identity of each symbol (a character, or a word-like unit),
read through damped harmonic signatures of where each symbol occurred.

Question (the user's PolyRoPE design, window form, 2026-11-12): if the model is
told, for every symbol, a harmonic signature of its positions in the last R
characters, can it learn to read that for prediction, and how much of the gain
of exact attention over the same reach does it recover? Does the answer differ
between characters and units?

The harness is rnd/far-channel-probe's: no trie, a plain minibatch-Adam
transformer with the AGPT d64/L2 layout (pre-LN, RoPE with interleaved pairs,
ReLU FFN, untied output head). Layer 1 attends over offsets 0-15. The arms
differ only in layer 2:

  w16     softmax over offsets 0-15 (the baseline)
  exact   softmax over offsets 0 .. R-1 (the upper bound)
  hidden  softmax over 0-15, plus a far channel over offsets 16 .. R-1 whose
          keys and values come from the layer's input (far-channel-probe's
          cq arm, at reach R)
  char    the same far channel, but keys and values come only from the
          character at each position (one-hot content: a learned embedding of
          the symbol, nothing from context)
  unit    the same, with one entry per complete unit, at the unit's last
          character; units are cut as mobius's UnitCuts cuts them (where
          letters, punctuation and whitespace meet; an apostrophe after a
          letter is a letter; a single space joins the unit after it), and
          the 4,095 commonest units of the training text get their own ids

The far channel, per head, for the query at t and an entry at p, Δ = t - p:

    a[t, p] = phi(qf_t) . phi(k_p) * w_t(Δ),        16 <= Δ < R
    far_t   = sum_p a[t, p] v_p / sum_p a[t, p]
    w_t(Δ)  = |sum_j b_{t,j} lambda_j^(Δ - 16)|^2,   lambda_j = exp(-1/P_j) exp(2 pi i / P_j)

with P = 8, 16, ..., 1024 (8 constant-Q atoms, one octave above the far-channel
probe's, for an 8x longer reach). For `char` and `unit`, k_p and v_p depend only
on the symbol s_p, so the exact recurrent state can be kept per symbol type:

    H_v(mu) = sum over entries p of symbol v, 16 <= t - p < R, of mu^(t - p - 16)

for the 64 products mu = lambda_j conj(lambda_l). That is the user's PolyRoPE
signature (damped, windowed). The query reads it as

    G_t(v) = sum_mu b_{t,j} conj(b_{t,l}) H_v(mu)   (the kernel-weighted count of v)
    far_t  = sum_v G_t(v) phi(qf_t).phi(k_v) v_v / sum_v G_t(v) phi(qf_t).phi(k_v)

Training uses the parallel form, the same sum; a start-up check compares them.

Training: blocks of --block characters (default 2048) from the training split,
every position scored. Evaluation is DIAGNOSTIC, not lm-eval byte PPL: each
target of the 10 held-out chunks is scored once, with at least 1,040 preceding
characters where the chunk has them, whatever the arm's reach, so every run of
this probe is scored on the same targets. Buckets by the target's position in
its chunk: 1-30, 31-1038, and 1039 to the end (the main comparison).

Usage:
    python3 src/tools/agpt_polyrope_probe.py --arm unit --steps 30000 \\
        --out rnd/polyrope-probe/RUN_ID
    python3 src/tools/agpt_polyrope_probe.py --arm exact --reach 128 \\
        --eval-only rnd/far-channel-probe/RUN/model.pt --out OUT   # rescore
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SPLIT = Path('data/.splits/4fa9aec1db6b3aea')
WINDOW = 16
EVAL_REACH = 1024
RF_EVAL = (WINDOW - 1) + (EVAL_REACH - 1)     # 1038
CTX_MIN = RF_EVAL + 2                          # 1040
BUCKETS = [(1, 31), (31, RF_EVAL + 1), (RF_EVAL + 1, 10 ** 9)]
UNIT_CAPACITY = 4096                           # id 0: every other unit; 1..4095: the commonest


# ----------------------------------------------------------------- data

def build_vocab(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    return {c: i for i, c in enumerate(sorted(set(text)))}


def read(path):
    return Path(path).read_text(encoding='utf-8', errors='replace')


def encoder(vocab):
    """Text -> ids. Byte lookup when every symbol is one byte (Shakespeare, the
    PG-19 split); the same ids as the per-character dict, without a 20M-entry list."""
    if max(map(ord, vocab)) < 256:
        lut = torch.full((256,), -1, dtype=torch.long)
        for c, i in vocab.items():
            lut[ord(c)] = i

        def enc(s):
            ids = lut[torch.frombuffer(bytearray(s.encode('latin-1')), dtype=torch.uint8).long()]
            if (ids < 0).any():
                raise SystemExit('text has a character outside the vocabulary')
            return ids
        return enc
    return lambda s: torch.tensor([vocab[c] for c in s], dtype=torch.long)


LETTER, SPACE, PUNCT, APOS, SPACE_CHAR = range(5)


def raw_kind(c):
    if c.isalnum():
        return LETTER
    if c == ' ':
        return SPACE_CHAR
    if c.isspace():
        return SPACE
    if c in "'’":
        return APOS
    return PUNCT


def unit_ends(text):
    """mobius's UnitCuts (src/mobius/unit_bytes.cr), ported. Returns, for each
    position p, the complete unit whose last character is p, or None. A unit is
    complete once the character after it is read, so the last one is not."""
    raw = [raw_kind(c) for c in text]
    n = len(text)

    def kind(i):
        r = raw[i]
        if r == LETTER:
            return LETTER
        if r in (SPACE, SPACE_CHAR):
            return SPACE
        if r == APOS:
            return LETTER if i > 0 and raw[i - 1] == LETTER else PUNCT
        return PUNCT

    kinds = [kind(i) for i in range(n)]
    ends = [None] * n
    start = 0
    for g in range(1, n):                       # gap g lies between g-1 and g
        before, after = kinds[g - 1], kinds[g]
        if before == after:
            continue
        single_space = raw[g - 1] == SPACE_CHAR and (g < 2 or kinds[g - 2] != SPACE)
        if single_space and after != SPACE:
            continue
        ends[g - 1] = text[start:g]
        start = g
    return ends


def unit_table(train_text, capacity=UNIT_CAPACITY):
    counts = Counter(u for u in unit_ends(train_text) if u is not None)
    ids = {u: i + 1 for i, (u, _) in enumerate(counts.most_common(capacity - 1))}
    return ids, counts


def unit_symbols(text, ids, none_id):
    """Per position: the unit id ending there (0 for a rare unit), or none_id."""
    return torch.tensor([none_id if u is None else ids.get(u, 0) for u in unit_ends(text)],
                        dtype=torch.long)


# ---------------------------------------------------------------- model

def rope_cache(T, head_dim, base, device):
    half = head_dim // 2
    pos = torch.arange(T, dtype=torch.float32).view(-1, 1)
    i = torch.arange(half, dtype=torch.float32).view(1, -1)
    theta = pos / (base ** (2.0 * i / head_dim))
    cos = torch.repeat_interleave(torch.cos(theta), 2, dim=1)
    sin = torch.repeat_interleave(torch.sin(theta), 2, dim=1)
    return cos.to(device), sin.to(device)


def rope_apply(x, cos, sin):
    xe, xo = x[..., 0::2], x[..., 1::2]
    c, s = cos[..., 0::2], sin[..., 0::2]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


def span_mask(T, lo, hi, device):
    """(T, T) bool: key p visible to query t iff lo <= t - p < hi."""
    d = torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)[None, :]
    return (d >= lo) & (d < hi)


class FarChannel(nn.Module):
    """content: 'hidden' (keys from the layer input, values the layer's) or
    'symbol' (keys and values from a learned embedding of the symbol only)."""

    def __init__(self, D, H, content, reach, n_sym=0, d_sym=16, n_atoms=8):
        super().__init__()
        self.H, self.content, self.A, self.reach = H, content, n_atoms, reach
        self.n_sym = n_sym
        self.qf = nn.Linear(D, D)
        if content == 'hidden':
            self.kf = nn.Linear(D, D)
        else:
            self.emb = nn.Embedding(n_sym + 1, d_sym)       # row n_sym: no entry (masked)
            self.kf, self.vf = nn.Linear(d_sym, D), nn.Linear(d_sym, D)
        n = reach - WINDOW
        d = torch.arange(n, dtype=torch.float64)
        P = 2.0 ** torch.arange(3, n_atoms + 3, dtype=torch.float64)            # 8 .. 1024
        lam = torch.polar(torch.exp(-1.0 / P), 2 * math.pi / P)
        self.coef = nn.Linear(D, H * n_atoms * 2)
        self.register_buffer('basis', (lam[:, None] ** d[None, :]).to(torch.complex64), persistent=False)
        self.register_buffer('lam', lam, persistent=False)
        self.atoms = {'period': P.tolist(), 'decay_time': P.tolist()}
        self.gate = nn.Parameter(torch.ones(H))

    def profile(self, h):
        """Kernel over d = Δ - 16 for each query: (B, H, T, n)."""
        B, T, _ = h.shape
        c = self.coef(h).view(B, T, self.H, self.A, 2).transpose(1, 2)
        b = torch.complex(c[..., 0].contiguous(), c[..., 1].contiguous())
        amp = b @ self.basis
        return amp.real ** 2 + amp.imag ** 2

    def keys_values(self, h, v, sym):
        B, T, D = h.shape
        H, HD = self.H, D // self.H
        if self.content == 'hidden':
            return F.elu(self.kf(h).view(B, T, H, HD).transpose(1, 2)) + 1, v
        e = self.emb(sym)
        k = F.elu(self.kf(e).view(B, T, H, HD).transpose(1, 2)) + 1
        return k, self.vf(e).view(B, T, H, HD).transpose(1, 2)

    def weights(self, h, v, sym, valid, didx):
        """a[b, h, t, p] before normalization, and the values."""
        B, T, D = h.shape
        H, HD = self.H, D // self.H
        q = F.elu(self.qf(h).view(B, T, H, HD).transpose(1, 2)) + 1
        k, val = self.keys_values(h, v, sym)
        prof = self.profile(h)                                                # (B, H, T, n)
        w = torch.gather(prof, 3, didx[None, None].expand(B, H, T, T)) * valid
        return (q @ k.transpose(-1, -2)) * w, val, q

    def forward(self, h, v, sym, valid, didx):
        a, val, _ = self.weights(h, v, sym, valid, didx)
        far = (a @ val) / (a.sum(-1, keepdim=True) + 1e-6)
        return self.gate.view(1, -1, 1, 1) * far


class Block(nn.Module):
    def __init__(self, D, H, Fd, far):
        super().__init__()
        self.H = H
        self.ln1 = nn.LayerNorm(D)
        self.wq, self.wk = nn.Linear(D, D), nn.Linear(D, D)
        self.wv, self.wo = nn.Linear(D, D), nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.l1, self.l2 = nn.Linear(D, Fd), nn.Linear(Fd, D)
        self.far = far

    def forward(self, x, cos, sin, attn_mask, far_ctx):
        B, T, D = x.shape
        H, HD = self.H, D // self.H
        h = self.ln1(x)
        q = self.wq(h).view(B, T, H, HD).transpose(1, 2)
        k = self.wk(h).view(B, T, H, HD).transpose(1, 2)
        v = self.wv(h).view(B, T, H, HD).transpose(1, 2)
        att = F.scaled_dot_product_attention(
            rope_apply(q, cos, sin), rope_apply(k, cos, sin), v, attn_mask=attn_mask)
        if self.far is not None:
            att = att + self.far(h, v, *far_ctx)
        x = x + self.wo(att.transpose(1, 2).reshape(B, T, D))
        return x + self.l2(F.relu(self.l1(self.ln2(x))))


class PolyLM(nn.Module):
    def __init__(self, V, D, H, L, Fd, T, base, arm, reach, n_units, device):
        super().__init__()
        self.arm, self.reach = arm, reach
        far = None
        if arm == 'hidden':
            far = FarChannel(D, H, 'hidden', reach)
        elif arm == 'char':
            far = FarChannel(D, H, 'symbol', reach, n_sym=V)
        elif arm == 'unit':
            far = FarChannel(D, H, 'symbol', reach, n_sym=n_units)
        self.spans = [WINDOW] * (L - 1) + [reach if arm == 'exact' else WINDOW]
        self.tok = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block(D, H, Fd, far if l == L - 1 else None) for l in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
        cos, sin = rope_cache(T, D // H, base, device)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def init_weights(self, seed):
        """N(0, 0.02) weights, zero biases, unit LayerNorm; far gates stay 1."""
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for m in self.modules():  # deterministic order
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.02)
                    if getattr(m, 'bias', None) is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.weight.fill_(1.0)
                    m.bias.zero_()

    def far_ctx(self, T, sym, dev):
        far = self.blocks[-1].far
        if far is None:
            return (None, None, None)
        valid = span_mask(T, WINDOW, self.reach, dev).float()[None, None]
        if far.content == 'symbol':
            valid = valid * (sym != far.n_sym).float()[:, None, None, :]       # entries only
        d = torch.arange(T, device=dev)[:, None] - torch.arange(T, device=dev)[None, :]
        didx = (d - WINDOW).clamp(0, self.reach - WINDOW - 1)
        return (sym, valid, didx)

    def forward(self, ids, sym):
        T = ids.size(1)
        dev = ids.device
        x = self.tok(ids)
        cos, sin = self.cos[:T], self.sin[:T]
        ctx = self.far_ctx(T, sym, dev)
        for b, span in zip(self.blocks, self.spans):
            x = b(x, cos, sin, span_mask(T, 0, span, dev), ctx)
        return self.head(self.ln_f(x))


# --------------------------------------------------------------- checks

@torch.no_grad()
def check_recurrent(model, ids, sym, dev):
    """Far channel, parallel form against an explicit recurrence (head 0, batch 0,
    float64). For symbol content the recurrence keeps PolyRoPE's per-symbol
    signatures H_v(mu); for hidden content, sum_p mu^d k_p v_p^T as before.
    Returns the max relative error of far_t."""
    far = model.blocks[-1].far
    if far is None:
        return None
    T = ids.size(1)
    blk = model.blocks[-1]
    x = model.tok(ids)
    cos, sin = model.cos[:T], model.sin[:T]
    for b, span in zip(model.blocks[:-1], model.spans[:-1]):
        x = b(x, cos, sin, span_mask(T, 0, span, dev), (None, None, None))
    h = blk.ln1(x)
    D = h.size(-1)
    H, HD = blk.H, D // blk.H
    v = blk.wv(h).view(1, T, H, HD).transpose(1, 2)
    s_, valid, didx = model.far_ctx(T, sym, dev)
    a, val, q = far.weights(h, v, sym, valid, didx)
    par = ((a @ val) / (a.sum(-1, keepdim=True) + 1e-6))[0, 0].double()      # (T, HD)
    lam = far.lam.to(dev)
    mu = (lam[:, None] * lam.conj()[None, :]).reshape(-1)                      # (J*J,)
    c = far.coef(h).view(1, T, H, far.A, 2)[0, :, 0].double()
    bt = torch.complex(c[..., 0], c[..., 1])                                   # (T, J)
    coef = (bt[:, :, None] * bt.conj()[:, None, :]).reshape(T, -1)             # (T, J*J)
    q0 = q[0, 0].double().to(torch.complex128)
    n_hi = model.reach - WINDOW
    R = model.reach
    worst = 0.0
    if far.content == 'symbol':
        n_sym = far.n_sym
        E = far.emb.weight.double()
        kv = F.elu(F.linear(E, far.kf.weight.double(), far.kf.bias.double())) + 1
        kv = kv.view(-1, H, HD)[:, 0].to(torch.complex128)                              # (S+1, HD)
        vv = F.linear(E, far.vf.weight.double(), far.vf.bias.double()).view(-1, H, HD)[:, 0]
        Hs = torch.zeros(n_sym + 1, mu.numel(), dtype=torch.complex128, device=dev)   # H_v(mu)
        s0 = sym[0]
        for t in range(T):
            if t >= WINDOW and int(s0[t - WINDOW]) != n_sym:
                Hs[int(s0[t - WINDOW])] += 1
            if t >= R and int(s0[t - R]) != n_sym:
                Hs[int(s0[t - R])] -= mu ** n_hi
            if t >= WINDOW:
                G = (Hs @ coef[t]).real                       # kernel-weighted count of each symbol
                av = (kv @ q0[t]).real                        # phi(q_t) . phi(k_v)
                wgt = G * av
                rec = (wgt @ vv) / (wgt.sum() + 1e-6)
                worst = max(worst, float((rec - par[t]).abs().max() / par[t].abs().max().clamp_min(1e-12)))
            Hs *= mu[None, :]
        return worst
    k0, v0 = far.keys_values(h, v, sym)
    kc, vc = k0[0, 0].double().to(torch.complex128), v0[0, 0].double().to(torch.complex128)
    S = torch.zeros(mu.numel(), HD, HD, dtype=torch.complex128, device=dev)
    Z = torch.zeros(mu.numel(), HD, dtype=torch.complex128, device=dev)
    for t in range(T):
        if t >= WINDOW:
            p = t - WINDOW
            S += kc[p][None, :, None] * vc[p][None, None, :]
            Z += kc[p][None, :]
        if t >= R:
            p = t - R
            S -= (mu ** n_hi)[:, None, None] * kc[p][None, :, None] * vc[p][None, None, :]
            Z -= (mu ** n_hi)[:, None] * kc[p][None, :]
        if t >= WINDOW:
            num = torch.einsum('a,d,ade->e', coef[t], q0[t], S).real
            den = torch.einsum('a,d,ad->', coef[t], q0[t], Z).real
            rec = num / (den + 1e-6)
            worst = max(worst, float((rec - par[t]).abs().max() / par[t].abs().max().clamp_min(1e-12)))
        S *= mu[:, None, None]
        Z *= mu[:, None]
    return worst


# ----------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, chunks, T, batch):
    """chunks: list of (ids, sym). Each target x[i] scored once. Windows of T inputs
    start every T - CTX_MIN characters; a window scores its targets at positions
    >= CTX_MIN (the first window scores all of them). A last window ending at the
    chunk end scores whatever the full windows left."""
    model.eval()
    nll = torch.zeros(len(BUCKETS), dtype=torch.float64)
    cnt = torch.zeros(len(BUCKETS), dtype=torch.float64)
    stride = T - CTX_MIN
    for x, sy in chunks:
        n = x.numel()
        dev = x.device
        wins = [(s0, 0 if s0 == 0 else CTX_MIN) for s0 in range(0, n - T, stride)]
        last = wins[-1][0] + T
        if last < n - 1:
            s_f = n - 1 - T
            wins.append((s_f, last - s_f))
        for i in range(0, len(wins), batch):
            s = torch.tensor([w[0] for w in wins[i:i + batch]], device=dev)
            j0 = torch.tensor([w[1] for w in wins[i:i + batch]], device=dev)
            idx = s[:, None] + torch.arange(T + 1, device=dev)[None, :]
            win, swin = x[idx], sy[idx]
            lp = F.cross_entropy(model(win[:, :T], swin[:, :T]).transpose(1, 2), win[:, 1:], reduction='none')
            keep = torch.arange(T, device=dev)[None, :] >= j0[:, None]
            tpos = s[:, None] + 1 + torch.arange(T, device=dev)[None, :]
            tp, ls = tpos[keep], lp[keep]
            for bi, (lo, hi) in enumerate(BUCKETS):
                sel = (tp >= lo) & (tp < hi)
                nll[bi] += ls[sel].double().sum().cpu()
                cnt[bi] += sel.sum().cpu()
    model.train()
    total = float(nll.sum() / cnt.sum())
    by = {f'{lo}-{hi - 1 if hi < 10 ** 9 else "end"}': float(nll[b] / cnt[b]) for b, (lo, hi) in enumerate(BUCKETS)}
    return total, by, int(cnt.sum())


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', choices=['w16', 'exact', 'hidden', 'char', 'unit'], required=True)
    ap.add_argument('--reach', type=int, default=1024, help='layer 2 reaches offsets up to reach - 1')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch-size', type=int, default=4, help='blocks of --block characters')
    ap.add_argument('--block', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--min-lr-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rope-base', type=float, default=10000.0)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-layers', type=int, default=2)
    ap.add_argument('--d-ff', type=int, default=256)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--eval-batch', type=int, default=8)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--device', default=None)
    ap.add_argument('--eval-only', default=None, help='score this checkpoint (state dict) and exit')
    ap.add_argument('--data', default=str(SPLIT),
                    help='split directory: train_corpus.txt and heldout_chunks/chunk_*.txt')
    ap.add_argument('--vocab-file', default='data/input.txt', help='the vocabulary is its set of characters')
    ap.add_argument('--piece-chars', type=int, default=0,
                    help='the training text is pieces of this many chars from different books (0: one text)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / ('eval.log' if args.eval_only else 'train.log'), 'w')

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()

    if args.reach <= WINDOW:
        raise SystemExit('reach must exceed the window')
    if args.block <= CTX_MIN:
        raise SystemExit(f'block must exceed {CTX_MIN}, the evaluation context')
    data = Path(args.data)
    vocab = build_vocab(args.vocab_file)
    V = len(vocab)
    train_text = read(data / 'train_corpus.txt')
    chunk_texts = [read(p) for p in sorted((data / 'heldout_chunks').glob('chunk_*.txt'))]
    enc = encoder(vocab)
    x_train = enc(train_text)
    if args.arm == 'unit':
        units, unit_counts = unit_table(train_text)
        none_id = UNIT_CAPACITY
        sym_of = lambda s: unit_symbols(s, units, none_id)
        s_train = sym_of(train_text)
    else:
        sym_of = enc                       # characters (the char arm reads them; others ignore)
        s_train = x_train
    del train_text
    chunks = [(enc(s), sym_of(s)) for s in chunk_texts]
    T = args.block
    N = x_train.numel()
    P = args.piece_chars or N              # training blocks never cross a piece boundary
    if N % P or P <= T + 1:
        raise SystemExit(f'--piece-chars {P} must divide the {N} training chars and exceed the block')
    eval_desc = (f'diagnostic: multi-chunk held-out ({len(chunks)} x {len(chunk_texts[0])} chars, '
                 f'{data}), each target once, with >= {CTX_MIN} preceding chars where the chunk has them; '
                 'by_context buckets are the target position in its chunk; NOT lm-eval byte PPL')

    model = PolyLM(V, args.d_model, args.n_heads, args.n_layers, args.d_ff, T,
                   args.rope_base, args.arm, args.reach, UNIT_CAPACITY, dev)
    model.init_weights(args.seed)
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    chunks_dev = [(a.to(dev), b.to(dev)) for a, b in chunks]

    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=dev))
        total, by, n = evaluate(model, chunks_dev, T, args.eval_batch)
        res = {'arm': args.arm, 'reach': args.reach, 'checkpoint': args.eval_only,
               'heldout_nll': total, 'by_context': by, 'targets': n, 'params': n_params,
               'eval': 'diagnostic, polyrope-probe buckets (>= 1040 preceding chars where the chunk has them)'}
        (out / 'eval.json').write_text(json.dumps(res, indent=2) + '\n')
        log(f'eval-only {args.eval_only}: heldout_nll={total:.4f} targets={n} by_context=' +
            ' '.join(f'{k}:{v:.4f}' for k, v in by.items()))
        return

    # checks before training: receptive fields, and the far channel's recurrent form
    rf = (WINDOW - 1) + (args.reach - 1) if args.arm != 'w16' else 2 * (WINDOW - 1)
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(0, V, (2, T), generator=g)
        inv = {i: c for c, i in vocab.items()}

        def syms(ids_):
            return torch.stack([sym_of(''.join(inv[int(i)] for i in row)) for row in ids_])
        t = T - 1
        base_out = model(ids.to(dev), syms(ids).to(dev))[:, t]

        def changed(lo, hi):  # change the inputs at offsets lo .. hi from t
            ids2 = ids.clone()
            ids2[:, t - hi:t - lo + 1] = (ids2[:, t - hi:t - lo + 1] + 1) % V
            return float((model(ids2.to(dev), syms(ids2).to(dev))[:, t] - base_out).abs().max())
        far_rf = (WINDOW - 1) + (args.reach - 1)
        span_beyond = changed(far_rf + 1, T - 1) if far_rf + 1 <= T - 1 else None   # 0 for every arm
        span_far = changed(2 * (WINDOW - 1) + 1, min(far_rf, T - 1))                 # 0 for w16 only
        ck = torch.randint(0, V, (1, min(T, args.reach + 200)), generator=g)
        rec_err = check_recurrent(model, ck.to(dev), syms(ck).to(dev), dev)
    checks = {f'change_at_offsets_{far_rf + 1}_up_max_abs': span_beyond,
              f'change_at_offsets_{2 * (WINDOW - 1) + 1}_{far_rf}_max_abs': span_far,
              'far_recurrent_vs_parallel_max_rel_err': rec_err}
    log(f'checks: {checks}')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_dev, s_dev = x_train.to(dev), s_train.to(dev)
    rng = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    ar = torch.arange(T + 1, device=dev)
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'src/tools'],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        sha, dirty = None, None
    far = model.blocks[-1].far
    unit_info = None
    if args.arm == 'unit':
        tot = sum(unit_counts.values())
        unit_info = {'distinct_train_units': len(unit_counts), 'capacity': UNIT_CAPACITY,
                     'train_units': tot, 'mean_unit_chars': N / tot,
                     'coverage_of_ids': sum(n for _, n in unit_counts.most_common(UNIT_CAPACITY - 1)) / tot}
    (out / 'args.json').write_text(json.dumps({**vars(args), 'git_sha': sha, 'tools_dirty': dirty,
                                               'atoms': far.atoms if far is not None else None,
                                               'units': unit_info, 'train_chars': N,
                                               'params': n_params, 'checks': checks}, indent=2) + '\n')
    log(f'arm={args.arm} reach={args.reach} seed={args.seed} steps={args.steps} batch={args.batch_size}x{T} '
        f'lr={args.lr} params={n_params} device={dev} units={unit_info}')

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))

    curve = []
    ema = None
    t_start = time.time()
    for step in range(args.steps):
        for gp in opt.param_groups:
            gp['lr'] = lr_at(step)
        if P == N:
            s = torch.randint(0, N - T - 1, (args.batch_size,), generator=rng).to(dev)
        else:
            piece = torch.randint(0, N // P, (args.batch_size,), generator=rng)
            s = (piece * P + torch.randint(0, P - T - 1, (args.batch_size,), generator=rng)).to(dev)
        idx = s[:, None] + ar[None, :]
        xb, sb = x_dev[idx], s_dev[idx]
        logits = model(xb[:, :T], sb[:, :T])
        loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = loss.item()
        ema = lv if ema is None else 0.99 * ema + 0.01 * lv
        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            gate = '' if far is None else ' gate=' + ','.join(f'{g:+.3f}' for g in far.gate.tolist())
            log(f'step {step + 1} loss_ema={ema:.4f} lr={lr_at(step):.2e} {time.time() - t_start:.0f}s{gate}')
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            total, by, n = evaluate(model, chunks_dev, T, args.eval_batch)
            curve.append({'step': step + 1, 'heldout_nll': total, 'heldout_ppl': math.exp(total),
                          'by_context': by, 'train_loss_ema': ema})
            log(f'EVAL step {step + 1} heldout_nll={total:.4f} ppl={math.exp(total):.4f} targets={n} '
                f'by_context=' + ' '.join(f'{k}:{v:.4f}' for k, v in by.items()))
    final = curve[-1]
    metrics = {
        'arm': args.arm, 'reach': args.reach, 'seed': args.seed, 'steps': args.steps,
        'eval': eval_desc,
        'heldout_nll': final['heldout_nll'], 'heldout_ppl': final['heldout_ppl'],
        'by_context': final['by_context'], 'curve': curve,
        'gate': far.gate.tolist() if far is not None else None,
        'params': n_params, 'checks': checks,
        'train_seconds': time.time() - t_start,
        'ms_per_step': 1000 * (time.time() - t_start) / args.steps,
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    torch.save(model.state_dict(), out / 'model.pt')
    log(f'done: heldout_nll={final["heldout_nll"]:.4f} ppl={final["heldout_ppl"]:.4f}')


if __name__ == '__main__':
    main()
