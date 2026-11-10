#!/usr/bin/env python3
"""How many moments does softmax over superposed occurrences need?

Measured on a trained checkpoint's layer-1 heads, with no training. In layer 1
a query and a key depend only on their characters, so the score of the query
at t against the occurrence at p = t - D is exactly

    s(t, p) = Re sum_j zq_j(x_t) * conj(zk_j(x_p)) * exp(i w_j D) / sqrt(hd).

Superposing the older occurrences of character c (offsets D >= window) into
one state keeps the sum of their RoPE points phi(D) = (exp(i w_j D))_j. That
gives sum_p s_p exactly, which is the linear readout. Softmax needs

    Z_c = sum_{p: x_p = c} exp(s_p),

and with the moment tensors of the cloud up to order m, the exact quantities
available are sum_p s_p^k for k <= m. (s^k is linear in phi^{(x)k}; this is
checked numerically below for k = 2 and 3.) So any degree-m polynomial P gives
sum_p P(s_p) exactly. Three order-m estimators are scored:

    taylor0:   Z_c ~ sum_p T_m(s_p)                      T_m = degree-m Taylor of exp at 0
    centred:   Z_c ~ exp(sbar_c) sum_p T_m(s_p - sbar_c) sbar_c = mean score (order 1)
    chebyshev: Z_c ~ sum_p C_m(s_p)                      C_m = degree-m Chebyshev interpolant
                                                         of exp on [lo, hi]

For chebyshev, [lo, hi] is the range of the score over every offset in
[window, max_offset) for this (query char, key char, head). It is known before
any occurrence is seen, so C_m is a fixed polynomial and the estimate needs only
the order-<=m moments. Chebyshev interpolation is near-best in uniform error.

Only even m is used, where T_m > 0. The metric is the KL divergence, for each
(query, head), between the exact attention distribution and the approximate
one. That distribution covers the window positions (kept exact) plus one group
per character for the older occurrences. Since values in layer 1 depend only
on the character, this distribution is what the head's output depends on.

Usage:
    python3 src/tools/agpt_superposition_moments.py \\
        --model rnd/.../checkpoint.model --window 16 --max-offset 64 --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from agpt_ppl import build_rope_cache, build_vocab, load_model, rope_apply  # noqa: E402

SPLIT = Path('data/.splits/4fa9aec1db6b3aea')
ORDERS = [2, 4, 6, 8, 10, 12, 16, 20, 24]


def taylor_exp(x, m):
    """Degree-m Taylor polynomial of exp at 0, elementwise (float64)."""
    out = torch.ones_like(x)
    term = torch.ones_like(x)
    for k in range(1, m + 1):
        term = term * x / k
        out = out + term
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--window', type=int, default=16)
    ap.add_argument('--max-offset', type=int, default=64, help='older offsets D in [window, max_offset)')
    ap.add_argument('--query-start', type=int, default=1024)
    ap.add_argument('--query-stride', type=int, default=23)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    cfg, sd = load_model(args.model)
    D, H = cfg['d_model'], cfg['n_heads']
    HD = D // H
    J = HD // 2
    vocab, V = build_vocab('data/input.txt')
    assert V == cfg['vocab_size']

    # layer-1 queries and keys per character (no positional embedding; RoPE only)
    emb = sd['token_emb'].double()
    ln = torch.nn.functional.layer_norm(emb, (D,), sd['l0.ln1_g'].view(-1).double(),
                                        sd['l0.ln1_b'].view(-1).double(), eps=1e-5)
    q = (ln @ sd['l0.wq_w'].double() + sd['l0.wq_b'].double()).view(V, H, HD)
    k = (ln @ sd['l0.wk_w'].double() + sd['l0.wk_b'].double()).view(V, H, HD)
    zq = torch.complex(q[..., 0::2], q[..., 1::2])  # (V, H, J)
    zk = torch.complex(k[..., 0::2], k[..., 1::2])
    omega = 10000.0 ** (-2.0 * torch.arange(J, dtype=torch.float64) / HD)
    A = zq[:, None] * zk.conj()[None, :] / math.sqrt(HD)  # (Vq, Vk, H, J): a = q conj(k)/sqrt(hd)

    def scores(a_char, k_chars, offs):
        """s for one query char against key chars at offsets: (H, n)."""
        ph = torch.polar(torch.ones(len(offs), J, dtype=torch.float64), offs[:, None].double() * omega)
        return torch.einsum('nhj,nj->hn', A[a_char][k_chars], ph).real

    # --- check 1: factored score == the head's actual RoPE logit, offsets 0..window-1
    cos, sin = build_rope_cache(args.window, HD)
    qa, kb = 7, 23
    qr = rope_apply(q[qa].float().view(H, 1, HD), cos[-1:].view(1, 1, HD), sin[-1:].view(1, 1, HD))
    kr = rope_apply(k[kb].float().view(H, 1, HD).expand(H, args.window, HD), cos.view(1, args.window, HD),
                    sin.view(1, args.window, HD))
    actual = (qr * kr).sum(-1) / math.sqrt(HD)  # (H, window): query at window-1, keys at 0..window-1
    offs = torch.arange(args.window - 1, -1, -1)
    fact = scores(qa, torch.full((args.window,), kb), offs)
    err_rope = float((actual.double() - fact).abs().max())

    # --- check 2: sum_p s_p^k from the order-k moment tensor of the real RoPE points
    rng_offs = torch.randint(args.window, 900, (40,))
    a = A[qa, kb, 0]  # (J,) complex
    a_r = torch.cat([a.real, -a.imag])  # Re(a e^{i w D}) = a_r . [cos wD, sin wD]
    phi = torch.cat([torch.cos(rng_offs[:, None].double() * omega), torch.sin(rng_offs[:, None].double() * omega)], 1)
    s_direct = phi @ a_r
    err_mom = 0.0
    for kk in (2, 3):
        M = phi
        for _ in range(kk - 1):
            M = torch.einsum('n...,nj->n...j', M, phi)
        M = M.sum(0)  # moment tensor
        via = M
        for _ in range(kk):
            via = via @ a_r
        err_mom = max(err_mom, float(abs(via - (s_direct ** kk).sum()) / (s_direct ** kk).abs().sum()))

    # --- the measurement
    chunks = []
    for p in sorted((SPLIT / 'heldout_chunks').glob('chunk_*.txt')):
        txt = p.read_text(encoding='utf-8', errors='replace')
        chunks.append(torch.tensor([vocab[c] for c in txt]))

    W = args.window
    ESTS = ('taylor0', 'centred', 'chebyshev')

    # chebyshev: per (query char, key char, head), the score range over all older offsets
    # and the degree-m interpolant of exp on it, as coefficients of T_n(u), u in [-1, 1]
    all_old = torch.arange(W, args.max_offset)
    ph_old = torch.polar(torch.ones(len(all_old), J, dtype=torch.float64), all_old[:, None].double() * omega)
    s_all = torch.einsum('qkhj,nj->qkhn', A, ph_old).real        # (Vq, Vk, H, n_old)
    lo, hi = s_all.min(-1).values, s_all.max(-1).values
    mid, half = (hi + lo) / 2, ((hi - lo) / 2).clamp_min(1e-12)
    cheb = {}
    for m in ORDERS:
        kk = torch.arange(m + 1, dtype=torch.float64)
        theta = math.pi * (kk + 0.5) / (m + 1)                    # Chebyshev nodes u_k = cos(theta_k)
        f = torch.exp(half[..., None] * torch.cos(theta))        # exp(half * u_k), mid factored out
        c = (2.0 / (m + 1)) * f @ torch.cos(kk[:, None] * theta[None, :]).T  # c_n, n = 0..m
        c[..., 0] /= 2
        cheb[m] = c                                              # (Vq, Vk, H, m+1)

    # --- check 3: the largest-order interpolant equals exp at its own nodes (query char qa), as
    # uniform error relative to exp(hi). (Relative to exp(lo) it is limited by float64 cancellation
    # once the interval is wide: the coefficients are ~exp(hi), the smallest values exp(lo).)
    mt = ORDERS[-1]
    uu = torch.cos(math.pi * (torch.arange(mt + 1, dtype=torch.float64) + 0.5) / (mt + 1))
    Tg = torch.cos(torch.arange(mt + 1, dtype=torch.float64) * torch.arccos(uu)[:, None])  # (nodes, mt+1)
    approx = torch.exp(mid[qa][..., None]) * (cheb[mt][qa] @ Tg.T)                    # (Vk, H, nodes)
    truth = torch.exp(mid[qa][..., None] + half[qa][..., None] * uu)
    err_cheb = float(((approx - truth).abs() / torch.exp(hi[qa])[..., None]).max())

    kl = {(est, m): [] for est in ESTS for m in ORDERS}
    older_mass, logit_range, win_entropy, win_maxw, n_occ = [], [], [], [], []
    for x in chunks:
        n = x.numel()
        for t in range(max(args.query_start, args.max_offset), n, args.query_stride):
            a_char = int(x[t])
            win_offs = torch.arange(0, W)
            older_offs = torch.arange(W, args.max_offset)
            s_win = scores(a_char, x[t - win_offs], win_offs)          # (H, W)
            older_chars = x[t - older_offs]
            s_old = scores(a_char, older_chars, older_offs)           # (H, n_old)
            # exact attention over window positions + per-character groups of older occurrences
            onehot = torch.nn.functional.one_hot(older_chars, V).double()  # (n_old, V)
            present = onehot.sum(0) > 0
            Zc = torch.exp(s_old) @ onehot                            # (H, V)
            logits_exact = torch.cat([s_win, torch.log(Zc[:, present])], 1)
            lp_exact = torch.log_softmax(logits_exact, 1)
            p_exact = lp_exact.exp()
            cnt = onehot.sum(0)[present]
            sbar = (s_old @ onehot)[:, present] / cnt
            mid_p = mid[a_char][older_chars].T                         # (H, n_old)
            half_p = half[a_char][older_chars].T
            acos_u = torch.arccos(((s_old - mid_p) / half_p).clamp(-1, 1))
            for m in ORDERS:
                Z0 = (taylor_exp(s_old, m) @ onehot)[:, present]
                dev = s_old - (sbar @ onehot[:, present].T)          # s_p - sbar_{c(p)}
                Zc_ = (taylor_exp(dev, m) @ onehot)[:, present] * torch.exp(sbar)
                cp = cheb[m][a_char][older_chars].permute(1, 0, 2)    # (H, n_old, m+1)
                Tn = torch.cos(torch.arange(m + 1, dtype=torch.float64) * acos_u[..., None])
                Zch = ((torch.exp(mid_p) * (cp * Tn).sum(-1)) @ onehot)[:, present]
                for est, Zhat in (('taylor0', Z0), ('centred', Zc_), ('chebyshev', Zch)):
                    la = torch.cat([s_win, torch.log(Zhat.clamp_min(1e-300))], 1)
                    lp = torch.log_softmax(la, 1)
                    kl[(est, m)].append((p_exact * (lp_exact - lp)).sum(1))
            older_mass.append(p_exact[:, W:].sum(1))
            logit_range.append(s_old.abs().max(1).values)
            pw = torch.softmax(s_win, 1)
            win_entropy.append(-(pw * pw.log()).sum(1))
            win_maxw.append(pw.max(1).values)
            n_occ.append(float(len(older_offs)))

    def pct(xs, qs=(0.5, 0.9, 0.99)):
        t = torch.cat(xs) if isinstance(xs[0], torch.Tensor) else torch.tensor(xs)
        t = t.flatten().double()
        return {f'p{int(q * 100)}': float(torch.quantile(t, q)) for q in qs} | {'mean': float(t.mean())}

    per_head = lambda xs: [float(torch.stack(xs)[:, h].mean()) for h in range(H)]  # noqa: E731
    res = {
        'model': args.model, 'model_seq_len': cfg['seq_len'], 'window': W, 'max_offset': args.max_offset,
        'queries': len(older_mass), 'heads': H, 'freqs_per_head': J,
        'checks': {'factored_vs_rope_logit_max_abs_err': err_rope, 'moment_tensor_rel_err_k2_k3': err_mom,
                   f'chebyshev_m{ORDERS[-1]}_err_at_nodes_rel_to_interval_max': err_cheb},
        'older_attention_mass': pct(older_mass) | {'per_head_mean': per_head(older_mass)},
        'older_logit_range_max_abs_s': pct(logit_range) | {'per_head_mean': per_head(logit_range)},
        'window_entropy_nats': pct(win_entropy) | {'uniform': math.log(W), 'per_head_mean': per_head(win_entropy)},
        'window_max_weight': pct(win_maxw),
        'kl_exact_vs_order_m': {est: {str(m): pct(kl[(est, m)]) for m in ORDERS} for est in ESTS},
    }
    for est in ESTS:
        need = None
        for m in ORDERS:
            if res['kl_exact_vs_order_m'][est][str(m)]['p99'] < 1e-3:
                need = m
                break
        res[f'order_needed_p99_kl_below_1e-3_{est}'] = need
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k != 'kl_exact_vs_order_m'}, indent=1))
    for est in ESTS:
        print(est, ' '.join(f"m={m}:p50={res['kl_exact_vs_order_m'][est][str(m)]['p50']:.2e},"
                            f"p99={res['kl_exact_vs_order_m'][est][str(m)]['p99']:.2e}" for m in ORDERS))


if __name__ == '__main__':
    main()
