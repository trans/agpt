"""Event-level AGPT step plans, replayed by the minibatch trainer
(rnd/one-pass-gap, trie-plan arms).

An event is (window start i, context length L): the prediction of token i+L
from tokens i..i+L-1, for 0 <= i < n_starts and 1 <= L <= T. A depth-T trie
holds the same events, each node carrying its occurrences. A trie unit is a
set of events, and its gradient is the sum of the events' instance gradients
(AGPT's per-unit SGD-equivalence). So training on a plan's steps through
ordinary windows gives the updates the trie trainer would give for that plan,
by a different route: each step's windows go through autograd with a loss mask
selecting the step's events, instead of through the trie.

Plans:

- pd(k): bin/agpt_train_v2's partition_depth k plan with experimental.train_crown.
  One unit per radix node whose edge spans depth k (the node's whole subtree,
  including any shallower part of its edge), and one per radix node whose edge
  ends above depth k. One unit per step, in a new random order each pass.
- packed(P, n_steps): adaptive pieces packed into equal steps. A window's events
  at depth >= p go to the piece of its p-character prefix, where p is the
  shallowest depth whose subtree holds at most P events; its shallower events go
  to the node of their own context, split by occurrence into near-equal chunks
  of at most P events. Pieces and chunks are packed at random into n_steps steps
  of equal event counts, repacked each pass.

plan_cost() counts the positions a trie trainer would compute for one pass of
a plan: in each step, every distinct context the step's events read (each
event's context and all its prefixes), once per step.
"""
from __future__ import annotations

import numpy as np

_HB = np.uint64(0x9E3779B97F4A7C15)
_C_PIECE = np.uint64(0xD6E8FEB86659FD93)
_C_CROWN = np.uint64(0xA0761D6478BD642F)
_C_NODE = np.uint64(0xE7037ED1A0B428DB)
_C_CHUNK = np.uint64(0x8EBC6AF09C88C6E3)
_C_COST = np.uint64(0x589965CC75374CC3)


def prefix_keys(tok: np.ndarray, n_starts: int, T: int) -> np.ndarray:
    """keys[d-1, i] = 64-bit hash of tok[i:i+d], d = 1..T."""
    keys = np.empty((T, n_starts), dtype=np.uint64)
    h = np.zeros(n_starts, dtype=np.uint64)
    t = tok.astype(np.uint64) + np.uint64(1)
    for d in range(T):
        h = h * _HB + t[d:d + n_starts]
        keys[d] = h
    return keys


def occurrences(keys: np.ndarray) -> np.ndarray:
    """occ[d-1, i] = number of window starts whose first d tokens equal window i's."""
    occ = np.empty(keys.shape, dtype=np.int32)
    for d in range(keys.shape[0]):
        _, inv, cnt = np.unique(keys[d], return_inverse=True, return_counts=True)
        occ[d] = cnt[inv]
    return occ


def n_children(keys: np.ndarray, d: int) -> np.ndarray:
    """Per window, how many distinct next tokens follow its depth-d context."""
    _, idx = np.unique(keys[d], return_index=True)
    parents = keys[d - 1][idx]
    pu, pc = np.unique(parents, return_counts=True)
    return pc[np.searchsorted(pu, keys[d - 1])]


def pd_unit_tags(keys: np.ndarray, k: int) -> np.ndarray:
    """[n_starts, T] unit tag of every event under v2's pd=k + train_crown plan."""
    T, n = keys.shape
    tags = np.empty((n, T), dtype=np.uint64)
    piece = keys[k - 1] * _C_PIECE + np.uint64(k)
    tags[:, k - 1:] = piece[:, None]
    # Depth L < k: the event belongs to the radix node whose edge holds depth L.
    # That edge ends at the first depth >= L where the context branches; if it
    # runs on to depth k, the node is an anchor and the event is in its piece.
    end_tag = piece
    for L in range(k - 1, 0, -1):
        crown = keys[L - 1] * _C_CROWN + np.uint64(L)
        end_tag = np.where(n_children(keys, L) >= 2, crown, end_tag)
        tags[:, L - 1] = end_tag
    return tags


def packed_unit_tags(keys: np.ndarray, occ: np.ndarray, piece_max: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """[n_starts, T] unit tags for the packed plan's pieces and node chunks,
    and each window's piece depth p (T+1: no piece, every event a node event)."""
    T, n = keys.shape
    p = np.full(n, T + 1, dtype=np.int32)
    for d in range(T, 0, -1):
        p = np.where(occ[d - 1].astype(np.int64) * (T - d + 1) <= piece_max, d, p)
    rows = np.arange(n)
    pd_idx = np.minimum(p, T) - 1
    piece = keys[pd_idx, rows] * _C_PIECE + p.astype(np.uint64)
    tags = np.empty((n, T), dtype=np.uint64)
    for L in range(1, T + 1):
        col = piece.copy()
        node = np.nonzero(L < p)[0]
        if node.size:
            kk = keys[L - 1][node]
            order = np.lexsort((rng.random(node.size), kk))
            ks = kk[order]
            first = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1]
            size = np.diff(np.r_[first, ks.size])
            rank = np.arange(ks.size) - np.repeat(first, size)
            nchunks = np.repeat((size + piece_max - 1) // piece_max, size)
            chunk = np.empty(node.size, dtype=np.uint64)
            chunk[order] = (rank % nchunks).astype(np.uint64)
            col[node] = kk * _C_NODE + np.uint64(L) + chunk * _C_CHUNK
        tags[:, L - 1] = col
    return tags, p


def units_from_tags(tags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """unit index of every event (flat, row-major [i, L]) and each unit's event count."""
    _, inv = np.unique(tags.ravel(), return_inverse=True)
    inv = inv.astype(np.int32)
    return inv, np.bincount(inv)


def shuffled_steps(n_units: int, rng: np.random.Generator) -> np.ndarray:
    """One unit per step, in random order: step index of each unit."""
    step = np.empty(n_units, dtype=np.int32)
    step[rng.permutation(n_units)] = np.arange(n_units, dtype=np.int32)
    return step


def packed_steps(sizes: np.ndarray, n_steps: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Units in random order, cut into n_steps runs of equal events (a unit goes
    to the step holding its midpoint). Empty steps are dropped."""
    perm = rng.permutation(sizes.size)
    s = sizes[perm].astype(np.float64)
    mid = np.cumsum(s) - s / 2
    per = s.sum() / n_steps
    raw = np.minimum((mid / per).astype(np.int64), n_steps - 1)
    used, dense = np.unique(raw, return_inverse=True)
    step = np.empty(sizes.size, dtype=np.int32)
    step[perm] = dense.astype(np.int32)
    return step, used.size


def step_windows(unit_of_event: np.ndarray, step_of_unit: np.ndarray, n: int, T: int):
    """Reduce events to (step, window) pairs sorted by step then window:
    returns step, window start, bitmask of the window's event depths (bit L-1)."""
    s = step_of_unit[unit_of_event]
    key = s.astype(np.int64) * n + (np.arange(s.size, dtype=np.int64) // T)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bits = np.left_shift(1, (order % T).astype(np.int32)).astype(np.int32)
    del key
    first = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1]
    sw_bits = np.bitwise_or.reduceat(bits, first)
    sw_key = ks[first]
    return (sw_key // n).astype(np.int32), (sw_key % n).astype(np.int64), sw_bits


def bit_length(bits: np.ndarray) -> np.ndarray:
    return np.floor(np.log2(np.maximum(bits, 1))).astype(np.int32) + 1


def plan_cost(keys: np.ndarray, sw_step: np.ndarray, sw_i: np.ndarray, sw_bits: np.ndarray) -> int:
    """Positions a trie trainer computes for this pass: distinct contexts per step."""
    lmax = bit_length(sw_bits)
    total = 0
    for L in range(1, keys.shape[0] + 1):
        m = lmax >= L
        t = keys[L - 1][sw_i[m]] * _C_COST + sw_step[m].astype(np.uint64)
        total += int(np.unique(t).size)
    return total


def context_targets(tok: np.ndarray, n_starts: int, T: int, V: int):
    """The trie's next-token counts for every event's context.

    Returns ctx [T, n_starts] int32, at ctx[L-1, i] the id of event (i, L)'s
    context if it occurs at two or more window starts, else -1; a CSR table
    over those repeated contexts' ids (indptr int64, child uint8, count int32)
    listing for each every token that follows it at some event (j, L) and how
    often, which are the trie node's child counts n(c, a); and the number of
    distinct contexts and of those seen once. A context seen once is followed
    by its own event's next token only, so it needs no entry.
    """
    if V > 256:
        raise ValueError(f"context_targets: vocab {V} does not fit uint8")
    t = tok.astype(np.int64)
    tk = tok.astype(np.uint64) + np.uint64(1)
    h = np.zeros(n_starts, dtype=np.uint64)   # prefix_keys' hash, one depth at a time
    ctx = np.empty((T, n_starts), dtype=np.int32)
    per_ctx, child, count = [], [], []
    base = n_all = n_single = 0
    for L in range(1, T + 1):
        h = h * _HB + tk[L - 1:L - 1 + n_starts]
        _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
        rep = cnt >= 2
        nrep = int(rep.sum())
        n_all += cnt.size
        n_single += cnt.size - nrep
        rid = np.full(cnt.size, -1, dtype=np.int64)
        rid[rep] = base + np.arange(nrep)
        r = rid[inv]
        ctx[L - 1] = r
        w = np.nonzero(r >= 0)[0]
        up, pc = np.unique(r[w] * V + t[w + L], return_counts=True)
        per_ctx.append(np.bincount(up // V - base, minlength=nrep))
        child.append((up % V).astype(np.uint8))
        count.append(pc.astype(np.int32))
        base += nrep
    indptr = np.r_[0, np.cumsum(np.concatenate(per_ctx))].astype(np.int64)
    return ctx, indptr, np.concatenate(child), np.concatenate(count), n_all, n_single


def context_targets_from(tok: np.ndarray, n_starts: int, T: int, V: int, ref: np.ndarray,
                         chunk: int = 1 << 23):
    """context_targets with the counts taken from another token stream.

    The events and their contexts are this corpus's (tok); the next-token counts
    of each context are counted over `ref`, normally a larger corpus that
    contains this one: n_ref(c, a) at every position of ref where c occurs and
    is followed by a token. So the trie keeps its nodes and gets the larger
    corpus's child counts. A context that never occurs in ref gets -1 and keeps
    its own event's next token. Same return layout as context_targets, with the
    CSR over the contexts found in ref; n_single counts those found exactly once.
    ref is read in chunks of `chunk` positions, one context length at a time, so
    memory stays bounded for a ref of 100M tokens.
    """
    if V > 256:
        raise ValueError(f"context_targets_from: vocab {V} does not fit uint8")
    tk = tok.astype(np.uint64) + np.uint64(1)
    h = np.zeros(n_starts, dtype=np.uint64)
    ctx = np.empty((T, n_starts), dtype=np.int32)
    uniq, base, nr = [], 0, ref.size
    for L in range(1, T + 1):
        h = h * _HB + tk[L - 1:L - 1 + n_starts]
        u, inv = np.unique(h, return_inverse=True)
        ctx[L - 1] = base + inv
        uniq.append((u, base))
        base += u.size
    n_all = base
    keys, cnts = [], []
    for L in range(1, T + 1):
        u, b = uniq[L - 1]
        kl, cl = [], []
        for s in range(0, max(0, nr - L), chunk):
            e = min(nr - L, s + chunk)          # context starts s..e-1, next token at j + L
            hc = np.zeros(e - s, dtype=np.uint64)
            for d in range(L):
                hc = hc * _HB + (ref[s + d:e + d].astype(np.uint64) + np.uint64(1))
            idx = np.minimum(np.searchsorted(u, hc), u.size - 1)
            hit = np.nonzero(u[idx] == hc)[0]
            k = (idx[hit].astype(np.int64) + b) * V + ref[s + L + hit].astype(np.int64)
            uk, uc = np.unique(k, return_counts=True)
            kl.append(uk)
            cl.append(uc)
        if kl:
            uk, inv2 = np.unique(np.concatenate(kl), return_inverse=True)
            keys.append(uk)
            cnts.append(np.bincount(inv2, weights=np.concatenate(cl)).astype(np.int64))
    key = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
    cnt = np.concatenate(cnts) if cnts else np.zeros(0, dtype=np.int64)
    if cnt.size and cnt.max() > np.iinfo(np.int32).max:
        raise ValueError("context_targets_from: a count overflows int32")
    cid = key // V
    per = np.bincount(cid, minlength=n_all)
    found = per > 0
    new_id = np.full(n_all, -1, dtype=np.int64)
    new_id[found] = np.arange(int(found.sum()))
    for L in range(T):
        ctx[L] = new_id[ctx[L]]
    tot = np.bincount(cid, weights=cnt, minlength=n_all)
    indptr = np.r_[0, np.cumsum(per[found])].astype(np.int64)
    return (ctx, indptr, (key % V).astype(np.uint8), cnt.astype(np.int32), n_all,
            int((tot[found] == 1).sum()))


def hash_path(path) -> np.uint64:
    """prefix_keys' hash of one token path."""
    h = np.uint64(0)
    with np.errstate(over="ignore"):
        for t in path:
            h = h * _HB + np.uint64(int(t) + 1)
    return h


def order_from_dump(keys: np.ndarray, unit_of_event: np.ndarray, unit_sizes: np.ndarray, rows, k: int) -> np.ndarray:
    """Step index of every pd=k unit in the order of a v2 grad dump's units.tsv rows.

    A row names its unit by the token path to its anchor, cut at depth k,
    and the anchor's edge depths. A crown unit (edge ending above depth k)
    holds depth `anchor_endpoint_depth` of the windows with that path; a
    piece holds depth k of the windows with its k-character prefix. Each
    row's unit is found through one such event, and every unit must be
    found exactly once with the row's trained_events."""
    T, n = keys.shape
    order = {}
    step = np.full(unit_sizes.size, -1, dtype=np.int32)
    for r, row in enumerate(rows):
        path = [int(x) for x in row["context_tokens"].split(",")]
        L = min(int(row["anchor_endpoint_depth"]), k)
        if len(path) != L:
            raise ValueError(f"row {r}: path length {len(path)} != probe depth {L}")
        if L not in order:
            order[L] = np.argsort(keys[L - 1], kind="stable")
        o = order[L]
        h = hash_path(path)
        j = np.searchsorted(keys[L - 1][o], h)
        if j >= n or keys[L - 1][o[j]] != h:
            raise ValueError(f"row {r}: no window has path {path}")
        u = unit_of_event[int(o[j]) * T + L - 1]
        if step[u] >= 0:
            raise ValueError(f"row {r}: unit {u} already placed at step {step[u]}")
        if int(unit_sizes[u]) != round(float(row["trained_events"])):
            raise ValueError(f"row {r}: unit {u} has {unit_sizes[u]} events, the dump {row['trained_events']}")
        step[u] = r
    if (step < 0).any():
        raise ValueError(f"{int((step < 0).sum())} units not in the dump")
    return step


def describe(sizes: np.ndarray) -> str:
    q = np.percentile(sizes, [10, 50, 90, 99])
    return (f"n={sizes.size} mean={sizes.mean():.0f} p10={q[0]:.0f} median={q[1]:.0f} "
            f"p90={q[2]:.0f} p99={q[3]:.0f} max={sizes.max()} cv={sizes.std() / sizes.mean():.3f}")
