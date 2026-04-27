"""
p2s-attention corpus-walk evaluator (PyTorch).

Loads a saved p2s model checkpoint and runs evaluation in the
*corpus-positional* setting:
  - Pick held-out corpus positions
  - For each, get the actual D-char prefix window
  - Walk the prefix radix tree from root following the corpus chars to
    find the prefix-leaf id
  - Look up that leaf's match-index entry
  - Run the model: encode prefix, encode each candidate σ, cross-attend,
    predict next-char distribution
  - Compute CE against the actual corpus[p+1]
  - Average → corpus-positional PPL

This number is directly comparable to L4 / det-AGPT / SGD numbers.
"""
import os, sys, struct, random, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure we can import P2SModel and helpers from the trainer module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2s_train import (
    P2SModel, load_radix_tree, path_tokens, iter_match_records,
    PREFIX_DIR, SUFFIX_DIR, MATCH_PATH, CORPUS_PATH, D, DEVICE,
)

TAG = os.environ.get("P2S_TAG", "default")
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"p2s_model_d{D}_{TAG}.pt")
N_EVAL    = 4096
MAX_K     = 8
SEED      = 7

def build_children_index(parent_arr, edge_arr):
    """For each parent_id, build [(first_token, child_id, edge_tokens)] list."""
    children = {}
    for cid in range(1, len(parent_arr)):
        e = edge_arr[cid]
        if e is None or len(e) == 0:
            continue
        p = parent_arr[cid]
        children.setdefault(p, []).append((e[0], cid, e))
    return children

def find_leaf(tokens, children):
    """
    Walk the prefix radix tree from root (id=0) following the chars of `tokens`.
    Returns the leaf node id reached when no more tokens match, or the deepest
    matched node when out of tokens. The "prefix leaf" for a corpus window is
    the deepest node such that the path-from-root chars are a prefix of `tokens`.
    """
    cur = 0
    pos = 0
    while pos < len(tokens):
        ch = tokens[pos]
        kids = children.get(cur)
        if not kids:
            return cur
        match = None
        for first_tok, child_id, edge in kids:
            if first_tok == ch:
                match = (child_id, edge)
                break
        if match is None:
            return cur
        child_id, edge = match
        # Verify the rest of the edge matches
        ok = True
        for i in range(1, len(edge)):
            if pos + i >= len(tokens) or tokens[pos + i] != edge[i]:
                ok = False
                break
        if not ok:
            # We can't descend the full edge, but we've matched up to the first
            # char of this edge — return the parent (cur). Strict prefix-leaf is
            # the last node we fully reached.
            return cur
        cur = child_id
        pos += len(edge)
    return cur

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("[eval] loading checkpoint:", CKPT_PATH, flush=True)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    cfg = ckpt['config']
    print(f"[eval]   config: {cfg}", flush=True)

    print("[eval] loading prefix tree...", flush=True)
    pp, pe, vocab_pre, _ = load_radix_tree(PREFIX_DIR)
    print("[eval] loading suffix tree...", flush=True)
    sp, se, vocab_suf, _ = load_radix_tree(SUFFIX_DIR)
    vocab_size = vocab_pre

    print("[eval] building children index for prefix tree...", flush=True)
    t0 = time.time()
    children = build_children_index(pp, pe)
    print(f"[eval]   children index: {len(children)} parents, {time.time() - t0:.1f}s", flush=True)

    print("[eval] indexing match records by prefix_id...", flush=True)
    match_by_pid = {}
    n_records = 0
    for pid, max_k, sigmas in iter_match_records(MATCH_PATH):
        match_by_pid[pid] = (max_k, sigmas)
        n_records += 1
    print(f"[eval]   indexed {n_records} match records", flush=True)

    print("[eval] loading corpus...", flush=True)
    text = open(CORPUS_PATH).read()
    chars_sorted = sorted(set(text))
    char_to_tok = {c: i for i, c in enumerate(chars_sorted)}
    corpus_tokens = [char_to_tok[c] for c in text]
    print(f"[eval]   corpus length: {len(corpus_tokens)}", flush=True)

    # Sample held-out corpus positions: pick from the LAST 5% of the corpus to
    # reduce overlap with training (training used the first 1M leaves, which
    # roughly corresponds to a chunk of corpus positions; using the tail
    # minimizes overlap)
    held_start = int(len(corpus_tokens) * 0.95)
    eligible_positions = list(range(held_start + D, len(corpus_tokens) - 1))
    random.shuffle(eligible_positions)
    eval_positions = eligible_positions[:N_EVAL]
    print(f"[eval]   sampled {len(eval_positions)} held-out positions from tail 5%",
          flush=True)

    # Build model and load weights
    eval_max_len = cfg.get('max_len', cfg['D'])
    model = P2SModel(
        vocab_size=cfg['vocab_size'],
        embed_size=cfg['vocab_size'] + 1,
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        max_len=eval_max_len,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[eval]   loaded model: {sum(p.numel() for p in model.parameters())} params",
          flush=True)

    # Walk each position's prefix and collect the (π_path, candidates) tuple
    print("[eval] walking prefix tree per position...", flush=True)
    PAD = vocab_size
    skipped_no_leaf = 0
    skipped_no_match = 0

    # We'll batch positions together (small batches)
    BATCH = 32
    total_loss = 0.0
    total_count = 0

    @torch.no_grad()
    def run_batch(items):
        nonlocal total_loss, total_count
        B = len(items)
        pi = torch.full((B, eval_max_len), PAD, dtype=torch.long)
        pim = torch.ones((B, eval_max_len), dtype=torch.bool)
        sg = torch.full((B, MAX_K, D), PAD, dtype=torch.long)
        sgm = torch.ones((B, MAX_K, D), dtype=torch.bool)
        sgp = torch.ones((B, MAX_K), dtype=torch.bool)
        targets = torch.zeros(B, dtype=torch.long)
        for b, (pi_path, sigmas_fwd, true_next) in enumerate(items):
            pt = pi_path[-eval_max_len:]   # take TAIL (most recent context)
            pi[b, :len(pt)] = torch.tensor(pt, dtype=torch.long)
            pim[b, :len(pt)] = False
            for ki, st in enumerate(sigmas_fwd[:MAX_K]):
                st_d = st[:D]
                sg[b, ki, :len(st_d)] = torch.tensor(st_d, dtype=torch.long)
                sgm[b, ki, :len(st_d)] = False
                sgp[b, ki] = False
            targets[b] = true_next
        pi, pim, sg, sgm, sgp, targets = (x.to(DEVICE) for x in (pi, pim, sg, sgm, sgp, targets))
        logits, alpha = model(pi, pim, sg, sgm, sgp)
        loss = F.cross_entropy(logits, targets, reduction='sum').item()
        total_loss += loss
        total_count += B

    batch = []
    for p_idx, p in enumerate(eval_positions):
        # window is corpus_tokens[p-D+1 .. p], next char is corpus_tokens[p+1]
        # We use the LEAF of the D-window walk to find the match set, but the
        # encoder input for the prefix is the longer eval_max_len tail of the
        # corpus (so a wrap-trained model gets the longer context it was
        # trained on, instead of being evaluated on short prefix only).
        win = corpus_tokens[max(0, p - D + 1) : p + 1]
        if len(win) < D:
            continue
        leaf_id = find_leaf(win, children)
        if leaf_id == 0:
            skipped_no_leaf += 1
            continue
        if leaf_id not in match_by_pid:
            skipped_no_match += 1
            continue
        max_k, sigmas = match_by_pid[leaf_id]
        if not sigmas:
            skipped_no_match += 1
            continue
        # build candidate forward paths
        sigmas_fwd = []
        for sid in sigmas[:MAX_K]:
            sp_path = path_tokens(sid, sp, se)
            if not sp_path:
                continue
            sf = list(reversed(sp_path))   # reversed-corpus → forward
            if max_k >= len(sf):
                continue
            sigmas_fwd.append(sf)
        if not sigmas_fwd:
            skipped_no_match += 1
            continue

        # Use longer-context corpus chars as the prefix encoder input
        # (matches training distribution when wrap_cycles > 1)
        long_win = corpus_tokens[max(0, p - eval_max_len + 1) : p + 1]
        pi_path = long_win

        true_next = corpus_tokens[p + 1]
        batch.append((pi_path, sigmas_fwd, true_next))

        if len(batch) >= BATCH:
            run_batch(batch)
            batch = []
        if (p_idx + 1) % 500 == 0:
            print(f"  processed {p_idx + 1}/{len(eval_positions)} positions...",
                  flush=True)
    if batch:
        run_batch(batch)

    print(f"[eval] skipped: no-leaf={skipped_no_leaf}, no-match={skipped_no_match}",
          flush=True)
    if total_count == 0:
        print("[eval] ERROR: no positions evaluated", flush=True)
        return
    mean_loss = total_loss / total_count
    print(f"[eval] corpus-positional NLL={mean_loss:.4f}, PPL={math.exp(mean_loss):.2f}, "
          f"bpc={mean_loss / math.log(2):.4f}, n={total_count}", flush=True)

if __name__ == "__main__":
    main()
