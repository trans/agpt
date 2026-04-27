"""
p2s-attention prototype trainer (PyTorch).

PROTOTYPE — for validating that the architecture trains. Once we see
loss decreasing on a held-out set, port to Crystal/CUDA for production.

Reads:
  - Match index: /home/trans/agpt-tries/g5m_d32_p2s_match.bin  ('P2SC' format)
  - Prefix tree: /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus/
  - Suffix tree: /home/trans/agpt-tries/gutenberg_5m_d32_suffix_radix/
  - Corpus:      /home/trans/Projects/agpt/data/gutenberg_5m.txt

Both radix trees use the format documented in
src/agpt/radix_trie_reader.cr (depth files: magic 'RDXA', version 2).

Architecture:
  - Single shared transformer encoder
  - Encodes prefix π (full path-from-root, ≤D=32 chars)
  - Encodes each candidate suffix σ (full path-from-root, ≤D=32 chars; in
    forward-corpus order, i.e. with the suffix tree's reversed-corpus path
    re-reversed)
  - Cross-attention: q = W_q · h_π[-1], for each σ k_σ = W_k · h_σ[-1],
    v_σ = W_v · h_σ[-1]; softmax(q·k/√d) over candidates
  - context = Σ α · v
  - logits = W_out · context, CE vs target
  - Target: distribution over candidates' "predicted next-char" values
    (each σ's σ.fwd_path[overlap_k] is its predicted next char)

Variables held constant for prototype:
  d_model=64, n_heads=4, n_layers=2, batch_size=16, lr=3e-4
"""

import os, sys, struct, random, math, time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- paths and config ----------------
# D can be overridden via env var P2S_D (16, 24, 32 supported); paths derived
# Used by both p2s_train.py and p2s_eval_corpus.py
D = int(os.environ.get("P2S_D", "32"))
PREFIX_DIR  = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_radix_corpus"
SUFFIX_DIR  = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_suffix_radix"
MATCH_PATH  = f"/home/trans/agpt-tries/g5m_d{D}_p2s_match.bin"
CORPUS_PATH = "/home/trans/Projects/agpt/data/gutenberg_5m.txt"

# Tag for log + checkpoint filenames; lets us run multiple recipes per D
TAG = os.environ.get("P2S_TAG", "default")

# Wrap-around: extended HISTORY context for the prefix encoder. The "2x tree"
# interpretation — instead of just D chars of history, give the model
# CONTEXT_LEN chars of history (≥D, typically 2×D or 4×D). Position-based
# training: sample corpus positions, walk the prefix tree at the position's
# D-window to find the match leaf, but encode corpus[p-CONTEXT_LEN+1..p] as
# the prefix input. Target = corpus[p+1] (true next char).
CONTEXT_LEN = int(os.environ.get("P2S_CONTEXT_LEN", str(D)))
MAX_LEN     = CONTEXT_LEN
# Legacy chained variant (now deprecated; left for one-off comparison)
WRAP_CYCLES = int(os.environ.get("P2S_WRAP", "1"))
D_MODEL      = int(os.environ.get("P2S_D_MODEL", "256"))
N_HEADS      = int(os.environ.get("P2S_N_HEADS", "8"))
N_LAYERS     = int(os.environ.get("P2S_N_LAYERS", "6"))
BATCH_SIZE   = int(os.environ.get("P2S_BATCH",   "32"))
LR           = float(os.environ.get("P2S_LR",    "5e-4"))
N_TRAIN_STEPS = int(os.environ.get("P2S_STEPS",  "30000"))
WARMUP_STEPS  = int(os.environ.get("P2S_WARMUP", "1000"))
WEIGHT_DECAY  = float(os.environ.get("P2S_WD",   "0.01"))
N_HELDOUT     = 4096
PRINT_EVERY   = 500
MAX_RECORDS   = int(os.environ.get("P2S_RECORDS", "2000000"))  # 2M of 5M
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RNG_SEED     = 42

# ---------------- read radix tree depth files ----------------
RDXA_MAGIC = 0x52445841  # 'RDXA'

def load_radix_tree(dirpath, with_mass=False):
    """Returns parent_arr, edge_arr, vocab_size, radix_count, [mass_arr if with_mass]."""
    meta_path = os.path.join(dirpath, "meta.bin")
    with open(meta_path, "rb") as f:
        magic, = struct.unpack("<I", f.read(4))
        assert magic == RDXA_MAGIC, f"bad magic {magic:x}"
        version, = struct.unpack("<i", f.read(4))
        assert version == 2
        radix_count, = struct.unpack("<i", f.read(4))
        depth_file_count, = struct.unpack("<i", f.read(4))
        total_edge_chars, = struct.unpack("<q", f.read(8))
        corpus_token_count, = struct.unpack("<i", f.read(4))
        vocab_size, = struct.unpack("<i", f.read(4))
        corpus_hash, = struct.unpack("<Q", f.read(8))
        tlen, = struct.unpack("<i", f.read(4))
        f.read(tlen)  # tokenizer tag

    parent_arr = [0] * (radix_count + 1)
    edge_arr = [None] * (radix_count + 1)
    mass_arr = [0] * (radix_count + 1) if with_mass else None

    print(f"[load] {dirpath}: {radix_count} nodes, {depth_file_count} depth files",
          flush=True)
    t0 = time.time()
    for d in range(depth_file_count):
        depth_path = os.path.join(dirpath, f"radix_depth_{d:03d}.bin")
        if not os.path.exists(depth_path):
            continue
        with open(depth_path, "rb") as f:
            buf = f.read()
        pos = 0
        magic, = struct.unpack_from("<I", buf, pos); pos += 4
        assert magic == RDXA_MAGIC
        stored_d, = struct.unpack_from("<i", buf, pos); pos += 4
        assert stored_d == d
        n, = struct.unpack_from("<i", buf, pos); pos += 4
        for _ in range(n):
            rid, parent, fcd, edge_len = struct.unpack_from("<iiii", buf, pos); pos += 16
            edge = list(struct.unpack_from(f"<{edge_len}i", buf, pos)); pos += 4 * edge_len
            edge_mass, = struct.unpack_from("<i", buf, pos); pos += 4
            ec, = struct.unpack_from("<i", buf, pos); pos += 4
            pos += 8 * ec  # skip (tok, cnt) pairs
            parent_arr[rid] = parent
            edge_arr[rid] = edge
            if mass_arr is not None:
                mass_arr[rid] = edge_mass
    print(f"[load]   loaded in {time.time() - t0:.1f}s", flush=True)
    if with_mass:
        return parent_arr, edge_arr, vocab_size, radix_count, mass_arr
    return parent_arr, edge_arr, vocab_size, radix_count

def path_tokens(node_id, parent, edge):
    """Walk parent chain back to root collecting edge tokens (root->leaf order)."""
    segments = []
    cur = node_id
    while cur > 0:
        e = edge[cur]
        if e is not None:
            segments.append(e)
        p = parent[cur]
        if p is None:
            break
        cur = p
    out = []
    for seg in reversed(segments):
        out.extend(seg)
    return out

# ---------------- read match index ----------------
P2SC_MAGIC = 0x50325343  # 'P2SC'

def iter_match_records(path):
    """Yields (prefix_id, max_k, [sigma_ids]) for each record."""
    with open(path, "rb") as f:
        buf = f.read()
    pos = 0
    magic, = struct.unpack_from("<I", buf, pos); pos += 4
    assert magic == P2SC_MAGIC
    pos += 12  # prefix_count, suffix_count, suffix_leaf_count
    while pos < len(buf):
        pid, max_k, n_sigmas = struct.unpack_from("<iii", buf, pos); pos += 12
        sigmas = list(struct.unpack_from(f"<{n_sigmas}i", buf, pos)); pos += 4 * n_sigmas
        yield pid, max_k, sigmas

# ---------------- build training examples ----------------
def build_children_index(parent_arr, edge_arr):
    """For walking prefix tree from root: parent → [(first_token, child_id, edge_tokens)]."""
    children = {}
    for cid in range(1, len(parent_arr)):
        e = edge_arr[cid]
        if e is None or len(e) == 0:
            continue
        p = parent_arr[cid]
        children.setdefault(p, []).append((e[0], cid, e))
    return children

def find_leaf(tokens, children_index):
    """Walk prefix tree from root following tokens, return deepest matched node id."""
    cur = 0
    pos = 0
    while pos < len(tokens):
        ch = tokens[pos]
        kids = children_index.get(cur)
        if not kids:
            return cur
        match = None
        for first_tok, child_id, edge in kids:
            if first_tok == ch:
                match = (child_id, edge); break
        if match is None:
            return cur
        child_id, edge = match
        ok = True
        for i in range(1, len(edge)):
            if pos + i >= len(tokens) or tokens[pos + i] != edge[i]:
                ok = False; break
        if not ok:
            return cur
        cur = child_id
        pos += len(edge)
    return cur

def build_examples_position_based(corpus_tokens, match_lookup,
                                   suffix_parent, suffix_edge,
                                   children_index, n_examples,
                                   max_candidates=8, context_len=32, D=32, seed=42):
    """
    Position-based training-example builder. Each example:
      - Sample corpus position p (uniformly)
      - Find prefix-tree leaf for the D-window ending at p
      - Look up the leaf's match record
      - Encode corpus[p-context_len+1..p] as prefix input (extended history)
      - Target = corpus[p+1] (true next char)
    """
    rng = random.Random(seed)
    examples = []
    n_skipped_no_leaf = 0
    n_skipped_no_match = 0
    L = len(corpus_tokens)
    eligible = list(range(D, L - 1))   # need D chars before, 1 after
    rng.shuffle(eligible)

    for p in eligible:
        if len(examples) >= n_examples:
            break
        win = corpus_tokens[p - D + 1: p + 1]
        leaf_id = find_leaf(win, children_index)
        if leaf_id <= 0:
            n_skipped_no_leaf += 1
            continue
        if leaf_id not in match_lookup:
            n_skipped_no_match += 1
            continue
        max_k, sigmas = match_lookup[leaf_id]
        if not sigmas:
            n_skipped_no_match += 1
            continue
        sigma_specs = []
        for sid in sigmas[:max_candidates]:
            sp_path = path_tokens(sid, suffix_parent, suffix_edge)
            sf = list(reversed(sp_path))
            if max_k >= len(sf):
                continue
            sigma_specs.append({
                'tokens': sf,
                'overlap_k': max_k,
                'pred_char': sf[max_k],
            })
        if not sigma_specs:
            n_skipped_no_match += 1
            continue

        history = corpus_tokens[max(0, p - context_len + 1): p + 1]
        true_next = corpus_tokens[p + 1]
        examples.append({
            'pi_tokens': history,
            'sigmas': sigma_specs,
            'target_char': true_next,
            'corpus_pos': p,
        })
        if len(examples) % 100000 == 0:
            print(f"[build-pos] {len(examples)} examples...", flush=True)

    print(f"[build-pos] done: {len(examples)} examples; "
          f"skipped {n_skipped_no_leaf} no-leaf, {n_skipped_no_match} no-match",
          flush=True)
    return examples

def build_examples(match_path, prefix_parent, prefix_edge,
                   suffix_parent, suffix_edge, max_records=None,
                   max_candidates=8, prefix_mass=None, skip_k1=False,
                   wrap_cycles=1, children_index=None):
    """
    Returns list of dicts, each:
      {
        'pi_tokens': [int]   # prefix full path, length ≤ D
        'sigmas':    [{'tokens': [int], 'overlap_k': int, 'pred_char': int}, ...]
      }

    Suffix tokens are in forward-corpus order (reverse the suffix-tree path,
    which is in reversed-corpus order). pred_char = sigma_tokens_fwd[overlap_k]
    when that index is in range; otherwise this candidate is skipped.

    Limits:
      - max_records: cap on examples (None = all)
      - max_candidates: cap on candidates per example (large match sets are
        truncated; fine for prototyping)
    """
    # If wrap_cycles > 1, we need a fast lookup of all matches by pid for chaining.
    match_lookup = None
    if wrap_cycles > 1:
        if children_index is None:
            raise ValueError("wrap_cycles>1 requires children_index for prefix-tree walks")
        print(f"[build] (wrap={wrap_cycles}) preloading match_lookup...", flush=True)
        match_lookup = {}
        for p, k, ss in iter_match_records(match_path):
            match_lookup[p] = (k, ss)
        print(f"[build]   match_lookup: {len(match_lookup)} entries", flush=True)

    def get_sigma_specs(pid_, max_k_, sigmas_):
        """Build sigma_specs list for one cycle's match record."""
        out = []
        for sid in sigmas_[:max_candidates]:
            sp_path = path_tokens(sid, suffix_parent, suffix_edge)
            sf = list(reversed(sp_path))
            if max_k_ >= len(sf):
                continue
            out.append({
                'tokens': sf,
                'overlap_k': max_k_,
                'pred_char': sf[max_k_],
            })
        return out

    examples = []
    n_skipped_no_pred = 0
    n_skipped_k1 = 0
    n_seen = 0
    n_chains_short = 0  # chains that broke before wrap_cycles completed
    chain_lens = [0] * (wrap_cycles + 1)

    iter_records = (match_lookup.items() if match_lookup is not None
                    else ((p, (k, s)) for p, k, s in iter_match_records(match_path)))

    for pid, kv in iter_records:
        n_seen += 1
        max_k, sigmas = kv
        if skip_k1 and len(sigmas) <= 1:
            n_skipped_k1 += 1
            continue
        if max_records and len(examples) >= max_records:
            break

        # Build the chain
        chain_specs = []   # list of [sigma_specs] per cycle
        chain_pi_tokens = []  # π_tokens of cycle 0
        curr_pid = pid
        curr_max_k = max_k
        curr_sigmas = sigmas
        for cycle in range(wrap_cycles):
            specs = get_sigma_specs(curr_pid, curr_max_k, curr_sigmas)
            if not specs:
                break
            if cycle == 0:
                chain_pi_tokens = path_tokens(curr_pid, prefix_parent, prefix_edge)
                if not chain_pi_tokens:
                    break
            chain_specs.append({'specs': specs, 'overlap_k': curr_max_k})
            # Advance: top-1 σ's content -> walk prefix tree -> next pid
            if cycle == wrap_cycles - 1 or match_lookup is None:
                break
            top_sigma_fwd = specs[0]['tokens']
            next_pid = find_leaf(top_sigma_fwd, children_index)
            if next_pid <= 0 or next_pid == curr_pid:
                break
            if next_pid not in match_lookup:
                break
            curr_pid = next_pid
            curr_max_k, curr_sigmas = match_lookup[curr_pid]

        if not chain_specs:
            continue
        if len(chain_specs) < wrap_cycles:
            n_chains_short += 1
        chain_lens[len(chain_specs)] += 1

        # Combined prefix tokens: π_0's path + each subsequent σ's "new chars past overlap"
        combined = list(chain_pi_tokens)
        for cycle_idx in range(1, len(chain_specs)):
            prev = chain_specs[cycle_idx - 1]
            top_sigma_fwd = prev['specs'][0]['tokens']
            new_chars = top_sigma_fwd[prev['overlap_k']:]
            combined.extend(new_chars)

        # Final cycle's σ specs are the targets for cross-attention loss
        final_specs = chain_specs[-1]['specs']

        ex = {
            'pi_tokens': combined,
            'sigmas': final_specs,
        }
        if prefix_mass is not None:
            ex['mass'] = prefix_mass[pid] if 0 <= pid < len(prefix_mass) else 1
        examples.append(ex)

        if len(examples) % 100000 == 0:
            print(f"[build] {len(examples)} examples (scanned {n_seen})...", flush=True)

    print(f"[build] done: {len(examples)} examples (scanned {n_seen}); "
          f"skipped {n_skipped_no_pred} candidate slots, {n_skipped_k1} K=1 records",
          flush=True)
    if wrap_cycles > 1:
        hist = ", ".join(f"len={i}:{c}" for i, c in enumerate(chain_lens) if c > 0)
        print(f"[build] chain length histogram: {hist}, short_chains={n_chains_short}",
              flush=True)
    return examples

# ---------------- model ----------------
class P2SModel(nn.Module):
    def __init__(self, vocab_size, embed_size, d_model=64, n_heads=4, n_layers=2, max_len=32):
        """vocab_size = real-vocab output size; embed_size = vocab + 1 PAD slot"""
        super().__init__()
        self.embed = nn.Embedding(embed_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def encode(self, tokens, attn_mask):
        """tokens: [B, L]  attn_mask: [B, L] (True = pad)
           Returns last non-pad position's hidden state, shape [B, d_model]."""
        B, L = tokens.shape
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.embed(tokens) + self.pos(pos_ids)
        # nn.TransformerEncoderLayer expects src_key_padding_mask: True at PAD
        h = self.encoder(x, src_key_padding_mask=attn_mask)
        # Take hidden state at last non-pad position
        last_idx = (~attn_mask).long().sum(dim=1) - 1   # [B]
        last_idx = last_idx.clamp(min=0)
        h_last = h[torch.arange(B, device=tokens.device), last_idx]   # [B, d_model]
        return h_last

    def forward(self, pi_tokens, pi_mask, sigma_tokens, sigma_mask, sigma_pad):
        """
        pi_tokens: [B, D]      pi_mask: [B, D] (True at pad)
        sigma_tokens: [B, K, D]   sigma_mask: [B, K, D] (True at pad)
        sigma_pad: [B, K] (True for slots that are not real candidates)
        Returns logits [B, vocab].

        Two modes:
          - direct (P2S_DIRECT=1): bypass cross-attn, predict logits = W_out · h_π
          - default: cross-attention path through σ candidates (original p2s)
        """
        B, K, _ = sigma_tokens.shape
        h_pi = self.encode(pi_tokens, pi_mask)                  # [B, d_model]

        if os.environ.get("P2S_DIRECT", "0") == "1":
            # Skip σ encoding and cross-attention; predict directly from h_π
            logits = self.W_out(h_pi)
            return logits, None

        # encode each candidate
        sigma_flat = sigma_tokens.reshape(B * K, -1)
        sigma_mask_flat = sigma_mask.reshape(B * K, -1)
        # any all-pad rows would NaN; replace with single non-pad to avoid crash
        all_pad = sigma_mask_flat.all(dim=1)
        sigma_mask_flat = sigma_mask_flat.clone()
        sigma_mask_flat[all_pad, 0] = False  # un-mask first position for stability
        h_sigma = self.encode(sigma_flat, sigma_mask_flat)       # [B*K, d_model]
        h_sigma = h_sigma.view(B, K, -1)
        # cross-attention scoring
        q = self.W_q(h_pi)                                       # [B, d_model]
        k = self.W_k(h_sigma)                                    # [B, K, d_model]
        v = self.W_v(h_sigma)                                    # [B, K, d_model]
        scores = torch.einsum('bd,bkd->bk', q, k) / math.sqrt(self.d_model)
        scores = scores.masked_fill(sigma_pad, float('-inf'))
        alpha = F.softmax(scores, dim=-1)                        # [B, K]
        # If a row has all candidates padded, alpha will be NaN — guard:
        alpha = torch.nan_to_num(alpha, nan=0.0)
        context = torch.einsum('bk,bkd->bd', alpha, v)           # [B, d_model]
        logits = self.W_out(context)                             # [B, vocab]
        return logits, alpha

# ---------------- batching ----------------
def collate(examples, vocab_size, max_k, pi_max_len=None):
    """Build padded tensors. PAD token = vocab_size (extra index)."""
    PAD = vocab_size
    B = len(examples)
    pi_len = pi_max_len if pi_max_len is not None else D
    pi = torch.full((B, pi_len), PAD, dtype=torch.long)
    pi_mask = torch.ones((B, pi_len), dtype=torch.bool)
    sigma = torch.full((B, max_k, D), PAD, dtype=torch.long)
    sigma_mask = torch.ones((B, max_k, D), dtype=torch.bool)
    sigma_pad = torch.ones((B, max_k), dtype=torch.bool)
    target = torch.zeros((B, vocab_size + 1), dtype=torch.float32)  # +1 for PAD slot

    for b, ex in enumerate(examples):
        pt = ex['pi_tokens'][-pi_len:]   # take TAIL (most recent context)
        pi[b, :len(pt)] = torch.tensor(pt, dtype=torch.long)
        pi_mask[b, :len(pt)] = False
        # σ candidates (still used for cross-attention K/V)
        for ki, s in enumerate(ex['sigmas'][:max_k]):
            st = s['tokens'][:D]
            sigma[b, ki, :len(st)] = torch.tensor(st, dtype=torch.long)
            sigma_mask[b, ki, :len(st)] = False
            sigma_pad[b, ki] = False
        # target: prefer ex['target_char'] (corpus-aligned, position-based) when
        # present; otherwise use the σ candidates' pred_char distribution.
        if 'target_char' in ex:
            target[b, ex['target_char']] = 1.0
        else:
            for s in ex['sigmas'][:max_k]:
                target[b, s['pred_char']] += 1.0
            target[b, vocab_size] = 0.0
            tot = target[b, :vocab_size].sum()
            if tot > 0:
                target[b, :vocab_size] /= tot
    return pi, pi_mask, sigma, sigma_mask, sigma_pad, target[:, :vocab_size]

# ---------------- main ----------------
def main():
    random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    SKIP_K1 = os.environ.get("P2S_SKIP_K1", "0") == "1"
    MASS_WEIGHT = os.environ.get("P2S_MASS_WEIGHT", "off")
    POS_BASED = os.environ.get("P2S_POS_BASED", "0") == "1" or CONTEXT_LEN > D
    print(f"[main] config: D={D} TAG={TAG} CONTEXT_LEN={CONTEXT_LEN} "
          f"POS_BASED={POS_BASED} SKIP_K1={SKIP_K1} MASS_WEIGHT={MASS_WEIGHT}",
          flush=True)

    print("[main] loading prefix tree...", flush=True)
    if MASS_WEIGHT != "off":
        pp, pe, vocab_pre, _, pmass = load_radix_tree(PREFIX_DIR, with_mass=True)
    else:
        pp, pe, vocab_pre, _ = load_radix_tree(PREFIX_DIR)
        pmass = None
    print("[main] loading suffix tree...", flush=True)
    sp, se, vocab_suf, _ = load_radix_tree(SUFFIX_DIR)
    assert vocab_pre == vocab_suf
    vocab_size = vocab_pre
    print(f"[main] vocab_size={vocab_size}", flush=True)

    children_index = None
    if WRAP_CYCLES > 1 or POS_BASED:
        print("[main] building children index for prefix tree...",
              flush=True)
        t0 = time.time()
        children_index = build_children_index(pp, pe)
        print(f"[main]   children_index: {len(children_index)} parents, {time.time()-t0:.1f}s",
              flush=True)

    if POS_BASED:
        print(f"[main] loading corpus for position-based training...", flush=True)
        text = open(CORPUS_PATH).read()
        chars_sorted = sorted(set(text))
        c2t = {c: i for i, c in enumerate(chars_sorted)}
        corpus_tokens = [c2t[c] for c in text]
        print(f"[main]   corpus length: {len(corpus_tokens)}", flush=True)

        print(f"[main] preloading match index by pid...", flush=True)
        t0 = time.time()
        match_lookup = {}
        for p_, k_, ss_ in iter_match_records(MATCH_PATH):
            match_lookup[p_] = (k_, ss_)
        print(f"[main]   match_lookup: {len(match_lookup)} entries, {time.time()-t0:.1f}s",
              flush=True)

        print(f"[main] building position-based training examples (n={MAX_RECORDS})...",
              flush=True)
        examples = build_examples_position_based(
            corpus_tokens, match_lookup, sp, se, children_index,
            n_examples=MAX_RECORDS, max_candidates=8, context_len=CONTEXT_LEN, D=D,
            seed=42,
        )
    else:
        print(f"[main] building training examples (max_records={MAX_RECORDS})...",
              flush=True)
        examples = build_examples(MATCH_PATH, pp, pe, sp, se,
                                  max_records=MAX_RECORDS, max_candidates=8,
                                  prefix_mass=pmass, skip_k1=SKIP_K1,
                                  wrap_cycles=WRAP_CYCLES, children_index=children_index)
    # free some memory
    del pp, pe, sp, se

    random.shuffle(examples)
    held = examples[:N_HELDOUT]
    train = examples[N_HELDOUT:]
    print(f"[main] {len(train)} train, {len(held)} held-out", flush=True)

    model = P2SModel(vocab_size, vocab_size + 1, D_MODEL, N_HEADS, N_LAYERS, MAX_LEN).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] model: {n_params} params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # warmup-cosine LR schedule
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, N_TRAIN_STEPS - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # max candidate count we cap to (matches collate's max_k arg)
    MAX_K = 8

    def train_batch(batch):
        model.train()
        pi, pim, sg, sgm, sgp, tgt = collate(batch, vocab_size, MAX_K, pi_max_len=MAX_LEN)
        pi, pim, sg, sgm, sgp, tgt = (x.to(DEVICE) for x in (pi, pim, sg, sgm, sgp, tgt))
        logits, _ = model(pi, pim, sg, sgm, sgp)
        logp = F.log_softmax(logits, dim=-1)
        loss = -(tgt * logp).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        return loss.item()

    @torch.no_grad()
    def eval_loss(examples):
        model.eval()
        total = 0.0
        n = 0
        i = 0
        while i < len(examples):
            batch = examples[i:i + BATCH_SIZE]
            i += BATCH_SIZE
            pi, pim, sg, sgm, sgp, tgt = collate(batch, vocab_size, MAX_K, pi_max_len=MAX_LEN)
            pi, pim, sg, sgm, sgp, tgt = (x.to(DEVICE) for x in (pi, pim, sg, sgm, sgp, tgt))
            logits, _ = model(pi, pim, sg, sgm, sgp)
            logp = F.log_softmax(logits, dim=-1)
            loss = -(tgt * logp).sum(dim=-1).mean()
            total += loss.item() * len(batch)
            n += len(batch)
        return total / max(n, 1)

    # Build mass-weighted sampling weights (if enabled) — uses log/sqrt/linear
    # to compress the heavy-tail mass distribution.
    sampling_weights = None
    if MASS_WEIGHT != "off" and 'mass' in train[0]:
        raw = [ex['mass'] for ex in train]
        if MASS_WEIGHT == "log":
            sampling_weights = [math.log(1 + m) for m in raw]
        elif MASS_WEIGHT == "sqrt":
            sampling_weights = [math.sqrt(max(1, m)) for m in raw]
        elif MASS_WEIGHT == "linear":
            sampling_weights = [max(1, m) for m in raw]
        wmin, wmax = min(sampling_weights), max(sampling_weights)
        wmean = sum(sampling_weights) / len(sampling_weights)
        print(f"[main] sampling weights ({MASS_WEIGHT}): "
              f"min={wmin:.2f} mean={wmean:.2f} max={wmax:.2f}", flush=True)

    print("[main] training...", flush=True)
    losses = []
    for step in range(N_TRAIN_STEPS):
        if sampling_weights is None:
            batch = random.sample(train, BATCH_SIZE)
        else:
            batch = random.choices(train, weights=sampling_weights, k=BATCH_SIZE)
        loss = train_batch(batch)
        losses.append(loss)
        if (step + 1) % PRINT_EVERY == 0:
            mean_recent = sum(losses[-PRINT_EVERY:]) / PRINT_EVERY
            held_loss = eval_loss(held[:512])
            print(f"  step {step+1:5d}/{N_TRAIN_STEPS}  "
                  f"train_loss={mean_recent:.4f}  held_loss={held_loss:.4f}  "
                  f"held_ppl={math.exp(held_loss):.2f}", flush=True)

    print("[main] final eval (full held-out)...", flush=True)
    final = eval_loss(held)
    print(f"[main] final held-out loss={final:.4f}, ppl={math.exp(final):.2f}", flush=True)

    # Save checkpoint so corpus-walk eval can load it (per-D + tag filename)
    ckpt_path = os.path.join(os.path.dirname(__file__), f"p2s_model_d{D}_{TAG}.pt")
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'D': D,
            'max_k': MAX_K,
            'max_len': MAX_LEN,
            'wrap_cycles': WRAP_CYCLES,
        },
    }, ckpt_path)
    print(f"[main] saved checkpoint to {ckpt_path}", flush=True)

if __name__ == "__main__":
    main()
