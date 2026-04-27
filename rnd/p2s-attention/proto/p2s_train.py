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
PREFIX_DIR  = "/home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus"
SUFFIX_DIR  = "/home/trans/agpt-tries/gutenberg_5m_d32_suffix_radix"
MATCH_PATH  = "/home/trans/agpt-tries/g5m_d32_p2s_match.bin"
CORPUS_PATH = "/home/trans/Projects/agpt/data/gutenberg_5m.txt"

D            = 32
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 4
BATCH_SIZE   = 32
LR           = 3e-4
N_TRAIN_STEPS = 10000   # cut from 20K — diminishing returns past ~10K
N_HELDOUT    = 4096
PRINT_EVERY  = 500
MAX_RECORDS  = 500_000  # cut from 1M for faster iteration; we have a saved 1M run already
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RNG_SEED     = 42

# ---------------- read radix tree depth files ----------------
RDXA_MAGIC = 0x52445841  # 'RDXA'

def load_radix_tree(dirpath):
    """Returns {id: parent}, {id: edge_tokens}, vocab_size."""
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
            _edge_mass, = struct.unpack_from("<i", buf, pos); pos += 4
            ec, = struct.unpack_from("<i", buf, pos); pos += 4
            pos += 8 * ec  # skip (tok, cnt) pairs
            parent_arr[rid] = parent
            edge_arr[rid] = edge
    print(f"[load]   loaded in {time.time() - t0:.1f}s", flush=True)
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
def build_examples(match_path, prefix_parent, prefix_edge,
                   suffix_parent, suffix_edge, max_records=None,
                   max_candidates=8):
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
    examples = []
    n_skipped_no_pred = 0
    for i, (pid, max_k, sigmas) in enumerate(iter_match_records(match_path)):
        if max_records and i >= max_records:
            break
        pi_path = path_tokens(pid, prefix_parent, prefix_edge)
        if not pi_path:
            continue
        sigma_specs = []
        for sid in sigmas[:max_candidates]:
            sigma_path = path_tokens(sid, suffix_parent, suffix_edge)
            sigma_fwd = list(reversed(sigma_path))  # reversed-corpus -> forward
            if max_k >= len(sigma_fwd):
                n_skipped_no_pred += 1
                continue
            pred_char = sigma_fwd[max_k]
            sigma_specs.append({
                'tokens': sigma_fwd,
                'overlap_k': max_k,
                'pred_char': pred_char,
            })
        if not sigma_specs:
            continue
        examples.append({'pi_tokens': pi_path, 'sigmas': sigma_specs})
        if len(examples) % 100000 == 0:
            print(f"[build] {len(examples)} examples...", flush=True)
    print(f"[build] done: {len(examples)} examples; "
          f"skipped {n_skipped_no_pred} candidate slots (overlap == leaf-edge length)",
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
        """
        B, K, _ = sigma_tokens.shape
        h_pi = self.encode(pi_tokens, pi_mask)                  # [B, d_model]
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
def collate(examples, vocab_size, max_k):
    """Build padded tensors. PAD token = vocab_size (extra index)."""
    PAD = vocab_size
    B = len(examples)
    pi = torch.full((B, D), PAD, dtype=torch.long)
    pi_mask = torch.ones((B, D), dtype=torch.bool)
    sigma = torch.full((B, max_k, D), PAD, dtype=torch.long)
    sigma_mask = torch.ones((B, max_k, D), dtype=torch.bool)
    sigma_pad = torch.ones((B, max_k), dtype=torch.bool)
    target = torch.zeros((B, vocab_size + 1), dtype=torch.float32)  # +1 for PAD slot

    for b, ex in enumerate(examples):
        pt = ex['pi_tokens'][:D]
        pi[b, :len(pt)] = torch.tensor(pt, dtype=torch.long)
        pi_mask[b, :len(pt)] = False
        # build target: fraction of candidates predicting each char
        for ki, s in enumerate(ex['sigmas'][:max_k]):
            st = s['tokens'][:D]
            sigma[b, ki, :len(st)] = torch.tensor(st, dtype=torch.long)
            sigma_mask[b, ki, :len(st)] = False
            sigma_pad[b, ki] = False
            target[b, s['pred_char']] += 1.0
        # normalize target to a distribution over real-vocab chars
        target[b, vocab_size] = 0.0
        s = target[b, :vocab_size].sum()
        if s > 0:
            target[b, :vocab_size] /= s
    return pi, pi_mask, sigma, sigma_mask, sigma_pad, target[:, :vocab_size]

# ---------------- main ----------------
def main():
    random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    print("[main] loading prefix tree...", flush=True)
    pp, pe, vocab_pre, _ = load_radix_tree(PREFIX_DIR)
    print("[main] loading suffix tree...", flush=True)
    sp, se, vocab_suf, _ = load_radix_tree(SUFFIX_DIR)
    assert vocab_pre == vocab_suf
    vocab_size = vocab_pre
    print(f"[main] vocab_size={vocab_size}", flush=True)

    print(f"[main] building training examples (max_records={MAX_RECORDS})...",
          flush=True)
    examples = build_examples(MATCH_PATH, pp, pe, sp, se,
                              max_records=MAX_RECORDS, max_candidates=8)
    # free some memory
    del pp, pe, sp, se

    random.shuffle(examples)
    held = examples[:N_HELDOUT]
    train = examples[N_HELDOUT:]
    print(f"[main] {len(train)} train, {len(held)} held-out", flush=True)

    model = P2SModel(vocab_size, vocab_size + 1, D_MODEL, N_HEADS, N_LAYERS, D).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] model: {n_params} params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # max candidate count we cap to (matches collate's max_k arg)
    MAX_K = 8

    def train_batch(batch):
        model.train()
        pi, pim, sg, sgm, sgp, tgt = collate(batch, vocab_size, MAX_K)
        pi, pim, sg, sgm, sgp, tgt = (x.to(DEVICE) for x in (pi, pim, sg, sgm, sgp, tgt))
        logits, _ = model(pi, pim, sg, sgm, sgp)
        logp = F.log_softmax(logits, dim=-1)
        loss = -(tgt * logp).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
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
            pi, pim, sg, sgm, sgp, tgt = collate(batch, vocab_size, MAX_K)
            pi, pim, sg, sgm, sgp, tgt = (x.to(DEVICE) for x in (pi, pim, sg, sgm, sgp, tgt))
            logits, _ = model(pi, pim, sg, sgm, sgp)
            logp = F.log_softmax(logits, dim=-1)
            loss = -(tgt * logp).sum(dim=-1).mean()
            total += loss.item() * len(batch)
            n += len(batch)
        return total / max(n, 1)

    print("[main] training...", flush=True)
    losses = []
    for step in range(N_TRAIN_STEPS):
        batch = random.sample(train, BATCH_SIZE)
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

    # Save checkpoint so corpus-walk eval can load it
    ckpt_path = os.path.join(os.path.dirname(__file__), "p2s_model.pt")
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'D': D,
            'max_k': MAX_K,
        },
    }, ckpt_path)
    print(f"[main] saved checkpoint to {ckpt_path}", flush=True)

if __name__ == "__main__":
    main()
