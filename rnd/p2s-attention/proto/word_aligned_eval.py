"""Inference-time mask using WORD-ALIGNED matching.

Same evaluation as before: for each held-out corpus position, get the
candidate set, apply mask to model logits if multi-distinct, compute CE.

Key change: candidates derived from word-aligned matching (head trie of
truncated suffix forwards) instead of char-aligned matching.
"""
import sys, os, math, random, time, array
sys.path.insert(0, "/home/trans/Projects/agpt/rnd/p2s-attention/proto")
import torch
import torch.nn.functional as F
from p2s_train import P2SModel, load_radix_tree, path_tokens

D = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX_DIR = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_radix_corpus"
SUFFIX_DIR = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_suffix_radix"
CORPUS_PATH = "/home/trans/Projects/agpt/data/gutenberg_5m.txt"
CKPT_PATH = "/home/trans/Projects/agpt/rnd/p2s-attention/proto/p2s_model_d32_direct_ctx128.pt"
N_EVAL = 4096
SEED = 7

text = open(CORPUS_PATH).read()
chars_sorted = sorted(set(text))
char_to_tok = {c: i for i, c in enumerate(chars_sorted)}
SPACE_TOK = char_to_tok[' ']
corpus_tokens = [char_to_tok[c] for c in text]

def truncate_at_last_space(seq):
    last_sp = -1
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == SPACE_TOK:
            last_sp = i
            break
    if last_sp < 0:
        return list(seq)
    return list(seq[:last_sp + 1])

print("Loading checkpoint and trees...", flush=True)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
cfg = ckpt['config']
pp, pe, vocab_size, _ = load_radix_tree(PREFIX_DIR)
sp, se, _, _ = load_radix_tree(SUFFIX_DIR)

# Identify suffix leaves
print("Identifying suffix leaves...", flush=True)
is_p_s = [False] * len(sp)
for cid in range(1, len(sp)):
    p = sp[cid]
    if 0 < p < len(sp): is_p_s[p] = True
s_leaves = [cid for cid in range(1, len(sp)) if not is_p_s[cid]]
print(f"  {len(s_leaves)} suffix leaves")

# Build word-aligned head trie + map sigma_id -> truncated_length
# (so we can compute σ.fwd[k] later)
print("Building word-aligned head trie...", flush=True)
t0 = time.time()
node_char = array.array('i', [-1])
node_first_child = array.array('i', [-1])
node_next_sib = array.array('i', [-1])
node_first_term = array.array('i', [-1])
term_sigma = array.array('i')
term_next = array.array('i')
sigma_truncated_len = {}   # sigma_id -> truncated path length
n_inserted = 0
for sleaf in s_leaves:
    sigma_path = path_tokens(sleaf, sp, se)
    sigma_fwd = sigma_path[::-1]
    sigma_truncated = truncate_at_last_space(sigma_fwd)
    sigma_truncated_len[sleaf] = len(sigma_truncated)
    cur = 0
    for ch in sigma_truncated:
        child = node_first_child[cur]
        while child != -1:
            if node_char[child] == ch: break
            child = node_next_sib[child]
        if child == -1:
            ni = len(node_char)
            node_char.append(ch)
            node_first_child.append(-1)
            node_next_sib.append(node_first_child[cur])
            node_first_term.append(-1)
            node_first_child[cur] = ni
            cur = ni
        else:
            cur = child
    ti = len(term_sigma)
    term_sigma.append(sleaf)
    term_next.append(node_first_term[cur])
    node_first_term[cur] = ti
    n_inserted += 1
    if n_inserted % 1000000 == 0:
        print(f"  inserted {n_inserted}/{len(s_leaves)}, head trie {len(node_char)} nodes",
              flush=True)
print(f"  head trie: {len(node_char)} nodes, built in {time.time()-t0:.1f}s")

# Cache full sigma forward paths for the candidates we'll look up
# (lazy: compute on demand)
sigma_fwd_cache = {}
def get_sigma_fwd(sleaf):
    if sleaf not in sigma_fwd_cache:
        sigma_fwd_cache[sleaf] = path_tokens(sleaf, sp, se)[::-1]
    return sigma_fwd_cache[sleaf]

# Build prefix-tree children index
print("Building prefix-tree children index...", flush=True)
children = {}
for cid in range(1, len(pp)):
    e = pe[cid]
    if e is None or len(e) == 0: continue
    par = pp[cid]
    children.setdefault(par, []).append((e[0], cid, e))

def find_leaf(tokens):
    cur, pos = 0, 0
    while pos < len(tokens):
        ch = tokens[pos]
        kids = children.get(cur)
        if not kids: return cur
        match = None
        for first, cid, edge in kids:
            if first == ch: match = (cid, edge); break
        if match is None: return cur
        cid, edge = match
        ok = True
        for i in range(1, len(edge)):
            if pos+i >= len(tokens) or tokens[pos+i] != edge[i]: ok = False; break
        if not ok: return cur
        cur = cid; pos += len(edge)
    return cur

eval_max_len = cfg.get('max_len', cfg['D'])

# Build model
model = P2SModel(
    vocab_size=cfg['vocab_size'], embed_size=cfg['vocab_size']+1,
    d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
    max_len=eval_max_len,
).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded ({sum(p.numel() for p in model.parameters())} params)")

PAD = vocab_size

def get_word_aligned_candidates(p):
    """For corpus position p, return list of distinct candidate next-char predictions."""
    win = corpus_tokens[max(0, p - D + 1):p + 1]
    if len(win) < D: return None, 0
    leaf_id = find_leaf(win)
    if leaf_id == 0: return None, 0
    # Compute truncated prefix path
    pi_path = path_tokens(leaf_id, pp, pe)
    truncated = truncate_at_last_space(pi_path)
    if len(truncated) == 0: return None, 0
    # Walk head trie with truncated path's chars (longest-first to find max k)
    L = len(truncated)
    max_k = 0; matched_node = -1
    k = L
    while k >= 1:
        cur = 0; ok = True
        for i in range(k):
            ch = truncated[L - k + i]
            child = node_first_child[cur]
            while child != -1:
                if node_char[child] == ch: break
                child = node_next_sib[child]
            if child == -1: ok = False; break
            cur = child
        if ok:
            max_k = k; matched_node = cur; break
        k -= 1
    if matched_node == -1: return None, 0
    # Collect candidates' σ.fwd[overlap_k] (the char right after the truncated portion)
    # = first char after where the matched portion ends in σ's full forward path
    cand_chars = []
    stack = [matched_node]
    seen = 0
    while stack and seen < 64:
        n = stack.pop()
        t = node_first_term[n]
        while t != -1 and seen < 64:
            sleaf = term_sigma[t]
            t_len = sigma_truncated_len[sleaf]
            sigma_fwd_full = get_sigma_fwd(sleaf)
            if t_len < len(sigma_fwd_full):
                cand_chars.append(sigma_fwd_full[t_len])
            seen += 1
            t = term_next[t]
        c = node_first_child[n]
        while c != -1:
            stack.append(c); c = node_next_sib[c]
    return cand_chars, max_k

# Sample held-out positions: arbitrary AND space-ending
random.seed(SEED)
held_start = int(len(corpus_tokens) * 0.95)
all_positions = list(range(held_start + D, len(corpus_tokens) - 1))
space_positions = [p for p in all_positions if corpus_tokens[p] == SPACE_TOK]
random.shuffle(space_positions)
random.shuffle(all_positions)
eval_pos_space = space_positions[:N_EVAL]
eval_pos_all = all_positions[:N_EVAL]

@torch.no_grad()
def eval_with_mask(positions, label):
    total_loss = 0.0; total_n = 0
    applied = 0; skipped_oos = 0; no_match = 0
    BATCH = 32

    items = []  # (long_win, target, cand_chars)
    for p in positions:
        long_win = corpus_tokens[max(0, p - eval_max_len + 1):p + 1]
        target = corpus_tokens[p + 1]
        cand_chars, _ = get_word_aligned_candidates(p)
        if cand_chars is None or len(cand_chars) == 0:
            no_match += 1
            cand_chars = []
        items.append((long_win, target, cand_chars))

    for i in range(0, len(items), BATCH):
        chunk = items[i:i+BATCH]
        B = len(chunk)
        pi = torch.full((B, eval_max_len), PAD, dtype=torch.long)
        pim = torch.ones((B, eval_max_len), dtype=torch.bool)
        sg = torch.full((B, 8, D), PAD, dtype=torch.long)
        sgm = torch.ones((B, 8, D), dtype=torch.bool)
        sgp = torch.ones((B, 8), dtype=torch.bool)
        targets = torch.zeros(B, dtype=torch.long)
        cand_lists = []
        for b, (lw, tgt, cc) in enumerate(chunk):
            pt = lw[-eval_max_len:]
            pi[b,:len(pt)] = torch.tensor(pt, dtype=torch.long)
            pim[b,:len(pt)] = False
            targets[b] = tgt
            cand_lists.append(cc)
        pi,pim,sg,sgm,sgp,targets = (x.to(DEVICE) for x in (pi,pim,sg,sgm,sgp,targets))
        os.environ["P2S_DIRECT"] = "1"
        logits, _ = model(pi, pim, sg, sgm, sgp)
        # Apply word-aligned mask
        vs = logits.shape[1]
        mask = torch.ones((B, vs), dtype=torch.bool, device=DEVICE)
        tgt_cpu = targets.cpu().tolist()
        for b, cc in enumerate(cand_lists):
            ds = set(cc)
            if len(ds) > 1 and tgt_cpu[b] in ds:
                in_set = torch.zeros(vs, dtype=torch.bool, device=DEVICE)
                for c in ds:
                    if 0 <= c < vs: in_set[c] = True
                mask[b] = in_set
                applied += 1
            elif len(ds) > 1:
                skipped_oos += 1
        masked_logits = logits.masked_fill(~mask, -1e9)
        loss = F.cross_entropy(masked_logits, targets, reduction='sum').item()
        total_loss += loss; total_n += B
    nll = total_loss / total_n
    print(f"[{label}] n={total_n}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}  "
          f"applied={applied} skipped_oos={skipped_oos} no_match={no_match}", flush=True)
    return nll, math.exp(nll)

@torch.no_grad()
def eval_no_mask(positions, label):
    total_loss = 0.0; total_n = 0
    BATCH = 32
    for i in range(0, len(positions), BATCH):
        chunk = positions[i:i+BATCH]
        B = len(chunk)
        pi = torch.full((B, eval_max_len), PAD, dtype=torch.long)
        pim = torch.ones((B, eval_max_len), dtype=torch.bool)
        sg = torch.full((B, 8, D), PAD, dtype=torch.long)
        sgm = torch.ones((B, 8, D), dtype=torch.bool)
        sgp = torch.ones((B, 8), dtype=torch.bool)
        targets = torch.zeros(B, dtype=torch.long)
        for b, p in enumerate(chunk):
            lw = corpus_tokens[max(0,p-eval_max_len+1):p+1]
            pt = lw[-eval_max_len:]
            pi[b,:len(pt)] = torch.tensor(pt, dtype=torch.long)
            pim[b,:len(pt)] = False
            targets[b] = corpus_tokens[p+1]
        pi,pim,sg,sgm,sgp,targets = (x.to(DEVICE) for x in (pi,pim,sg,sgm,sgp,targets))
        os.environ["P2S_DIRECT"] = "1"
        logits, _ = model(pi, pim, sg, sgm, sgp)
        loss = F.cross_entropy(logits, targets, reduction='sum').item()
        total_loss += loss; total_n += B
    nll = total_loss / total_n
    print(f"[{label}] n={total_n}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}", flush=True)
    return nll, math.exp(nll)

print("\n=== ALL POSITIONS ===")
print("baseline (no mask):")
nll_all_no, ppl_all_no = eval_no_mask(eval_pos_all, "all-noprior")
print("with word-aligned mask:")
nll_all_yes, ppl_all_yes = eval_with_mask(eval_pos_all, "all-wordaligned")

print("\n=== SPACE-ENDING POSITIONS ===")
print("baseline (no mask):")
nll_sp_no, ppl_sp_no = eval_no_mask(eval_pos_space, "space-noprior")
print("with word-aligned mask:")
nll_sp_yes, ppl_sp_yes = eval_with_mask(eval_pos_space, "space-wordaligned")

print(f"\n=== Δ from word-aligned mask ===")
print(f"  all positions:    {ppl_all_no:.2f} → {ppl_all_yes:.2f}  (Δ {ppl_all_yes-ppl_all_no:+.3f})")
print(f"  space-ending:     {ppl_sp_no:.2f} → {ppl_sp_yes:.2f}  (Δ {ppl_sp_yes-ppl_sp_no:+.3f})")
