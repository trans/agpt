"""Re-run the corpus-walk eval but filtered to word-boundary positions
(those where corpus[p] is a space). Compare with-prior vs without-prior
on this subset to see if the structural matching helps more at word
boundaries than overall.
"""
import sys, os, math, random, time
sys.path.insert(0, "/home/trans/Projects/agpt/rnd/p2s-attention/proto")
import torch
import torch.nn.functional as F
from p2s_train import P2SModel, load_radix_tree, path_tokens, iter_match_records

D = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX_DIR = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_radix_corpus"
SUFFIX_DIR = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_suffix_radix"
MATCH_PATH = f"/home/trans/agpt-tries/g5m_d{D}_p2s_match.bin"
CORPUS_PATH = "/home/trans/Projects/agpt/data/gutenberg_5m.txt"
CKPT_PATH = "/home/trans/Projects/agpt/rnd/p2s-attention/proto/p2s_model_d32_direct_ctx128.pt"
N_EVAL = 4096
SEED = 7

print("Loading checkpoint and trees...", flush=True)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
cfg = ckpt['config']
pp, pe, vocab_size, _ = load_radix_tree(PREFIX_DIR)
sp, se, _, _ = load_radix_tree(SUFFIX_DIR)

text = open(CORPUS_PATH).read()
chars_sorted = sorted(set(text))
char_to_tok = {c: i for i, c in enumerate(chars_sorted)}
SPACE_TOK = char_to_tok.get(' ', -1)
print(f"  space token = {SPACE_TOK}", flush=True)
corpus_tokens = [char_to_tok[c] for c in text]

# Build prefix-tree children index for walks
print("Building prefix-tree children index...", flush=True)
children = {}
for cid in range(1, len(pp)):
    e = pe[cid]
    if e is None or len(e) == 0: continue
    p = pp[cid]
    children.setdefault(p, []).append((e[0], cid, e))

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

print("Indexing match records by pid...", flush=True)
match_by_pid = {}
for pid, max_k, sigmas in iter_match_records(MATCH_PATH):
    match_by_pid[pid] = (max_k, sigmas)

# Build model
eval_max_len = cfg.get('max_len', cfg['D'])
model = P2SModel(
    vocab_size=cfg['vocab_size'], embed_size=cfg['vocab_size']+1,
    d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
    max_len=eval_max_len,
).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded, {sum(p.numel() for p in model.parameters())} params", flush=True)

# Sample positions where corpus[p] is a space (word boundary at the prefix's last char)
random.seed(SEED)
held_start = int(len(corpus_tokens) * 0.95)
all_positions = list(range(held_start + D, len(corpus_tokens) - 1))
space_positions = [p for p in all_positions if corpus_tokens[p] == SPACE_TOK]
random.shuffle(space_positions)
random.shuffle(all_positions)
eval_pos_space = space_positions[:N_EVAL]
eval_pos_all = all_positions[:N_EVAL]
print(f"Sampled {len(eval_pos_space)} space-ending and {len(eval_pos_all)} arbitrary positions",
      flush=True)

PAD = vocab_size
MAX_K = 8

@torch.no_grad()
def eval_subset(eval_positions, use_prior, label):
    items = []
    for p in eval_positions:
        win = corpus_tokens[max(0,p-D+1):p+1]
        if len(win) < D: continue
        leaf_id = find_leaf(win)
        if leaf_id == 0 or leaf_id not in match_by_pid: continue
        max_k, sigmas = match_by_pid[leaf_id]
        if not sigmas: continue
        sigmas_fwd = []
        cand_chars = []
        for sid in sigmas[:MAX_K]:
            sp_path = path_tokens(sid, sp, se)
            if not sp_path: continue
            sf = list(reversed(sp_path))
            if max_k >= len(sf): continue
            sigmas_fwd.append(sf)
            cand_chars.append(sf[max_k])
        if not sigmas_fwd: continue
        long_win = corpus_tokens[max(0,p-eval_max_len+1):p+1]
        target = corpus_tokens[p+1]
        items.append((long_win, sigmas_fwd, target, cand_chars))

    total_loss = 0.0; total_n = 0; applied = 0; skipped_oos = 0
    BATCH = 32
    for i in range(0, len(items), BATCH):
        chunk = items[i:i+BATCH]
        B = len(chunk)
        pi = torch.full((B, eval_max_len), PAD, dtype=torch.long)
        pim = torch.ones((B, eval_max_len), dtype=torch.bool)
        sg = torch.full((B, MAX_K, D), PAD, dtype=torch.long)
        sgm = torch.ones((B, MAX_K, D), dtype=torch.bool)
        sgp = torch.ones((B, MAX_K), dtype=torch.bool)
        targets = torch.zeros(B, dtype=torch.long)
        cand_chars_list = []
        for b, (lw, sf_list, tgt, cc) in enumerate(chunk):
            pt = lw[-eval_max_len:]
            pi[b,:len(pt)] = torch.tensor(pt, dtype=torch.long)
            pim[b,:len(pt)] = False
            for ki, st in enumerate(sf_list[:MAX_K]):
                std = st[:D]
                sg[b,ki,:len(std)] = torch.tensor(std, dtype=torch.long)
                sgm[b,ki,:len(std)] = False
                sgp[b,ki] = False
            targets[b] = tgt
            cand_chars_list.append(cc)
        pi,pim,sg,sgm,sgp,targets = (x.to(DEVICE) for x in (pi,pim,sg,sgm,sgp,targets))
        # Direct mode: bypass cross-attn
        os.environ["P2S_DIRECT"] = "1"
        logits, _ = model(pi, pim, sg, sgm, sgp)
        if use_prior:
            vs = logits.shape[1]
            mask = torch.ones((B, vs), dtype=torch.bool, device=DEVICE)
            tgt_cpu = targets.cpu().tolist()
            for b, cc in enumerate(cand_chars_list):
                ds = set(cc)
                if len(ds) > 1 and tgt_cpu[b] in ds:
                    in_set = torch.zeros(vs, dtype=torch.bool, device=DEVICE)
                    for c in ds:
                        if 0<=c<vs: in_set[c] = True
                    mask[b] = in_set
                    applied += 1
                elif len(ds) > 1:
                    skipped_oos += 1
            logits = logits.masked_fill(~mask, -1e9)
        loss = F.cross_entropy(logits, targets, reduction='sum').item()
        total_loss += loss
        total_n += B
    nll = total_loss / total_n
    print(f"[{label}] n={total_n}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}  "
          f"applied={applied} skipped_oos={skipped_oos}", flush=True)
    return nll

print("\n=== EVAL: arbitrary positions, NO prior ===")
nll_all_no = eval_subset(eval_pos_all, False, "all-noprior")
print("\n=== EVAL: arbitrary positions, WITH prior ===")
nll_all_yes = eval_subset(eval_pos_all, True, "all-withprior")
print("\n=== EVAL: space-ending positions, NO prior ===")
nll_sp_no = eval_subset(eval_pos_space, False, "space-noprior")
print("\n=== EVAL: space-ending positions, WITH prior ===")
nll_sp_yes = eval_subset(eval_pos_space, True, "space-withprior")

print(f"\n=== Δ from prior ===")
print(f"  arbitrary positions:    {math.exp(nll_all_no):.2f} → {math.exp(nll_all_yes):.2f}  "
      f"(Δ {math.exp(nll_all_yes) - math.exp(nll_all_no):+.3f})")
print(f"  space-ending positions: {math.exp(nll_sp_no):.2f} → {math.exp(nll_sp_yes):.2f}  "
      f"(Δ {math.exp(nll_sp_yes) - math.exp(nll_sp_no):+.3f})")
