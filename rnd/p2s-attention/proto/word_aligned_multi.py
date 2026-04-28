"""Find the 0.7% genuine multi-candidate cases in word-aligned matching
and decode them for display. These should be linguistically-real branch
points: the prefix is a word-aligned phrase, and multiple corpus
positions show different word-aligned continuations."""

import sys, os, time, random, array
from collections import defaultdict
sys.path.insert(0, "/home/trans/Projects/agpt/rnd/p2s-attention/proto")
from p2s_train import load_radix_tree, path_tokens

PREFIX_DIR = "/home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus"
SUFFIX_DIR = "/home/trans/agpt-tries/gutenberg_5m_d32_suffix_radix"
CORPUS_PATH = "/home/trans/Projects/agpt/data/gutenberg_5m.txt"

text = open(CORPUS_PATH).read()
chars_sorted = sorted(set(text))
char_to_tok = {c: i for i, c in enumerate(chars_sorted)}
tok_to_char = {i: c for c, i in char_to_tok.items()}
SPACE_TOK = char_to_tok[' ']

def render(tokens):
    out = []
    for t in tokens:
        c = tok_to_char.get(t, '?')
        if c == '\n': out.append('\\n')
        elif c == '\t': out.append('\\t')
        elif c == '\r': out.append('\\r')
        else: out.append(c)
    return ''.join(out)

print("Loading prefix tree...", flush=True)
pp, pe, vocab_size, _ = load_radix_tree(PREFIX_DIR)
print("Loading suffix tree...", flush=True)
sp, se, _, _ = load_radix_tree(SUFFIX_DIR)

def leaves_of(parent_arr):
    is_p = [False] * len(parent_arr)
    for cid in range(1, len(parent_arr)):
        p = parent_arr[cid]
        if 0 < p < len(parent_arr):
            is_p[p] = True
    return [cid for cid in range(1, len(parent_arr)) if not is_p[cid]]

print("Identifying leaves...", flush=True)
p_leaves = leaves_of(pp)
s_leaves = leaves_of(sp)

def truncate_at_last_space(seq):
    last_sp = -1
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == SPACE_TOK:
            last_sp = i
            break
    if last_sp < 0:
        return list(seq)
    return list(seq[:last_sp + 1])

print("Computing truncated prefix paths...", flush=True)
truncated_prefix = {leaf: truncate_at_last_space(path_tokens(leaf, pp, pe))
                    for leaf in p_leaves}
print("Computing truncated suffix forwards...", flush=True)
truncated_suffix = {leaf: truncate_at_last_space(path_tokens(leaf, sp, se)[::-1])
                    for leaf in s_leaves}

print("Building head trie...", flush=True)
node_char = array.array('i', [-1])
node_first_child = array.array('i', [-1])
node_next_sib = array.array('i', [-1])
node_first_term = array.array('i', [-1])
term_sigma = array.array('i')
term_next = array.array('i')

for sleaf, fwd in truncated_suffix.items():
    cur = 0
    for ch in fwd:
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
print(f"  head trie nodes: {len(node_char)}", flush=True)

print("Matching and collecting multi-cand cases...", flush=True)
multi_cand_samples = []
n_seen = 0
for pleaf, ppath in truncated_prefix.items():
    n_seen += 1
    L = len(ppath)
    if L == 0: continue
    cur = 0
    ok = True
    for i in range(L):
        ch = ppath[i]
        child = node_first_child[cur]
        while child != -1:
            if node_char[child] == ch: break
            child = node_next_sib[child]
        if child == -1:
            ok = False; break
        cur = child
    if not ok: continue
    matched_node = cur
    # collect terminals in subtree (capped at 32 for display)
    sigmas = []
    stack = [matched_node]
    while stack and len(sigmas) < 32:
        n = stack.pop()
        t = node_first_term[n]
        while t != -1 and len(sigmas) < 32:
            sigmas.append(term_sigma[t])
            t = term_next[t]
        c = node_first_child[n]
        while c != -1:
            stack.append(c)
            c = node_next_sib[c]
    if len(sigmas) > 1:
        # also count true total
        total_n = 0
        stack2 = [matched_node]
        while stack2 and total_n < 1000:
            n = stack2.pop()
            t = node_first_term[n]
            while t != -1 and total_n < 1000:
                total_n += 1
                t = term_next[t]
            c = node_first_child[n]
            while c != -1:
                stack2.append(c)
                c = node_next_sib[c]
        multi_cand_samples.append((pleaf, ppath, sigmas, total_n))

print(f"\n{len(multi_cand_samples)} multi-cand entries collected.", flush=True)

# Pick samples to display, prioritizing variety in candidate count
random.seed(7)
small_multi  = [m for m in multi_cand_samples if 2 <= m[3] <= 4]
medium_multi = [m for m in multi_cand_samples if 5 <= m[3] <= 10]
large_multi  = [m for m in multi_cand_samples if m[3] > 10]
print(f"  2-4 cands:   {len(small_multi)}")
print(f"  5-10 cands:  {len(medium_multi)}")
print(f"  10+ cands:   {len(large_multi)}")

def show_samples(samples, n, label):
    print(f"\n=== {label} (showing {min(n, len(samples))}) ===\n")
    for pleaf, ppath, sigmas, total_n in random.sample(samples, min(n, len(samples))):
        prefix_str = render(ppath)
        print(f"  prefix: «{prefix_str}»  (n_cands ≥ {total_n})")
        seen_starts = set()
        for sigma_id in sigmas:
            sigma_path = path_tokens(sigma_id, sp, se)
            sigma_fwd = sigma_path[::-1]
            # show full forward path so we see what comes after the prefix
            sigma_str = render(sigma_fwd)
            # only show unique starts to avoid redundancy
            start_key = tuple(sigma_fwd[:len(ppath) + 6])
            if start_key in seen_starts: continue
            seen_starts.add(start_key)
            print(f"    → «{sigma_str}»")
            if len(seen_starts) >= 5: break
        print()

show_samples(small_multi, 8, "Genuine 2-4 candidate cases")
show_samples(medium_multi, 6, "5-10 candidate cases")
show_samples(large_multi, 4, "10+ candidate cases")
