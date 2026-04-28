"""Build a word-aligned variant of the match index.

For each leaf, truncate its path-to-leaf at the LAST SPACE within the
leaf-edge zone. The truncated form is what gets used for matching, so
all matches structurally end at a word boundary.

For prefix-tree leaves: truncate path so it ends at last space (kept).
For suffix-tree leaves: take forward path, truncate at last space.

Then build a "head trie" from suffix truncated forwards, match prefix
truncated paths against it. Compare distribution to the original D=32
match index.
"""
import sys, os, struct, time
from collections import defaultdict
sys.path.insert(0, "/home/trans/Projects/agpt/rnd/p2s-attention/proto")
from p2s_train import load_radix_tree, path_tokens, iter_match_records

PREFIX_DIR = "/home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus"
SUFFIX_DIR = "/home/trans/agpt-tries/gutenberg_5m_d32_suffix_radix"
CORPUS_PATH = "/home/trans/Projects/agpt/data/gutenberg_5m.txt"

text = open(CORPUS_PATH).read()
chars_sorted = sorted(set(text))
char_to_tok = {c: i for i, c in enumerate(chars_sorted)}
SPACE_TOK = char_to_tok[' ']
print(f"space token = {SPACE_TOK}")

print("Loading prefix tree...", flush=True)
pp, pe, vocab_size, _ = load_radix_tree(PREFIX_DIR)
print("Loading suffix tree...", flush=True)
sp, se, _, _ = load_radix_tree(SUFFIX_DIR)

# Identify leaves
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
print(f"  prefix leaves: {len(p_leaves)}")
print(f"  suffix leaves: {len(s_leaves)}")

print("\nComputing truncated prefix paths (clip at last space)...", flush=True)
t0 = time.time()
truncated_prefix = {}
no_space_count = 0
length_dist_p = defaultdict(int)
for leaf in p_leaves:
    path = path_tokens(leaf, pp, pe)
    # find last space position in path
    last_sp = -1
    for i in range(len(path) - 1, -1, -1):
        if path[i] == SPACE_TOK:
            last_sp = i
            break
    if last_sp < 0:
        no_space_count += 1
        truncated_prefix[leaf] = path  # no space found; keep as-is
    else:
        truncated_prefix[leaf] = path[:last_sp + 1]   # up to and including space
    length_dist_p[len(truncated_prefix[leaf])] += 1
print(f"  done in {time.time() - t0:.1f}s, {no_space_count} prefixes had no space")

print("Computing truncated suffix forward paths (clip at last space)...", flush=True)
t0 = time.time()
truncated_suffix = {}
no_space_s = 0
length_dist_s = defaultdict(int)
for leaf in s_leaves:
    sigma_path = path_tokens(leaf, sp, se)
    sigma_fwd = sigma_path[::-1]   # reverse to forward corpus order
    last_sp = -1
    for i in range(len(sigma_fwd) - 1, -1, -1):
        if sigma_fwd[i] == SPACE_TOK:
            last_sp = i
            break
    if last_sp < 0:
        no_space_s += 1
        truncated_suffix[leaf] = sigma_fwd
    else:
        truncated_suffix[leaf] = sigma_fwd[:last_sp + 1]
    length_dist_s[len(truncated_suffix[leaf])] += 1
print(f"  done in {time.time() - t0:.1f}s, {no_space_s} suffixes had no space")

# Stats: length distribution of truncated edges
print("\nTruncated prefix path-length distribution:")
for L in sorted(length_dist_p):
    if length_dist_p[L] >= 1000:  # skip rare lengths
        pct = 100.0 * length_dist_p[L] / len(p_leaves)
        print(f"  len={L:2d}: {length_dist_p[L]:8d} ({pct:5.2f}%)")

# Build head trie from truncated suffix forwards
# Node: (char, first_child, next_sibling, terminal_list_head)
print("\nBuilding head trie from truncated suffix forwards...", flush=True)
t0 = time.time()
# Memory-conscious: use array.array for compact int storage
import array
node_char = array.array('i', [-1])
node_first_child = array.array('i', [-1])
node_next_sib = array.array('i', [-1])
node_first_term = array.array('i', [-1])
term_sigma = array.array('i')
term_next = array.array('i')

n_inserted = 0
total_s = len(truncated_suffix)
for sleaf, fwd in truncated_suffix.items():
    cur = 0
    for ch in fwd:
        child = node_first_child[cur]
        while child != -1:
            if node_char[child] == ch:
                break
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
        print(f"  inserted {n_inserted}/{total_s} suffixes, head trie has {len(node_char)} nodes",
              flush=True)
print(f"  head trie: {len(node_char)} nodes, {len(term_sigma)} terms, "
      f"{time.time() - t0:.1f}s", flush=True)
# Free input dicts since we have head trie now
truncated_suffix = None
import gc; gc.collect()

# Match prefix truncated paths' tails against head trie
print("\nMatching prefix truncated tails to head trie...", flush=True)
t0 = time.time()
overlap_hist = defaultdict(int)
size_hist = defaultdict(int)
n_matched = 0
n_no_match = 0

for pleaf, ppath in truncated_prefix.items():
    L = len(ppath)
    if L == 0:
        n_no_match += 1
        continue
    max_k = 0
    matched_node = -1
    # Try k from L down to 1
    k = L
    while k >= 1:
        cur = 0
        ok = True
        for i in range(k):
            ch = ppath[L - k + i]
            child = node_first_child[cur]
            while child != -1:
                if node_char[child] == ch:
                    break
                child = node_next_sib[child]
            if child == -1:
                ok = False
                break
            cur = child
        if ok:
            max_k = k
            matched_node = cur
            break
        k -= 1
    if max_k == 0:
        n_no_match += 1
        overlap_hist[0] += 1
        continue
    # Count terminals in subtree under matched_node (capped)
    stack = [matched_node]
    n_cands = 0
    while stack and n_cands < 64:
        n = stack.pop()
        t = node_first_term[n]
        while t != -1 and n_cands < 64:
            n_cands += 1
            t = term_next[t]
        c = node_first_child[n]
        while c != -1:
            stack.append(c)
            c = node_next_sib[c]
    overlap_hist[max_k] += 1
    if n_cands == 1:
        size_hist['1'] += 1
    elif n_cands <= 4:
        size_hist['2-4'] += 1
    elif n_cands <= 19:
        size_hist['5-19'] += 1
    else:
        size_hist['20+'] += 1
    n_matched += 1

print(f"  done in {time.time() - t0:.1f}s")
print(f"\n=== Word-aligned match summary (D=32 truncated at last space) ===")
print(f"  prefix leaves total:     {len(p_leaves)}")
print(f"  matched (k>=1):          {n_matched}")
print(f"  no match:                {n_no_match}")

print(f"\n=== Word-aligned overlap k histogram ===")
total = sum(v for k, v in overlap_hist.items() if k > 0)
ks = sorted(k for k in overlap_hist.keys() if k > 0)
for k in ks:
    pct = 100.0 * overlap_hist[k] / total
    bar = '█' * min(60, int(pct))
    print(f"  k={k:2d}: {overlap_hist[k]:8d} ({pct:5.2f}%) {bar}")

print(f"\n=== Word-aligned candidate-set size histogram ===")
labels = ['1', '2-4', '5-19', '20+']
for lab in labels:
    if size_hist[lab] > 0:
        pct = 100.0 * size_hist[lab] / n_matched
        print(f"  {lab:>5s}: {size_hist[lab]:8d} ({pct:5.2f}%)")
