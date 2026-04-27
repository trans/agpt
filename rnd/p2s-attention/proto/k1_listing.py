"""For each D, find ALL prefix/suffix entries with overlap=1 (single-char
radix overlap at the junction). Show them formatted as
  «prefix-pre-overlap[OVERLAP_CHAR]suffix-post-overlap»
so the single overlap char is visible right at the boundary.
"""
import sys, os, random
sys.path.insert(0, "/home/trans/Projects/agpt/rnd/p2s-attention/proto")
from p2s_train import load_radix_tree, path_tokens, iter_match_records

CORPUS_PATH = "/home/trans/Projects/agpt/data/gutenberg_5m.txt"
N_PER_D = 15
SEED = 11

text = open(CORPUS_PATH).read()
chars_sorted = sorted(set(text))
tok_to_char = {i: c for i, c in enumerate(chars_sorted)}

def render(tokens):
    out = []
    for t in tokens:
        c = tok_to_char.get(t, '?')
        if c == '\n': out.append('\\n')
        elif c == '\t': out.append('\\t')
        elif c == '\r': out.append('\\r')
        else: out.append(c)
    return ''.join(out)

def listing_for_d(D):
    pref_dir = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_radix_corpus"
    suff_dir = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_suffix_radix"
    match_path = f"/home/trans/agpt-tries/g5m_d{D}_p2s_match.bin"
    if not os.path.exists(match_path):
        print(f"[skip] D={D}: no match index built")
        return

    print(f"\n{'═' * 100}")
    print(f"D = {D}    (entries with overlap k = 1)")
    print('═' * 100)

    pp, pe, _, _ = load_radix_tree(pref_dir)
    sp, se, _, _ = load_radix_tree(suff_dir)

    # Find ALL k=1 entries
    k1_entries = []
    n_total = 0
    for pid, max_k, sigmas in iter_match_records(match_path):
        n_total += 1
        if max_k == 1:
            k1_entries.append((pid, sigmas))

    print(f"  {len(k1_entries)} entries with k=1 out of {n_total} total "
          f"({100 * len(k1_entries) / n_total:.3f}%)\n")

    rng = random.Random(SEED + D)
    samples = rng.sample(k1_entries, min(N_PER_D, len(k1_entries)))

    for pid, sigmas in samples:
        pi_path = path_tokens(pid, pp, pe)
        if len(pi_path) < 1:
            continue
        # Single-char overlap = π's last char = σ's first char
        prefix_pre = pi_path[:-1]
        overlap_char = pi_path[-1]

        if len(sigmas) == 1:
            # Single candidate — show the joined glue
            sigma_path = path_tokens(sigmas[0], sp, se)
            sigma_fwd = list(reversed(sigma_path))
            if not sigma_fwd:
                continue
            post = sigma_fwd[1:]
            print(f"  «{render(prefix_pre)}[{render([overlap_char])}]{render(post)}»  "
                  f"({len(sigmas)} cand)")
        else:
            # Multi-candidate — show prefix once, then each σ continuation
            print(f"  «{render(prefix_pre)}[{render([overlap_char])}]…»  "
                  f"({len(sigmas)} candidates)")
            shown = 0
            for sid in sigmas[:3]:
                sigma_path = path_tokens(sid, sp, se)
                sigma_fwd = list(reversed(sigma_path))
                if not sigma_fwd:
                    continue
                post = sigma_fwd[1:]
                print(f"      → «[{render([overlap_char])}]{render(post)}»")
                shown += 1
            if len(sigmas) > shown:
                print(f"      → ... and {len(sigmas) - shown} more candidates")

# Run from D=32 downward (only D values where we have the match index)
for D in [32, 30, 24, 16]:
    listing_for_d(D)
