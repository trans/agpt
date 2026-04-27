"""Generate a listing of prefix-overlap-suffix triples at varying D values.

Format per entry:
  «prefix-chars-before-overlap[OVERLAP_CHARS]suffix-chars-after-overlap»

The prefix's last k chars and suffix's first k chars are the same
(the structural overlap), shown bracketed in the middle.
"""
import sys, os, random, struct
sys.path.insert(0, "/home/trans/Projects/agpt/rnd/p2s-attention/proto")
from p2s_train import load_radix_tree, path_tokens, iter_match_records

CORPUS_PATH = "/home/trans/Projects/agpt/data/gutenberg_5m.txt"
N_PER_D = 12
SEED = 17

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

def list_for_d(D, target_n_candidates_filter=None):
    pref_dir = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_radix_corpus"
    suff_dir = f"/home/trans/agpt-tries/gutenberg_5m_d{D}_suffix_radix"
    match_path = f"/home/trans/agpt-tries/g5m_d{D}_p2s_match.bin"
    if not os.path.exists(match_path):
        return None
    print(f"\n{'═' * 100}", flush=True)
    print(f"D = {D}", flush=True)
    print('═' * 100, flush=True)

    pp, pe, _, _ = load_radix_tree(pref_dir)
    sp, se, _, _ = load_radix_tree(suff_dir)

    rng = random.Random(SEED + D)

    # First pass: collect a sample of records
    records = []
    for pid, max_k, sigmas in iter_match_records(match_path):
        records.append((pid, max_k, sigmas))

    # Filter to mass-1 (single-candidate) cases for cleanest demonstration
    # since those are 95% of matches at D=32 anyway
    single_cand = [(p, k, s) for p, k, s in records if len(s) == 1]
    multi_cand  = [(p, k, s) for p, k, s in records if len(s) > 1]
    print(f"  total: {len(records)} records ({len(single_cand)} single-cand, "
          f"{len(multi_cand)} multi-cand)", flush=True)

    print(f"\n--- {N_PER_D} single-candidate samples (D={D}) ---", flush=True)
    samples = rng.sample(single_cand, min(N_PER_D, len(single_cand)))
    for pid, max_k, sigmas in samples:
        pi_path = path_tokens(pid, pp, pe)
        sigma_path = path_tokens(sigmas[0], sp, se)
        sigma_fwd = list(reversed(sigma_path))
        # prefix's last max_k chars = overlap
        # suffix's first max_k chars = overlap
        if len(pi_path) < max_k or len(sigma_fwd) < max_k:
            continue
        pre_overlap = pi_path[:len(pi_path) - max_k]
        overlap    = pi_path[len(pi_path) - max_k:]
        post_overlap = sigma_fwd[max_k:]
        glued = (
            "«" + render(pre_overlap) +
            "[" + render(overlap) + "]" +
            render(post_overlap) + "»"
        )
        print(f"  k={max_k:2d}  {glued}", flush=True)

    if len(multi_cand) > 0:
        print(f"\n--- {min(4, len(multi_cand))} multi-candidate samples (D={D}) ---",
              flush=True)
        samples_m = rng.sample(multi_cand, min(4, len(multi_cand)))
        for pid, max_k, sigmas in samples_m:
            pi_path = path_tokens(pid, pp, pe)
            if len(pi_path) < max_k:
                continue
            pre_overlap = pi_path[:len(pi_path) - max_k]
            overlap = pi_path[len(pi_path) - max_k:]
            print(f"  k={max_k:2d}  «{render(pre_overlap)}[{render(overlap)}]…» "
                  f"({len(sigmas)} candidates)", flush=True)
            shown = 0
            for sid in sigmas[:3]:
                sigma_path = path_tokens(sid, sp, se)
                sigma_fwd = list(reversed(sigma_path))
                if len(sigma_fwd) < max_k:
                    continue
                post = sigma_fwd[max_k:]
                print(f"           → «[{render(overlap)}]{render(post)}»",
                      flush=True)
                shown += 1
            if len(sigmas) > 3:
                print(f"           → ... and {len(sigmas) - 3} more candidates",
                      flush=True)

# Run for each D we have
for D in [16, 24, 30, 32]:
    list_for_d(D)
