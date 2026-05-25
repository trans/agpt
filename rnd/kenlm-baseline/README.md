# KenLM KN Baseline + Per-Trie-Node Distribution Extractor

**Date:** 2026-05-24
**Status:** Three tools live: KN baseline sweep, trie-context dumper, per-node KN distribution extractor. Pipeline is ready for the "KN as soft target" experiment.

## Why

1. **Baseline sweep.** nltk's pure-Python KN is too slow above order 6 on 5M-char corpora (hours, multi-GB RAM). KenLM trains any order in seconds, letting us probe the upper end of the KN curve and see where it plateaus.
2. **Per-node distributions.** A possible AGPT extension is to use KN-smoothed P(w | context) at each trie node as a label-smoothing target during training (or to mix with one-hot targets via interpolation). The full pipeline produces these distributions as a side-table.

## Tools

| tool | purpose |
|---|---|
| `src/tools/kenlm_baseline.sh` | Train ARPA + report PPL for any KN order |
| `bin/dump_trie_contexts` (Crystal) | Walk a radix trie, emit per-node context string (one per line) |
| `src/tools/kn_extract_distributions.py` | Read ARPA + contexts → emit P(w \| context) side-table |

## Critical tokenization fix

Initial KenLM runs missed the **space** character entirely. KenLM treats whitespace as a token delimiter, so the original `' '.join(c for c in text)` tokenization caused the corpus space char itself to be collapsed into the delimiter — the model never saw "space" as a vocabulary item and queries like P(' ' | "the") returned ~0.

**Fix:** substitute spaces with `'_'` before tokenization. Real corpus spaces are now represented as `'_'` tokens in the model; downstream tools must remember the mapping. Applied in both `kenlm_baseline.sh` and `kn_extract_distributions.py`.

## Results (corrected, with space-fix)

**Gutenberg 5M, proper held-out (`/tmp/gut_holdout_proper.txt`), `--discount_fallback`:**

| order | context (chars) | PPL | wall (build+eval) |
|---|---|---|---|
| 3 | 2 | 7.850 | <1s |
| 4 | 3 | 5.524 | 1s |
| 6 | 5 | 4.155 | 3s |
| **8** | **7** | **4.089** ⭐ (plateau) | 5s |
| 10 | 9 | 4.131 | 7s |
| 12 | 11 | 4.145 | 9s |

Tokens scored: 196,305 (vs 164,548 pre-fix — the missing 32K were the space tokens).

**Reference points:**
- nltk KN order 6 (no fallback): 3.960
- AGPT L=8 d=128 100 SE seed 1: **3.6945** (current best)

## What the curve tells us

- **KN plateaus at order ~8** (PPL 4.09). Orders 10/12 slightly worse — overfit noise from `--discount_fallback` on rare long n-grams. Past 7-char context, classical KN extracts no additional value at this corpus size.
- **The plateau is the bar AGPT needs to clear.** AGPT at L=8 d=128 hits 3.69 — about **10% below KN's plateau**. With d=16 trie context (15 chars), AGPT extracts information KN cannot.
- **Why `--discount_fallback`?** Modified KN needs n-grams with counts 1, 2, and 3 to compute three discount parameters. Char-level corpora (65 unique chars in 5M chars) have no rare unigrams — every char appears hundreds of thousands of times. `--discount_fallback` substitutes a single fixed discount of 0.5 across all orders. Tradeoff: ~5% PPL penalty vs ideal smoothing (nltk's full implementation lands at 3.96 vs our 4.16 at order 6).

## Pipeline for per-node KN distributions

The intended downstream use is "KN-distilled targets" for AGPT training. For each radix-trie node (representing a unique context), compute P(w | context) for all 65 vocab items. Store as a side-table; load during training and use as a soft target alongside the one-hot.

```sh
# 1. Build KN ARPA from the same training corpus
src/tools/kenlm_baseline.sh data/gutenberg_5m.txt /dev/null 8 2>&1 | tail -2 # for PPL only
# (or skip the wrapper and call lmplz directly to keep the ARPA)
python3 -c "
with open('data/gutenberg_5m.txt') as f: t = f.read().replace(' ', '_').replace('\n', ' ')
print(' '.join(c for c in t if c != ' '))
" > /tmp/kn_train.txt
tools/kenlm/build/bin/lmplz -o 8 --discount_fallback < /tmp/kn_train.txt > /tmp/kn_o8.arpa

# 2. Dump every trie node's context as one line per node
bin/dump_trie_contexts /tmp/gutenberg_5m_baseline_d16_radix \
    --vocab data/gutenberg_5m.txt > /tmp/trie_contexts.txt

# 3. For each context, score P(w|context) for all vocab w — emit side-table
src/tools/kn_extract_distributions.py \
    --arpa /tmp/kn_o8.arpa \
    --contexts-file /tmp/trie_contexts.txt \
    --vocab data/gutenberg_5m.txt \
    --out /tmp/kn_distributions.bin
```

**Cost (Gutenberg 5M, d=16 trie):**
- Dump contexts: ~30 sec (2-3M nodes)
- Build ARPA: ~3 sec
- Extract distributions: ~5 min (Python pure-loop, ~10K ctx/sec)
- Total: ~5 min
- Output: ~600-800 MB (2M nodes × 65 floats × 4 bytes)

## Side-table format (`kn_distributions.bin`)

```
magic:    u32 = 0x4B4E4453 ("KNDS")
n_ctx:    u32  number of contexts (rows)
vocab:    u32  size of vocab
chars:    u8[vocab]  the char vocabulary (sorted); '_' represents space
probs:    float32[n_ctx * vocab]  P(w | context) row-major, linear domain
```

Each row sums to 1.0 (renormalized after KN backoff). Reading from Crystal/Python is straightforward: skip the 12-byte header + vocab bytes, then mmap or read the float matrix.

## Build setup (one-time)

KenLM ships with `KENLM_MAX_ORDER=6` baked in and references the obsolete `boost_system` library (made header-only in Boost 1.69). Rebuild:

```sh
git clone https://github.com/kpu/kenlm.git tools/kenlm
sed -i '/^  system$/d' tools/kenlm/CMakeLists.txt  # drop boost_system
cd tools/kenlm && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=12
make -j8
```

Build deps: cmake, g++, boost (program_options, thread, unit_test_framework), eigen3, zlib, bzip2, lzma. All already present on the dev machine.

The Python `kenlm` pip wrapper does NOT work on Python 3.14 (uses removed `_PyGen_SetStopIterationValue`). Our ARPA parser at `src/tools/kn_extract_distributions.py` is pure Python and avoids this.

## Cross-tool divergence (still relevant)

| evaluator | KN order | PPL on proper held-out |
|---|---|---|
| nltk `KneserNeyInterpolated` | 6 | 3.960 |
| KenLM `lmplz --discount_fallback` (space-fix) | 6 | 4.155 |
| KenLM `lmplz --discount_fallback` (space-fix) | 8 | 4.089 (plateau) |

Gap reduced from ~30% (pre-fix) to ~5% (post-fix). Still nonzero — driver is the single-fallback-discount vs nltk's full modified-KN. **Pick one tool per comparison; don't mix.** For headline AGPT-vs-KN narrative, nltk's 3.96 remains the more aggressive baseline. KenLM is for trend analysis + the per-node distribution pipeline.

## Files

- `src/tools/kenlm_baseline.sh` — wrapper script for KN PPL sweep
- `src/tools/dump_trie_contexts.cr` (Crystal) → `bin/dump_trie_contexts`
- `src/tools/kn_extract_distributions.py` — pure-Python ARPA parser + extractor
- `tools/kenlm/` — kenlm clone, built with KENLM_MAX_ORDER=12 + boost-system patch
- `cmake-boost.patch` — minimal CMakeLists diff for modern Boost compatibility

## Open follow-ups

- Wire `kn_distributions.bin` as a soft-target side-load in agpt_train (the actual distillation experiment)
- Decide whether to mix or replace: best results likely from `α·CE(one-hot) + (1-α)·KL(P_kn || model)` with α=0.5-0.9
- If the experiment looks promising and we want absolute KN numbers we trust, implement modified-KN in Crystal or use a small-corpus-aware discount estimator — the `--discount_fallback` ceiling isn't tight
- Per-node entropy histogram: which trie nodes have high-entropy KN distributions (low confidence)? These are the ones distillation would help most
