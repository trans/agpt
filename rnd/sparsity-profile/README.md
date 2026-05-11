# Trie Sparsity Profile

**Status**: complete — depth-by-depth sparsity profile measured.

**Trainer note**: not obviously trainer-dependent.

**Tool**: `bin/trie-profile <radix_dir>` (built from `src/tools/trie_profile.cr`).

## Hypothesis

The D-trie on a finite corpus becomes sparse past some depth: beyond that depth,
most radix endpoints are singletons (unique corpus paths, no real branching).
The "useful signal depth" should fall somewhere between "fully dense" (shallow)
and "fully unique" (deep). Measuring this directly tells us:
- Where training signal becomes noisy (sparse endpoints = high-variance targets).
- Why blending has a sweet spot at d=16 (empirically confirmed earlier).
- How much corpus information the D-trie actually carries.

## Method

Load each of three Shakespeare radix tries (d=8, d=16, d=32), iterate
endpoint-depth files, compute per-depth:
- `n_nodes`: number of radix endpoints at that depth.
- `total_count`: sum of corpus counts across all endpoints at that depth.
- `median_count`, `mean_count`, `max_count`: distribution of per-endpoint counts.
- `%singletons`: fraction of endpoints with a single observed next-token
  (effectively deterministic).
- `avg_branch_factor`: mean number of distinct next-tokens per endpoint.

Raw tables in `results/d8.txt`, `results/d16.txt`, `results/d32.txt`.

## Findings

**Shakespeare (1.1M chars, vocab_size 65):**

| depth | n_nodes  | mean_count | %singletons | avg_branch |
|---|---|---|---|---|
| 1     | 62       | 17,980     | 0%          | 22.58 |
| 2     | 1,139    | 968        | 0%          | 9.91  |
| 4     | 25,187   | 38.5       | 0%          | 4.59  |
| 8     | 75,929   | 5.17       | 0%          | 2.85  |
| 12    | 25,759   | 3.16       | 0%          | 2.42  |
| 16 (d=32 interior) | 6,119    | 3.02       | 0%          | 2.34  |
| 20    | 2,096    | 3.17       | 0%          | 2.43  |
| 24    | 691      | 2.34       | 0%          | 2.17  |
| 28    | 231      | 2.20       | 0%          | 2.11  |
| 31    | 132      | 2.14       | 0%          | 2.06  |

**Plus the CAP depth of each trie** (where all "ran off the end" paths terminate):

| Trie  | Cap depth | n_nodes    | %singletons | mean_count |
|---|---|---|---|---|
| d=8   | 8         | 609,659    | **87.55%**  | 1.83 |
| d=16  | 16        | 1,081,061  | **99.43%**  | 1.03 |
| d=32  | 32        | 1,114,330  | **99.99%**  | 1.00 |

### What this says

1. **Interior depths carry meaningful structure all the way to depth 31** —
   the mean count stays 2-3 and branching factor stays 2-2.5 even at depth 28,
   30, 31 (in d=32). These are mostly-mass-2 nodes with real multi-token
   distributions.

2. **The CAP absorbs singletons**. At the max depth of each trie, the
   overwhelming majority (87-99%) of nodes are singletons — literally unique
   corpus 8-grams, 16-grams, or 32-grams. The deeper the cap, the closer to
   100% singleton.

3. **"Past depth 12-16 the trie becomes sparse" — partially confirmed.** The
   sparsity isn't a smooth decline; it's concentrated at the CAP. The
   *interior* depths don't dramatically sparsify — they just have fewer and
   fewer nodes, but each one still has meaningful 2-3 mean count.

4. **The cap's singleton nature explains d=32 blending's failure.** At d=32,
   most training queries are at depth 32 (1.1M of them, 99.99% singletons).
   The "true" target there is a singleton (= "the next corpus char in this
   specific occurrence"). Blending a singleton with shallower distributions
   actively dilutes a correct deterministic signal.

5. **d=16's sweet spot explained.** At d=16, the cap is 99.43% singleton,
   but interior depths 8-15 have real multi-token distributions with mean
   counts 5 → 3. Blending's weighted sum across these interior depths
   captures meaningful uncertainty information. At d=8, the cap is only
   87.55% singleton (lots of real branching still at depth 8 — most paths
   haven't even reached unique corpus occurrences yet), so blending has
   less to add beyond what the baseline already captures.

6. **Branching factor collapses to ~2 by depth 10**. After depth 10 in
   Shakespeare, most radix endpoints are bi-branching (next token is one of
   ~2 possibilities with counts 2-3). This is where the corpus's "two-fork"
   structure lives — pairs of completions for a given context.

## Compression profile

Tool now also reports `avg_edge_len` (mean length of the unary-compressed edge
ending at each depth) and `chars_absorbed` (leveled-trie nodes collapsed into
that depth's incoming edges). Summary totals:

| Trie | radix endpoints | leveled nodes | compression | cap's share of absorbed chars |
|---|---|---|---|---|
| d=8  | 845,539   | 1,545,081  | 1.83×  | 78.7% |
| d=16 | 1,607,927 | 9,313,896  | 5.79×  | 93.7% |
| d=32 | 1,666,214 | 27,046,236 | **16.23×** | **97.9%** |

At d=32, the cap absorbs 24.8M of the 25.4M total absorbed characters. Each
cap endpoint has an avg_edge_len of 23.29 — meaning ~22 unary-chain depths
collapse into each cap edge on average.

Interior depths show a smooth gradient: avg_edge_len grows from 1.00 at depth 1
to 15.65 at depth 31, then explodes to 23.29 at the cap.

This says the trie's information structure is concentrated at the shallow-
interior branching points; everything past depth ~10 is mostly unique-tails
that compress efficiently into the cap.

## Implications for next experiments

- **Training budget should scale with depth differently than we've been doing.**
  The d=8 cap has 609K singletons + 74K real branching points. Training
  equally weights them; perhaps we should upweight the multi-branching
  positions to extract more learning per step.
- **The SGD-sanity-check idea** (random corpus windows as training inputs):
  the trie's aggregated gradient should match SGD at matched step budget.
  This profile suggests SGD would spend most of its gradient budget on
  cap-depth singletons (because those are where most corpus positions end
  up). Maybe an interesting test is whether AGPT's re-weighting compared to
  SGD matters at all.
- **Blending redesign**: instead of "blend at every endpoint," we could blend
  *only at non-singleton endpoints* (where branching factor > 1). Skip the
  cap's singletons entirely. That might preserve blending's benefit at
  d=16 while removing the harm at d=32.

## Files

- `results/d8.txt`, `results/d16.txt`, `results/d32.txt` — raw tool output.
