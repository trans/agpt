# Trie-as-Attention Framing — Decision/Identity Decomposition

> **Status (2026-04-28): closed.** Descriptive predictions confirmed across
> multiple corpora; prescriptive operationalizations all came in
> neutral-to-marginally-negative under the recipes tested. The framing's
> empirical reach and a clean limit on its prescriptive power are both
> documented. See `findings.md` for the full closeout. Artifacts and flags
> retained for future reference.

## Hypothesis

The radix trie over corpus prefixes structurally decomposes into two zones
that map onto Q/K/V attention:

```
d_optimal ≈ d_decision + d_identity
            ↑            ↑
            log₂(N)/H    set by phrase/clause length in English (≈ 21 chars)
            corpus-dep   corpus-INDEPENDENT
```

- **K = decision** (root-side ~11 chars, branching state). What positions are
  matchable by — "I'm at the same kind of branch you were at."
- **V = identity** (leaf-side ~21 chars, unique fingerprint pointing to the
  passage). What gets retrieved once K matches.
- **Q** lives in K's space — the current decision-state question.

The asymmetry that matters is **K vs V** (decision-space vs identity-space),
not Q vs K (which are partners in ask vs be-asked).

## What was tested

### Descriptive predictions (held)

1. **d* (branching depth) scales as log₂(N)/H** across 50× corpus-size
   range:

| Corpus | log₂(N)/2 predicted | Observed mean d* |
|---|---:|---:|
| Shakespeare 100k | 8.31 | 7.94 |
| Shakespeare 1M   | 10.04 | 9.71 |
| Gutenberg 5M     | 11.15 | 11.23 |

2. **d=32 is the sweet spot for English at 1-5M corpus size.** AGPT d-sweep
   on Shakespeare 1M:

| d | PPL | Δ vs d=32 |
|---|---:|---:|
| 16 | 15.37 | +2.38 |
| 32 | 12.99 | — |
| 48 | 12.94 | −0.05 |

Pattern matches: d=16 deficient (insufficient identity zone), d=32 optimal
(decision + identity zones both filled), d=48 saturated.

3. **Decision events carry ~97% of the learning** at 8% of training compute
(decision-only ablation, see findings.md).

### Prescriptive operationalizations (none beat baseline)

- **Static depth-routing** (mask Wk-grad past depth k, Wv-grad below k):
  k=11 within noise, k=7 lucky outlier didn't replicate, k=20 clearly
  worse. AGPT d=32 baseline 13.57 PPL, k=11 13.78 PPL.
- **Per-leaf d* routing** (variable threshold per node from the trie's
  structural d_split): borderline marginal-positive (gap 0.29 PPL,
  p≈0.07 across-session, p≈0.37 within-session, n=6 vs n=7).
- **Decision-only loss** (skip CE at queries past d_split): clean
  diminishing-returns curve — 8.2% events captures 96.8% of baseline
  learning, but no buffer config beat baseline at 3 epochs. Matched-compute
  experiment confounded by recipe overfitting at extended epochs.
- **Microgpt SGD parallel**: same depth-routing flag added to standard
  SGD trainer; clearly negative at seq_len=128 (+11.6%) and seq_len=32
  (+7.3%). The longer the context, the worse the routing.

The strict architectural realization (drop Wv entirely, V as
corpus-tail-lookup, decision-paced inference) was not built — it requires
a real architectural rebuild rather than a flag.

## Follow-up (2026-04-29): joint-mass per-position

After this directory's initial closeout, we extended joint-mass weighting
from the aggregate per-depth-mean proxy to true per-position lookups —
walking the suffix tree from each corpus position to compute the actual
suffix-mass at the complementary depth. Implemented as
`AGPT_JOINT_MASS=1` + `AGPT_CHAR_SUFFIX_MASS_PATH=<table.bin>` env vars,
where the precomputed table comes from
`proto/compute_char_suffix_mass.py`.

Result at 3-SE recipe: per-position joint-log gives **12.55 PPL** vs
plain log mass-weight **12.67 PPL** — marginal improvement (1%, p ≈ 0.5
not significant). The per-position resolution captures real
per-position variation (e.g. mean csm at d_p=16 is 1.48 with max 236,
vs the aggregate proxy's 1.03 mean) but doesn't translate into
significant PPL gain at this training budget.

**Important caveat from `../agpt-epoch-scaling/`:** all the joint-mass
results above were measured at 3 SE, which we discovered is severely
undertrained. At 20+ SE, AGPT achieves <7 PPL with the basic recipe —
twice as good as our "best" jmpp number. The joint-mass effect's
relative size and significance may shift dramatically under proper
training; this needs re-measurement before any conclusion about
joint-mass's prescriptive value can stand.

## Layout

- `findings.md` — full result tables, diagnoses, and the supervision-signal
  critique that explains why naive routing under existing CE loss can't
  realize the framing's V=identity claim.
- `run_dstar.sh` — reproduces d* analysis on Shakespeare 1M, 100k, and
  Gutenberg 5M (uses `proto/branching_depth.py`).
- `run_d_sweep.sh` — d=16 / d=32 / d=48 AGPT training on Shakespeare 1M.
- `run_depth_route_static.sh` — `AGPT_DEPTH_ROUTE_K` k-sweep.
- `run_depth_route_perleaf.sh` — `AGPT_DEPTH_ROUTE_PERLEAF` head-to-head
  vs baseline.
- `run_decision_only.sh` — `AGPT_DECISION_ONLY` + `AGPT_DECISION_BUFFER`
  buffer sweep.
- `proto/branching_depth.py` — parameterized d* analyzer (path argument).
- `logs/NOTE.md` — note that `/tmp/dr_*.log` artifacts from the live
  session were lost on reboot; result tables were preserved in this
  README and findings.md.

## Flags left in place (default off, no behavioral change)

In `agpt_train.cu`:
- `AGPT_DEPTH_ROUTE_K=N` — static threshold, mask dWk past k, dWv at/below k
- `AGPT_DEPTH_ROUTE_PERLEAF=1` — variable threshold from per-node d_split
- `AGPT_DECISION_ONLY=1` — zero loss + grad past d_split
- `AGPT_DECISION_BUFFER=N` — soften decision-only by keeping events for N
  chars past d_split

In `microgpt`:
- `--depth-route-k N` flag wired through to `MultiHeadAttention.backward`

The d_split precompute runs at trie load with no measurable cost.

## See also

- `../agpt-epoch-scaling/` — discovery that AGPT was severely undertrained
  at the 3-SE budget used for all experiments here. Pure AGPT at 20-40 SE
  achieves PPL@32 in the 5-7 range, far below this directory's "best"
  ~10.83. Re-running the joint-mass / depth-routing comparisons at 20+ SE
  would be the proper follow-up.
- `../subtree-dropout/` — the experiment that surfaced the undertraining.
  Random root-child masking helps slightly at low-SE budgets, vanishes
  at high-SE.

## Future directions (not pursued)

- **Re-measure all prescriptive operationalizations at 20+ SE.** Most
  important follow-up — the effects measured here may shift significantly
  under proper training.
- **Strict V=corpus-lookup architecture.** Drop Wv as a learnable parameter;
  V_b at branch point b becomes the embedding of the first char of the
  unary cap. Train Wq/Wk/Wo only. Decision-paced inference (emit unary tail
  in one step after attention selects). This is the actual realization of
  the framing's prescriptive claim; it requires architectural rebuild.
- **Soft-anneal routing.** Gradient weighting via sigmoid(α·(d_k − d))
  instead of hard mask. Would preserve all training events while still
  biasing the K and V projections toward their roles.
- **KL distillation against empirical π.** Train softmax(Q·K) directly
  against the trie's empirical branching distribution rather than CE
  on next-char.
- **Per-layer routing.** Different d_k per layer; possibly different
  routing strategies per layer.
- **Recipe re-tuning for matched-compute.** Use constant LR + weight decay
  to remove the overfitting confound that prevented clean matched-compute
  testing of decision-only.

## Reproduce headline results

```sh
# d* analysis (cheap, ~2 sec each)
bash rnd/trie-attention-framing/run_dstar.sh

# AGPT d-sweep (each ~80 sec)
bash rnd/trie-attention-framing/run_d_sweep.sh

# Depth-routing static k-sweep (~80 sec each, 8 runs)
bash rnd/trie-attention-framing/run_depth_route_static.sh

# Per-leaf d* head-to-head (~80 sec each, 6 runs)
bash rnd/trie-attention-framing/run_depth_route_perleaf.sh

# Decision-only buffer sweep (~80 sec each, 6 runs)
bash rnd/trie-attention-framing/run_decision_only.sh
```
