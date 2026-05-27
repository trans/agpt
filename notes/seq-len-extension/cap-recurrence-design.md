# Cap-recurrence design — working doc

**Date:** 2026-05-27.
**Branch:** `agpt-cap-recurrence`.
**Status:** Open-questions document. We resolve these before any code.

## Goal

Pass the cap hidden state `h_d` from the previous d-window back into the
root injection site of the next iteration. Decouple effective context
from depth `d` without growing memory.

This is the simpler-RNN version of the broader Bidirectional
Topological Automaton design (see `chord-position-encoding-design.md`
and the Gemini conversation that produced it). The full attention
version is a strict superset; this doc covers the minimal subset.

## Non-goals

- No chord / harmonic position encoding (separate work).
- No cycle-consistency / cross-tree consensus loss (full attention
  version's territory).
- No twin-model verification (already done: [[project_prefix_suffix_divergence]]
  found 33% top-1 agreement — synthesis regime confirmed).
- No replacement of standard prefix loss.
- No suffix-tree attention pool / pre-pruned candidate set.

## What's settled

- **h_cap exists per trie node**: current trainer already computes it
  at the deepest position of each prefix path during AGPT fires.
- **Gradient stops at the recurrence boundary**: incoming h_cap is a
  detached input constant. Model learns to USE incoming state; cannot
  propagate gradients through to past states.
- **Suffix tree provides predecessor structure**: for any K, the set
  of (K'_predecessor, count) pairs is enumerable from the existing
  suffix-radix data on disk.
- **Loss stays as standard prefix loss**: no new objective term.
- **Branch and dev setup**: separate branch (`agpt-cap-recurrence`),
  this doc fills first, then code.
- **The recurrence is epoch-temporal, NOT multi-stage**: there is
  no per-cap-state separate training pass. Each epoch is one full
  AGPT pass; the h_in table is read at each fire and refreshed at
  epoch end. Compute per epoch ≈ baseline AGPT (+ a small lookup
  cost). The "10^6 caps → 10^6 retraining passes → astronomical"
  picture does not arise.

## Open questions

### Q1: Injection form — RESOLVED: Option B (extra K-token at p=0)

Resolved 2026-05-27.

h_in enters the forward pass as an additional K/V slot prepended to the
attention sequence. Specifically:

1. Project h_in via two learned matrices: `K_inject = W_k_inject · h_in`,
   `V_inject = W_v_inject · h_in`. Each is d_model → d_model (per layer
   if we want per-layer specificity, or shared across layers for
   parameter economy — start shared).
2. Prepend (K_inject, V_inject) to the layer's existing KV sequence.
3. All Q's at positions ≥ 0 can attend to the injected slot through
   standard scaled-dot-product attention.
4. Position handling: the injected slot is at p=−1 conceptually, but
   since its position is fixed and there's only one of it, we treat
   it as position-encoding-free (no RoPE rotation applied). This makes
   it act as a position-agnostic memory token, which is what we want.
5. Causal mask: injected slot is always visible to all later positions.

Why over alternatives:
- Cleanest gradient story; well-understood in literature (Transformer-XL
  memory tokens).
- Preserves content channel — root token embedding stays unchanged.
- Bilinear Q · K_inject interaction gives per-pair learnability via Wq.
  Matches the empirical finding that per-pair amplitude control is
  load-bearing ([[project_rope_substitution_findings]]).
- Adds ~8K params per layer (W_k_inject + W_v_inject at d_model=64).
  Negligible relative to base model.

Rejected:
- A (additive bias): no per-pair learnability.
- C (FFN mix): architecturally invasive, touches every layer's FFN.
- D (gated replace): risks losing content if gate misbehaves.

### Q2: Aggregation across K's occurrences

K appears at many corpus positions. Each position has a different
preceding K' (different incoming state). What's K's effective h_in
during training?

This is the load-bearing design choice. Three options form an axis
from compute-cheap-info-lossy to compute-expensive-info-rich:

| Option | Form | Compute | Info preservation | Statistical efficiency |
|---|---|---|---|---|
| A | Mass-weighted average over suffix-tree predecessors: `h_in(K) = (1/Σc) Σ c_i · h_cap(K'_i)` | ~baseline | Lossy at shallow K (thousands of predecessors → mush). Sharp at deep K (few predecessors → preserved). | High — every K seen with one h_in value many times. |
| B | Per-fire stochastic sample: at each fire, sample one K' weighted by count | ~baseline | Preserved in expectation, noisy per fire. | Medium — variance hurts gradient quality. |
| C | Separate h_in per (K, K') pair | ~baseline (total corpus transitions ≈ trie node count for natural text) | Fully preserved. | LOW — each (K, K') seen only a few times in corpus. Model gradient signal spread too thin to learn how to USE h_in well. |
| **D** | **Persona/cluster table: cluster cap states into ~10^2–10^3 personas; each K stores soft mixture weights over personas** | **~baseline** | **Preserved at the granularity of meaningful state regimes (topic, genre, character mode). Loses sub-cluster variation.** | **High — each persona seen many times across all K's that use it.** |

**Why D is the actual recommendation:**

Options A, B, C are all single-h_in-per-K (with different ways to pick
which one). D acknowledges that the corpus has a small number of
*meaningful* h_in regimes (per Zipf, ~10^2–10^3 attractors in
Shakespeare-sized data), and represents each K's h_in as a soft
mixture over those.

Operationally:
- Persona table: ~10^3 vectors of dim d_model. Tiny (~256KB at d=64
  bf16). Re-clustered every N epochs (or after warmup, frozen).
- Per-K mixture weights: ~10^3 floats per trie node, or a sparse
  top-K-personas-per-trie-node table to keep size manageable.
- At fire time: `h_in(K) = Σ_p w_K[p] · persona[p]`. Cheap.
- After fire: K's actual h_cap is recorded; at epoch end, all
  recorded h_caps re-cluster the persona table (k-means or similar)
  and mixture weights get re-fit.

This sidesteps the compute issue (no 10^6-fold blowup) AND the
information-loss issue (preserves meaningful diversity) AND the
statistical efficiency issue (each persona is well-supported).

**Question to resolve:** how many personas? Probably want to start
small (~64 or ~256) and grow if PPL improves. What clustering
algorithm — k-means on h_cap vectors, or something structural
(group K's by suffix-tree neighborhood)?

### Q3: Training schedule — RESOLVED: Option A (single-pass, running-average h_caps)

Resolved 2026-05-27.

Single-pass per epoch:
1. At epoch start, load h_in table (persona-mixture weights per K +
   persona vectors) from previous epoch. Epoch 0 uses zero h_in
   (effectively the standard AGPT baseline).
2. Run AGPT fires normally; each fire's input includes h_in[K]
   (detached, treated as input constant).
3. During fires, record observed h_caps per K. Maintain a
   running-average per K (cheaper than storing all observations, and
   smoother than just taking the last value).
4. At epoch end, re-fit personas from the collected per-K average
   h_caps. Re-fit per-K mixture weights.
5. Next epoch starts with the updated table.

Running-average over final-value: smoother gradient signal across
epochs, and cheap (one EMA accumulator per K rather than full history).

Escalation paths if single-pass underfits:
- Option B (multi-pass per epoch): only if PPL shows signal but
  plateaus quickly.
- Option C (within-fire recurrence): unlikely; revisit only if A and B
  both fail.

Persona refit cadence: every epoch initially. If clustering proves
stable (personas don't drift much between epochs), drop to every K
epochs or freeze after warmup.

### Q4: Storage of h_caps between iterations

Now framed by Q2-D (persona table):

| Layer | Storage | Notes |
|---|---|---|
| Persona table | ~10^3 × d_model bf16 = ~256KB at d=64 | Refit at epoch boundaries (or frozen after warmup). |
| Per-K mixture weights | ~10^6 K × top-k-personas × (id + weight) | Sparse; top-8 personas per K ≈ 64MB. |
| Raw h_cap recordings (one epoch's worth) | ~10^6 × d_model bf16 = ~128MB at d=64 | Used to re-cluster personas at epoch end; can be discarded after. |

Total persistent state: ~70MB. Manageable.

If Q2 lands on A (averaging only) instead of D, storage is just the
single h_in vector per K: ~128MB at d=64 bf16.

### Q5: Inference semantics

At long-context generation, how does the recurrence carry forward?

Open sub-questions:
- When does the d-boundary fire during incremental token generation?
  Every d tokens of generated output? At each radix-cap of the implicit
  generated path?
- Does the model see all d previous tokens of generated text as
  standard attention K's, AND the h_in from the prior boundary? Or
  does the h_in replace the older K's (sliding-window style)?
- RoPE: does position counter continue past d at inference, or reset
  at the recurrence boundary (Gemini's "RoPE reset" suggestion)?

**Question to resolve:** this matters less for the training math but
critical for the seq-len-extension claim. Without a clean answer, the
RNN works architecturally but doesn't actually extend context.

### Q6: Evaluation

What's the empirical test that says "this works"?

Candidate probes:
- PPL on Shakespeare 1M at training seq_len (matches baseline = no regression).
- PPL at seq_len > d (the real test — does the recurrence carry info?).
- Ablation: zero h_in at eval time, compare to live-h_in PPL. Delta = "amount the model is using the recurrence."
- Cross-K aggregation noise check: PPL with average h_in vs PPL with predecessor-sampled h_in. Smaller is variance-tolerant.

**Preliminary plan:** PPL at seq_len ∈ {d, 2d, 4d} on Shakespeare and
Gutenberg, with and without h_in ablation, three seeds.

## What I'd build first (after Q1-Q5 resolved)

Sketch — concrete only after we close the open questions:

1. **Predecessor table builder**: walks suffix tree, emits
   `K → [(K', count)...]` per trie node. Offline tool, runs once per
   corpus. Builds atop existing suffix-radix.
2. **h_cap storage**: bf16 array, one per trie node. Initialized to
   zero. Loaded from disk at epoch start, saved at epoch end.
3. **Persona table** (if Q2-D): k-means cluster of all h_cap vectors
   into ~10^2-10^3 personas. Initialized to random unit vectors;
   refit at epoch end from collected h_caps.
4. **Per-K mixture weights** (if Q2-D): for each K, fit a soft
   mixture over personas based on observed h_in values for K's
   occurrences. Stored sparse (top-k personas per K).
5. **h_in lookup at fire time**: per K's fire, compute
   - Q2-A: `h_in = (Σ c_i · h_cap(K'_i)) / Σ c_i` (predecessor avg)
   - Q2-D: `h_in = Σ_p w_K[p] · persona[p]` (persona mixture)
   Vectorized over fire's K's.
6. **Injection at root**: per Q1 choice. If B: prepend h_in-derived
   K-token to KV sequence.
7. **Standard training**: no loss changes. h_cap updated at end of
   each fire (or epoch — per Q3).
8. **Evaluation hook**: ablation mode that forces h_in = 0 at
   inference.

## Once this works (preconditions for the full attention version)

If the RNN validates:
- Replace the mass-weighted aggregate with attention over the
  pre-pruned candidate set (Q=h_state, K=P_out of candidates,
  V=E_i).
- Add cycle-consistency loss term.
- Add RoPE-reset at recurrence boundary.

The full attention version is a strict incremental upgrade from a
working RNN. If the RNN fails, the attention version is unlikely to
rescue it — attention needs an informative recurrent state to attend
to. So validating the RNN first reduces risk.

## Implementation phases

Q1, Q2, Q3 resolved. Phased build order:

### Phase 0 — Survey (1 day)

Concrete questions to answer before cutting code:

- Where does the trainer currently compute hidden state at the deepest
  position of each prefix path? Is it materialized as a tensor, or
  consumed in-place by the loss kernel?
- What's the cleanest place in `agpt_train.cu`'s fire loop to capture
  these h_caps per trie node?
- What identifier do we use to index h_caps — trie node ID? prefix
  hash? Need a stable per-K key that's already in the trainer.

Output: a 1-page implementation map appended to this doc.

### Phase 1 — Infrastructure (no model changes, ~3-5 days)

1. **Predecessor table builder** (`src/tools/agpt_build_predecessor_table.cr`):
   walks suffix tree, emits `K → [(K'_predecessor, count)...]` per
   trie node. Offline. Result: a sidecar file like
   `data/input.predecessors.bin`.

2. **h_cap capture hook** in `agpt_train.cu`: at the appropriate
   point in forward pass, copy the deepest-position hidden state to
   a per-trie-node buffer. EMA-update the running average.

3. **h_cap serialization**: save/load running-average table to disk
   between epochs. Format: per-K-id × d_model bf16 array.

4. **Persona fitter** (`src/tools/agpt_fit_personas.cr` or python):
   k-means over the collected h_caps, ~256 personas initial. Output:
   `data/input.personas.bin` (P × d_model) and
   `data/input.k_persona_weights.bin` (per K, top-k personas with
   weights, sparse format).

At end of Phase 1: we have all the data structures, no training
behavior changes. Sanity check by training a baseline and verifying
the h_caps look reasonable (cluster into believable personas, deep
K's have low predecessor count, etc.).

### Phase 2 — Model integration (~3-5 days)

5. **W_k_inject, W_v_inject** parameter addition in model
   definition. Initialize to small random values.

6. **h_in lookup at fire time**: per K's fire, compute
   `h_in = Σ_p w_K[p] · persona[p]` using the top-k personas for K.
   Vectorize across the fire's K's.

7. **KV-cache prepend**: inject (K_inject, V_inject) at slot 0 of
   each layer's KV sequence. Attention machinery should already
   handle the extra slot if we're careful about position encoding
   (no RoPE rotation on the injected slot).

8. **Gradient flow**: ensure backward pass flows through
   W_k_inject/W_v_inject but NOT through h_in itself (stop-grad on
   the persona lookup).

### Phase 3 — Training and validation (~2-3 days)

9. **Epoch 0 — baseline parity**: zero h_in everywhere. PPL should
   match standard AGPT (within noise). Sanity check.

10. **Epochs 1+ — recurrence active**: h_in populated from previous
    epoch's persona table. PPL should improve (or at least not
    regress).

11. **Eval at seq_len > d**: PPL at seq_len ∈ {d, 2d, 4d} with and
    without h_in ablation (force h_in=0 at eval). The delta is the
    "recurrence is doing real work" signal.

12. **3-seed sweep** on Shakespeare 1M d=16, persistence to
    `rnd/cap-recurrence/`.

### Phase 4 — Iterate (open-ended)

Depending on Phase 3 results:
- If positive PPL signal but underfits: escalate Q3 to multi-pass.
- If persona table looks degenerate (all K's use same persona): tune
  persona count, clustering algorithm.
- If long-context PPL improvement is small: investigate inference
  semantics Q5 in detail (RoPE reset, sliding window).
- If completely flat: write up null and stop.

## What we'd cite for prior art (for write-up)

- Transformer-XL (Dai et al. 2019): segment-level recurrence with
  fixed memory.
- Compressive Transformers (Rae et al. 2019): hierarchical memory
  compression.
- Memorizing Transformers (Wu et al. 2022): kNN over external memory.
- Mixture-of-Experts routing (Shazeer et al. 2017, Fedus et al.
  2022): inspiration for the persona-mixture structure.

This combination — trie-structured weight sharing + persona-table
recurrence + suffix-tree-derived predecessor structure — appears
novel.
