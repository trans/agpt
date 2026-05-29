# Per-subtree training path (task #43)

Enables AGPT training at d=32+ by scoping the KV cache to one root-child subtree
at a time instead of the global corpus. Per `project_per_subtree_files.md`, the
builder + loader side is done; this doc scopes the trainer-side integration.

## Current gap

`bin/agpt_train` with format=2 (manifest.bin present) currently loads the
manifest, reports stats, and exits. Global-KV path hits the safety check at
d=32: "KV cache needs 27.7 GB but only 24.6 GB RAM+swap available."

## Why it isn't a 20-line wrapper

A naive "call `run_radix_training` once per subtree" wrapper runs into four
distinct problems, each of which is solvable but combined they make this a
~400-LOC refactor rather than a quick add:

1. **Virtual-root index mismatch.** Global `RadixTrieData` reserves
   `radix_id=0` as the virtual root (`radix_count` includes it; the scan loop
   at `run_radix_training` starts at `r=1` and uses `parents[r]==0` as the
   root-child test). `SubtreeData` has no virtual root: local_id 0 IS the real
   root-child, and its parent is encoded as -1. Calling `run_radix_training`
   on a subtree requires shifting every local index by +1 and synthesising a
   virtual-root entry, or refactoring the training body to accept either
   convention.

2. **Optimizer-state persistence across calls.** `run_radix_training`
   allocates `d_adam_m/v` and resets `adam_t=0` at entry, then frees nothing
   at exit (process-scoped). Calling it 65 times/epoch would reset RMSProp's
   running-average state per subtree — the first "step" of each call acts
   like raw SGD scaled by `1/sqrt(eps)` and blows up. Needs either
   (a) host-buffer roundtrip of `d_adam_m/v` and `adam_t` per call, or
   (b) hoisting those allocations into a persistent state struct.

3. **Per-call GPU alloc overhead.** Every invocation allocates ~30 chunk
   buffers (`d_x`, `d_q`, `d_k`, ..., all `sv_*` layer saves), cuBLAS handle,
   RoPE cache. Measured at ~1-2 sec per call. 65 subtrees × 50 epochs × 1.5s
   = ~80 min of pure alloc overhead for a single run. Acceptable for a proof
   but bad as the default.

4. **LR-schedule total-step estimate.** `compute_lr` needs `total_opt_steps`
   for cosine progress. Today run_radix_training estimates it from
   `n_root_children × subtree_splits × epochs`. With the per-subtree
   wrapper, each invocation sees `n_root_children=1` and its internal
   estimate is wrong. Needs a `total_opt_steps_override` parameter.

## Recommended shape

Refactor `run_radix_training` into three layers, then add the per-subtree
wrapper. Approximate LOC in parens.

```
struct PersistentRadixState { ... };                          // 40

PersistentRadixState alloc_persistent_state(cfg, wo, h_w);    // 100
   // weights, grads, adam_m/v, cublas, rope, clip scratch,
   // all chunk buffers + sv_* layer saves

void teardown_persistent_state(PersistentRadixState&, h_w);   //  40
   // copy d_weights → h_weights, free everything

int run_radix_training_on(
    PersistentRadixState& state,
    cfg, wo, RadixTrieData& trie,
    int epochs, [all existing knobs],
    int total_opt_steps_override);                            // 200
   // allocate KV cache sized to trie.total_edge_chars,
   // upload trie SoA buffers,
   // run the existing epoch/chunk/fwd/bwd loop (no changes),
   // free KV cache and trie GPU buffers,
   // leaves state.d_weights/adam_m/v updated,
   // state.adam_t monotonically counts across calls.

// Existing entry point: now a thin shell.
int run_radix_training(..., RadixTrieData& trie, epochs, ...) {
    auto st = alloc_persistent_state(cfg, wo, h_weights);
    run_radix_training_on(st, cfg, wo, trie, epochs, ...);
    teardown_persistent_state(st, h_weights);
    if (save_path) save_model_weights(...);
    return 0;
}

// New entry point for per-subtree.
int run_per_subtree_training(cfg, wo, h_weights, SubtreeManifest& m,
                              epochs, [knobs], save_path) {     // 150
    auto st = alloc_persistent_state(cfg, wo, h_weights);
    int total_opt_steps = m.n_subtrees * epochs * subtree_splits;
    for (int ep = 0; ep < epochs; ep++) {
      for (int i = 0; i < m.n_subtrees; i++) {
        SubtreeData s = load_subtree(m, i);
        RadixTrieData view = subtree_to_radix_view(s);    // shift indices
        run_radix_training_on(st, cfg, wo, view, 1, ..., total_opt_steps);
        free_subtree_view(view);
        free_subtree(s);
      }
      printf("Epoch %d: %d subtrees, ...\n", ep+1, m.n_subtrees);
      if (save_every > 0 && (ep+1) % save_every == 0) { ... }
    }
    teardown_persistent_state(st, h_weights);
    if (save_path) save_model_weights(...);
    return 0;
}

// New adapter.
RadixTrieData subtree_to_radix_view(SubtreeData& s);      //  80
   // malloc fresh parents/edge_*/counts_*/ancestor_* shifted by +1,
   // insert virtual-root at index 0 with edge_len=0 count=0.
```

Total: ~600 LOC, of which ~300 is code movement (existing alloc + body into
the split functions) and ~300 is genuinely new.

## Verification plan

1. **No-regression check at d=16.** After refactor, re-run the winning
   recipe through the existing `run_radix_training` path and confirm
   epoch-50 PPL is still 15.28 ± noise. This guards the working baseline.

2. **Parity check at d=16 per-subtree.** Run the per-subtree path against
   the same d=16 per-subtree file set (builder already emits these when
   `--agpt-per-subtree` is passed; `/tmp/agpt_input_d16_radix_per` would
   need to be built). Compare convergence to the global d=16 run. Expect
   same quality; differences are from optimizer-state accumulation order.

3. **d=32 smoke.** Run 5 epochs on `/tmp/agpt_input_d32_radix` rebuilt with
   `--agpt-per-subtree`. Peak KV per subtree should be ≈2.5 GB
   (vs 27.7 GB global). Should fit in 15 GB RAM headroom.

4. **d=32 full comparison.** 50 epochs, compare PPL to Window seq=32 (17.68)
   and the AGPT d=16 result (15.28).

## Prereq: d=32 per-subtree files

Need to rebuild d=32 with `--agpt-per-subtree`:

```
bin/microgpt --agpt --agpt-build-radix /tmp/agpt_input_d32 \
    --agpt-radix-out /tmp/agpt_input_d32_radix_per \
    --agpt-per-subtree --file data/input.txt
```

Peak per-subtree KV at d=32 Shakespeare, from manifest: largest subtree by
char count × 64 × 2 × 2 × 4 bytes. Early estimate puts this at 2-4 GB — well
under the RAM budget.

## Out of scope (follow-ups)

- fp16 KV cache: independent 2× reduction. Good combo with per-subtree for
  d=48+. Touches kernel arith, not the subtree plumbing.
- u8 on-disk tokens: saves ~3× on file size. Only matters if we start paging
  the subtree files themselves (d=64+ territory).
- Varint counts: negligible absolute savings; skip unless file size becomes
  a bottleneck.
