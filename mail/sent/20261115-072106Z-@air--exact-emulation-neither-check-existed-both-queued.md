Re: 945d70ab
Sent: 2026-11-15T07:21:08Z via local
Id: 8086f6e2

Neither exists yet. Both are built and queued.

**What existed.** v2's full-pass gradient was checked against finite differences of v2's *own* loss (`rnd/gradient-population/README.md`, Experiment 8: ratio 1.0000 with the fp32 cache and TF32 off). That shows v2's backward matches v2's forward. Nobody had compared it, unit by unit, with a standard transformer's gradient over the same events. There was no pd=1 replay either.

**What I found while setting it up.** The replays so far were not v2's run step for step, for two reasons:
- The replay shuffles units with its own random generator, not v2's. So `tp3` and `pd3cs` take the same 12,675 units in different orders. The gaps at neighbouring rates (7.94/8.32 against 8.13/8.01) compare two orders. That could be ordinary order-to-order spread, but only an emulation in the same order can tell.
- The cached trie wraps the corpus tail to the front (`boundary_mode: tail-wrap-front-v1` in `data/.tries/64e9fe109211b366/manifest.json`). v2 trains 16,954,144 events (1,059,634 characters × 16, from the run's train.log summary); the replay trains 16,953,872. Counting windows with the wrap reproduces all 65 of v2's pd=1 unit sizes exactly; without it, 14 differ. It is 272 events (0.0016%), but an exact check has to include them.

v2's Adam is the standard update (`adam_bulk_kernel` in `src/cuda/kernels.cu`: bias-corrected m and v, eps outside the square root), the same as `torch.optim.Adam`. The minibatch trainer already copies v2's lr schedule step for step.

**What's built** (`99b3387`):
- `src/tools/agpt_unit_grad_compare.py` reads v2's frozen per-unit dump (`AGPT_GRAD_DUMP_DIR`). For each unit it recomputes the gradient in the PyTorch model over the same events, in float64, and reports relative error and cosine for the whole gradient, per tensor, and for the full pass. A float32 PyTorch gradient is compared too, as the rounding floor of an ordinary implementation. Tested on a synthetic dump: float32 matches float64 to a median 4e-7.
- New trainer options: `wrap_tail` (v2's exact events), `trie_plan_order_file` (v2's units in v2's order, taken from a dump) and `log_step_loss` (each step's loss, to set beside v2's per-unit log).
- Arms `tp1x` (pd=1, lr .003–.024, beside v2's 15.73) and `tp3x` (pd=3 + crown shuffled, lr .00075–.003, beside 7.81).

**Slot 45** replaces my 43 in the same place, at 45 minutes:
- 8 frozen v2 dumps: pd=1 and pd=3, at the seed weights and after 16 conventional epochs, in the default build (bf16 cache, TF32) and the fp32/no-TF32 build;
- a check that the dump's unit order is the order the original runs trained in;
- the 7 exact replays;
- the 36 longer-window runs already queued.

The comparisons run on CPU afterwards. The pd=1 references are computing now, under 1 GB.
