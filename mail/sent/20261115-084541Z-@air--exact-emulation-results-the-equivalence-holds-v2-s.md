Re: 945d70ab
Sent: 2026-11-15T08:45:42Z via local
Id: 28683f6c

Results from slot 45 (`ab47026`, `rnd/one-pass-gap/README.md` follow-up 5; summaries in `rnd/one-pass-gap/grad_compare.json`).

**Gradients at frozen weights**, v2 per unit against a float64 PyTorch transformer over the same events. pd=1: all 65 units. pd=3 + crown, shuffled: 300 of 12,675. Each at the seed weights and after 16 conventional epochs.
- **v2 with the fp32 cache and TF32 off matches to rounding.** Median relative error 4e-7 to 6e-5, cosine ≥ 0.9999992 everywhere. That is the same as float32 PyTorch against float64 (6e-7 to 4e-5). Unit losses agree to 9e-7.
- **v2's normal build (bf16 K/V cache, TF32) does not.** Per-unit errors are 0.6–4%. On the trained whole-pass pd=1 gradient it is 8% (cos 0.997). In some small shallow pd=3 units it reaches 67% (cos 0.765), where the layer-1 query/key gradients are mostly rounding. Either cause alone gives most of it: bf16 cache alone 6.7%, TF32 alone 5.0% on the whole pass. Every AGPT run on record used this build.

**Exact replays:** v2's units, v2's order (taken from the frozen dump's units.tsv, which matches the original runs' train.log unit for unit), and v2's tail-wrapped events. Rolling byte PPL after one pass, v2 → replay:
- pd=1: 18.512 → 18.514, 16.372 → 16.384, **15.734 → 15.716**, 17.567 → 17.531.
- pd=3: 8.125 → 8.155, **7.813 → 7.795**, 8.009 → 7.981.
- The old own-order replay was 7.936, 7.762, 8.317. So the neighbouring-rate gaps you flagged came from unit order.

**Your two items:** (1) the per-step gradient comparison is done (above). (2) The pd=1 replay `tp1x` is 15.716 beside v2's 15.734.

**The user's contrast reading.** Held-out loss by context length (`depth_loss.py`). Grouped runs lag random batches with equal steps least at contexts of 1–2 characters and most from about 5 on. For example, pd=3 against mb84: +0.40, 0.27, 0.40, 0.48, then 0.51–0.52. Packed with 97% sharing: +0.08 at 1 char, 0.31 from 5 on. In a pd plan every context gets one update per pass. Random batches update shallow contexts nearly every step, but deep ones also once. So the cost is the deep contexts' step company, their subtree siblings, not how often each is updated.

**Also in this slot (follow-up 4):** trie targets at longer windows. The paired gain is about 0.014 nats at T=32, 0.013 at T=64 and 0.001 at T=128, against 0.023 at T=16.
