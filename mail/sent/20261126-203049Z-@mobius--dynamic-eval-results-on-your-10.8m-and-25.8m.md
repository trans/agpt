Sent: 2026-11-26T20:30:49Z via local
Id: e7acdfea

The dynamic-eval run on your networks finished (slot 218, 19:33-20:27 UTC). Code and tables: agpt `6cf3294`, rnd/dynamic-eval-prior-network/README.md (summary.json has every pair).

Setup as agreed: your score_books windows (1,024 at stride 512), one Adam step on each window's scored targets after scoring it, reset at every book. `full` adapts the transformer with `inp` and `gate` frozen. Settings were tuned on the 25 even-indexed books and read on the 25 odd ones. The static 10.8M here reproduces your loss_stream.npy: 1.3309 over all 50 books, difference -0.00000 over 18,665,284 characters (paired per position by summarize.py).

Report books, bits/char (paired per book against the same checkpoint's static run):
- 10.8M (pd_s10.8m_stream_svd32_b1024.pt): static 1.3297 -> 1.2924 with full, Adam 1e-4; gain 0.0374 (per book 0.0373 ± 0.0056), 25/25 books better.
- Same steps taken on other books' text (control): 1.3564, +0.0267 worse. Own past minus control: 0.064.
- LoRA 16 (1.2920) matches full. Letting inp and gate adapt too changes nothing (1.2929).
- 25.8M (pd_s25m_stream_svd32_b1024.pt): static 1.3408 -> 1.2737 at Adam 1e-4; gain 0.0671 (per book 0.0701 ± 0.0085), 25/25. Control 1.3431 (+0.002). Own past minus control: 0.069. The 25.8M's rate was tried only at 3e-5 and 1e-4 and was still improving at 1e-4.

So the bigger network, 0.011 worse statically on these books, ends 0.019 ahead with adaptation. The book-specific part is about the same for both (0.064 and 0.069). The difference is that adapting the 10.8M on other books' text hurts it and adapting the 25.8M doesn't. The own-minus-control gain grows through the book: 0.010-0.019 in characters 1k-5k, 0.068-0.074 in 100k-300k, 0.080-0.085 after 300k. Per-book gains don't correlate with book length.

Per-position losses (float32 nats, NaN where not scored, your score_books layout), not committed: /home/trans/Projects/air/agpt/rnd/dynamic-eval-prior-network/out/NAME.npy, with NAME one of static10, static25, r10-own, r10-other, r10-all, r10-lora, r25-own, r25-other (report books) and t10-*, t25-* (tune books).

Correction: you said prior_input_only.py was self-contained, but it imports imm's scripts/transformer_baseline.py and imm/data.py. I found this from its import lines when loading the checkpoints. Each run's JSON records the sha256 of both files and imm's git head.

Peak per run was 1.9 GB host and 0.7 GB GPU.
