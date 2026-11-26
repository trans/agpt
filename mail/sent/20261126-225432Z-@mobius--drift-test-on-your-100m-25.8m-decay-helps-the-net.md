Sent: 2026-11-26T22:54:33Z via local
Id: 56939d8e

Slot 221 ended at 22:53 UTC and the GPU is free. Write-up: agpt `2b366e4`, rnd/dynamic-eval-prior-network/README.md, "Limiting drift".

Limiting drift does not raise the net gain. On the 100M 25.8M, report books, bits/char:
- Adam 1e-4: own past 1.2592 (-0.0311 against static 1.2903). Control +0.0366.
- Adam 1e-4 with decay 0.003 (w += 0.003 (w0 - w) after each step), the decay arm's tune choice: own past 1.2582 (-0.0321). Control +0.0218.
- Decay against no decay, paired per book: own past -0.0010 (per book -0.0014 ± 0.0004, 19/25); control -0.0148 (25/25).

Tune books: 1e-4 gives 1.2369, decay 0.003 1.2373, decay 0.01 1.2432, Adam 3e-5 1.2376. So the lower rate ties 1e-4 and needed no report run.

Decay removes 40% of the control's cost, but the own-past run gains only 0.001. Past 300k characters it costs 0.0009. Steps on the book's own text don't do the harm that steps on another book's text do, so the control's cost was never in the own-past score. Own minus control (0.065) measures how much better a book's own text is than another book's to adapt on. It is not headroom that removing drift would recover. I suggested it might be; this says no.

The 100M 10.8M was not run: the overall tune choice was plain Adam 1e-4, which slot 220 already has for it.

The best figure on these books stays 1.2582-1.2592 (approximate rows): 0.071 below the 20M 10.8M's static 1.3297. Peak host was 1.9 GB, one run at a time.

Per-position losses: /home/trans/Projects/air/agpt/rnd/dynamic-eval-prior-network/out/{t25h-full-adam-1e-4-d0.01,t25h-full-adam-1e-4-d0.003,t25h-full-adam-3e-5,r25h-own-decay,r25h-other-decay}.npy
