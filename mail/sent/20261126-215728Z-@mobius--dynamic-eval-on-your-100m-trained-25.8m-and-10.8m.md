Sent: 2026-11-26T21:57:28Z via local
Id: 53459b78

Slot 220 finished at 21:55 UTC: dynamic eval of mobius's two 100M-trained networks. agpt `2e17076`, rnd/dynamic-eval-prior-network/README.md, section "The networks trained on 100M". It's approximate, on the 20M books' stream rows, with mobius's two caveats (~0.0003 and < 0.0001 bits). Peak host was 1.9 GB, one run at a time.

The answer to the user's question: no, the text gain and the adaptation gain don't add up to 0.1. Together they come to 0.071 bits/char.

Static on all 50 books:
- 25.8M on 100M (sha 44744c92…): 1.2883.
- 10.8M on 100M (sha 4e4d5595…): 1.3186.
- 20M 10.8M, for reference: 1.3309.

Report books, bits/char; full Adam 1e-4, chosen on the tune books (2e-4 was worse there: -0.030 against -0.050):

| network | static | own past | gain (per book ± s.e., books better) | other-book control | own - control |
|---|---|---|---|---|---|
| 10.8M, 20M | 1.3297 | 1.2924 | -0.0374 (-0.0373 ± 0.0056, 25/25) | +0.0267 | -0.0641 |
| 25.8M, 20M | 1.3408 | 1.2737 | -0.0671 (-0.0701 ± 0.0085, 25/25) | +0.0023 | -0.0694 |
| 10.8M, 100M | 1.3213 | 1.2922 | -0.0291 (-0.0275 ± 0.0042, 23/25) | +0.0342 | -0.0633 |
| 25.8M, 100M | 1.2903 | 1.2592 | -0.0311 (-0.0301 ± 0.0051, 23/25) | +0.0366 | -0.0677 |

- Against the 20M 10.8M's static 1.3297, the 100M 25.8M takes 0.039 statically and 0.031 more by adaptation, 0.071 in all. 1.2592 is the best figure on these books, but only 0.0145 below the adapted 20M 25.8M (1.2737).
- On the 10.8M, the 100M training's static gain is used up by adaptation: adapted, both 10.8Ms score 1.292.
- Own past against the control is nearly constant over the four networks (0.063-0.069; 25/25 books for each). The net gain falls short of it by what the control measures, the cost of steps taken on text that isn't the book: 0.027-0.037 on three networks, about 0 on the 20M 25.8M. That one is the network mobius found memorizes its training text.
- The own-past run need not pay the control's full cost. Even so, the obvious next test is a way to limit drift: decay toward the trained weights (Krause et al. 2018), or a rate below 1e-4 on the 100M networks. Neither is queued; that's the user's call.
- By position, the 100M 25.8M gains -0.002 in characters 1k-5k and -0.038 past 300k. Its own minus control is -0.083 past 300k, the same shape as on the 20M networks.

Per-position losses (float32 nats, NaN where not scored), not committed: /home/trans/Projects/air/agpt/rnd/dynamic-eval-prior-network/out/{static25h,static10h,t25h-full-adam-1e-4,t25h-full-adam-2e-4,r25h-own,r25h-other,r10h-own,r10h-other}.npy
