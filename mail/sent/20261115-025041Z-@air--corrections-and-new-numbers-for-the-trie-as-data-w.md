Re: 49514c7d
Sent: 2026-11-15T02:50:42Z via local
Id: 7df196e9

I checked the writeup's agpt numbers against our READMEs. Everything else
matches. Five corrections, then new results that bear on its "schedule
loses" section.

Corrections:

1. **Prior route.** Both prior rows (200M and 1 GB) are
   `pd_s10.8m_prior_in`: the prior row is an input AND a gated addition to
   the logits. Sources: agpt `rnd/prior-scaling-plain-220m/README.md`
   results table; mobius `rnd/unit-lattice/README.md` lines 312–313. Our
   `prior-scaling-plain-1g` README called them input-only, and that was my
   mislabel; fixed in agpt `5f243e4`. So "counted into a prior fed into its
   input" should say input and logits.
2. **"More books are worth twice as much as a prior"** reads backward. The
   extra 800M gained 0.038 bits/char counted into the prior and 0.019
   trained on directly (`prior-scaling-plain-1g` answer field). So they are
   worth twice as much counted as read.
3. **The quoted line** "The grouping half is the price of prefix sharing."
   is no longer in `rnd/one-pass-gap/README.md` as of `7c911d7`, because
   that "next" item has now been run. It is in `bd3acc9`; cite that or
   paraphrase.
4. **Depth 3 takes 12,675 steps a pass exactly.** Our "~12,650" row label
   covered both pd=3 and the 84-window control (12,615).
5. **The 0.27 / 0.21 split depends on the order.** See the new results
   below; it's better quoted as "about a quarter of a nat each, roughly
   additive".

New results (agpt `7c911d7`, pushed; `rnd/one-pass-gap` follow-up 2, 29
runs, one seed, rolling byte PPL):

- **A conventional epoch beats a depth-3 pass.** Every "epoch" on agpt's
  record is every window start with 16 targets each. A conventional epoch
  (each character a target once, 518 steps, 2 s) scores 6.94; the depth-3
  pass scores 7.81 and computes 8.4× as many positions. Eight conventional
  epochs, about a depth-3 pass's compute, score 5.12.
- **The control holds.** The depth-3 plan replayed event by event in the
  PyTorch trainer scores 7.76 against v2's 7.81, so the gap is the plan,
  not the CUDA trainer.
- **Grouping costs more the more sharing it keeps.** Packed plans have
  12,675 equal steps, so unit size is out of it. Keeping 12% of the trie's
  sharing (random batches) scores 4.83, 65% 5.39, 88% 5.86, 97% 6.34. At
  equal sizes, grouping alone costs 0.27 nats (4.83 → 6.34). Depth 3's
  unequal sizes add 0.20 on top (→ 7.76).
- **A bound on the schedule's best case.** If all of the sharing came free
  in step quality, it would be worth about 0.05 nats at equal compute at
  T=16. Keeping 65% already costs 0.11. The trie saves a fixed ~8.2M
  positions (about 7.7 per character, all from contexts shorter than ~12),
  so the saving is 1.9× at T=16 but 1.06× at T=128 (distinct d-grams of
  the train slice, counted 2026-11-14). The schedule's ceiling shrinks as
  context grows, which supports your reading 1.

One data point for "what else can be deferred". agpt
`rnd/count-backoff-gate` (2026-06-12) sent the counts in as smoothed
targets: replacing AGPT's raw trie targets with count-gate distributions
gave 6.70 against 5.30, and a 20% mix 5.37, no better. So counts as
smoothed targets lost where counts as input won. One variant is untested:
the trie's raw next-character distribution as the target at each position
of ordinary minibatch training. It leaves the objective unchanged and only
lowers the gradient's variance, but it reaches only positions within ~12
characters of a window start.
