Re: 28683f6c
Sent: 2026-11-15T14:53:49Z via local
Id: 7f635fd6

A correction to the "contrast reading" paragraph of my results mail. I said grouped runs lag random batches least at 1–2 characters and most from ~5 on, so the cost was deep contexts' step company. That was wrong. It compared runs at equal steps. A random conventional epoch (518 steps) trails mb84 in the same rising shape (+0.08 at 1 char, +0.41 from 5 on), so that profile only says a run is behind overall.

Compared at equal loss instead (`rnd/one-pass-gap/depth_matched.py`, agpt `87dd07f`): every one-pass checkpoint against random-batch runs interpolated to the same loss at contexts 5–16. AGPT pd=2/3/4 lag at one character by +0.26/+0.32/+0.63 nats, and match from 4 characters on. The pd=3 exact replay shows +0.33. Packed plans (+0.00 to +0.04) and random windows in pd=3's sizes (+0.01) show no root lag.

Cause: each step is the mean over its events, so a pd≥2 crown unit gives a 1-character context exactly one step per pass however common it is. The commonest, a space with 161,324 occurrences, gets 1 step at pd=3, where equal random batches give it 121 steps' worth (14 at pd=2, 559 at pd=4). The packed plans spread a heavy node's events in chunks over the pass, which restores frequency weight; their chunks are still single contexts. So the root lacked training weight, not variety within a step. It is small: about 0.03 of pd=3's 0.48 nats. pd=1 has no reference, since it is less trained than every random-batch run on record. README section, answer and handoff are corrected. Nothing is pushed yet.
