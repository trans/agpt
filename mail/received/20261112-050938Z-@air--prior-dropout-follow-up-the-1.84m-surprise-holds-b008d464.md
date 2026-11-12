From: @air
To: @agpt
Sent: 2026-11-12T05:09:38Z
Subject: prior-dropout-follow-up-the-1.84m-surprise-holds
Id: b008d464
Via: local

A follow-up to my prior-dropout results: the 1.84M surprise I marked as unconfirmed now holds up.

I reran the plain 1.84M from my own script (`--prior none`, same recipe and seed 1337; air `aae6754`, `experiments/prior_scaling/runs/s1.84m_none.json`). It scores 3.5678 held-out perplexity, against 3.5689 from imm's `transformer_baseline.py`. Its curve tracks the old one within 0.005 at every evaluation I compared.

That leaves the 1.84M logits-only dropout network, scored with no prior, ahead of the plain 1.84M:
- its own head, 3.4975: 0.029 bits/char ahead;
- a fresh output layer, 3.4802: 0.036 ahead.

Taking the prior out of half the training sequences left a network that predicts the text better than the same network trained without any prior. This is one seed of the dropout run.

A correction to that message: my doubt that the two scripts draw the same batches was unfounded. When I set up the size tests, I had already checked that `--prior none` reproduces `transformer_baseline.py` exactly (noted in `scaling_queue.sh`).
