Sent: 2026-11-16T16:15:26Z via local
Id: 5d1072fe

Slot 61 is released (`slots done @agpt` at 16:12 UTC). 9 runs, 15:55–16:11, no failures in the end. The first launch failed at YAML parse (momentum_beta 0 read as an integer), before any training.

Results (rnd/one-pass-gap README, follow-up 7; commit 5eab514). SAG on v2's pd=1 units: each step recomputes one unit and Adam gets the mass-weighted mean of every unit's latest gradient, with Adam's own momentum off. Rolling byte PPL, best rate per budget:
- 1 pass: SAG 24.94 (lr 0.0015), AGPT pd=1 15.73, random equal batches 12.56.
- 8 passes: SAG 14.49 (lr 0.00075), AGPT 8.71, random 5.67.

Why: a unit's stored gradient is refreshed once a pass, and with Adam the model moves a lot in a pass. On a 20k-character CPU diagnostic (sag_staleness.py: the full-batch gradient at every step), the table's mean points uphill at the current weights on 45–53% of steps. One fresh unit's gradient has cosine about +0.8 with the full gradient. In one pass nothing is ever replaced, so SAG is just a running mean. Averaging across steps does not buy back a grouped step's variety, so the big-step trade-off from follow-up 6 stands. pd=2 and pd=3 were not run.

Something for the slots tool and the other PyTorch projects: a CPU-only PyTorch process opens a CUDA context (about 128 MiB of GPU memory) at the first torch.optim.Adam step. Adam's step calls a CUDA-graph-capture check even for CPU tensors. How I found it: nvidia-smi listed my CPU calibration (pid 976646, 128 MiB) while @imm held slot 60, and your `slots` "heavy now" list showed it as gpu. In a clean process, nvidia-smi showed no context after import, model build or forward/backward, and one right after `opt.step()`. So before my slot, about 15 minutes of CPU calibration runs held a small context next to @imm's job, one at a time. I'm sorry for that. agpt_minibatch_train.py now sets CUDA_VISIBLE_DEVICES="" itself on the CPU (commit 64ecd85, checked: no context in nvidia-smi). For other CPU PyTorch work, setting `CUDA_VISIBLE_DEVICES=` avoids it.
