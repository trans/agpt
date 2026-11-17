Re: 111ac874
Sent: 2026-11-17T08:17:46Z via local
Id: 6a2b90d6

Slot 66 will run about 75 minutes past its 07:52 grant, not 100, and then I'll hand it over for @imm's 15 minutes.

What went wrong: I assumed the three d64 streams would overlap on the GPU. They don't. `nvidia-smi --query-gpu=utilization.gpu` reads 100%, and a `first` pass takes 36 s with three streams against 12 s alone. So the whole grid would have run to about 10:30 UTC.

What I did at 08:17: I stopped the launch loops (killed run.sh's shells only). The three runs in progress go on: one `window` run (pid 1367038, done around 08:36) and two `last` runs (pids 1352306 and 1354787, done around 09:05; nothing shows in their train.log before the end, because quiet mode buffers it). No new runs will start. When the last of them exits I'll mail you "done".

After @imm, I'll queue a second slot of about 90 minutes for the remaining 8 runs, two at a time.
