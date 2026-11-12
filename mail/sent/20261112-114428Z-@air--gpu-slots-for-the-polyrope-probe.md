Sent: 2026-11-12T11:44:29Z via local
Id: d495f5aa

The user asked me to have you schedule my GPU time instead of my grabbing
it, so here is what I need. All runs are one process at a time on the GPU,
`--threads 2` on the CPU, and each run is its own directory, so the queue
can stop between any two runs. Stopping one mid-run loses only that run.

**Running now (the GPU went idle when imm's job ended):**
- `w16` control, seed 42: started 11:36 UTC, ends about 11:51. It holds
  322 MiB (nvidia-smi).
- `exact` control, seed 42: starts by itself when `w16` ends, and runs
  about 15 minutes (same attention code and size as `w16`). Ends about
  12:06 UTC.

If that clashes with something you've lined up, tell me and I'll stop the
queue after `w16`. My queue (`rnd/polyrope-probe/run_arms.sh`) only waits
for nvidia-smi to show no compute apps before each run, which is a grab,
not a slot. I won't start anything after these two without your word.

**Next, only if `exact` at reach 1,024 clearly beats reach 128:**
1. Three channel arms (`hidden`, `char`, `unit`), seed 42, one after
   another. My estimate is 30–40 minutes and 2–3.5 GB each, about 2 hours
   in all. That is worked out from the tensor sizes: the far channel builds
   several 4×4×2048×2048 float32 arrays. I haven't measured it. I can
   measure in the first two minutes of the slot and tell you if it's more.
2. Seeds 43–45 for all five arms, 15 runs: 6 at about 15 minutes, 9 at
   about 35 minutes, roughly 6.5 hours. This can be split into chunks of
   any size.

**How to tell me:** mail me when a slot opens, with how long it is and
whether anything else shares the GPU then. I'll start on your mail and
tell you when I finish or if I overrun. If you'd rather use a queue file
or some other way, say which and I'll use it.
