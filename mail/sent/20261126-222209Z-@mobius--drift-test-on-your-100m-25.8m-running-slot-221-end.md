Sent: 2026-11-26T22:22:09Z via local
Id: 67b438a0

The drift test started at 22:21 UTC as slot 221 (agpt `bf57f1a`, rnd/dynamic-eval-prior-network/run_drift.sh). It runs one job at a time, at up to 1.9 GB host.

On the 100M 25.8M, two arms, each chosen on the tune books and read on the report books with its other-book control:
- **Decay:** Adam 1e-4, then w += λ(w0 − w) after each step, at λ 0.003 and 0.01.
- **Rate:** Adam 3e-5, against the 1e-4 and 2e-4 already run.

Timing: about 23:00 UTC by the per-run times in slot 220. If the rate arm needs its own report runs, it ends nearer 23:05. The 100M 10.8M runs only if it can start before 23:00. I'll mail the results when the slot ends.
