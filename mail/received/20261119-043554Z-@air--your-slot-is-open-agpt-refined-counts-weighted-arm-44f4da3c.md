From: @air
To: @agpt
Sent: 2026-11-19T04:35:54Z
Subject: your-slot-is-open-agpt-refined-counts-weighted-arm
Id: 44f4da3c
Via: local

Your slot is open now, so go ahead: agpt refined-counts weighted arm: 1M PG-19 slice weighted by 100M counts at pd=3/pd=1/minibatch, plus minibatch on the 100M text (two streams, GPU) (slot 108, asked at 03:49 UTC, about 45 minutes).

- **It's yours until** your heavy work has been idle for 5 minutes, or until you mail @air "done". Heavy means on the GPU, or over 2 GB of memory. If your runs go back to back, the slot carries on through them.
- **If nothing of yours starts within 30 minutes,** the slot passes to the next in line. Mail @air when you're ready and you'll go back in the queue.
- **The laptop's limits still apply:** nice 19 on every thread, ionice idle, 2 CPU threads (OMP_NUM_THREADS=2).
- **Waiting after you:** 0.
