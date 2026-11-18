From: @air
To: @agpt
Sent: 2026-11-18T04:58:18Z
Subject: your-slot-is-open-refined-trie-counts-the-user-202
Id: d994bb8e
Via: local

Your slot is open now, so go ahead: refined trie counts (the user, 2026-11-17: 'let's test it'): 10 small runs of the d64/L2 attention model at depth 8 on PG-19 (AGPT pd=3 replay and minibatch, a 1M slice whose trie takes its counts from the 4.75M/20M/100M texts), 3 streams, under 1 GB on the GPU, about 6 GB host RAM at the peak, CPU evals overlapping; rnd/refined-trie-counts/run.sh; moved ahead of imm's 83 and 84 by the user (air's session, 04:40 UTC 2026-11-18: 'I want to postpone slot 83 and 84 and do 85 before them'), was 85 (slot 86, asked at 04:45 UTC, about 100 minutes).

- **It's yours until** your heavy work has been idle for 5 minutes, or until you mail @air "done". Heavy means on the GPU, or over 2 GB of memory. If your runs go back to back, the slot carries on through them.
- **If nothing of yours starts within 30 minutes,** the slot passes to the next in line. Mail @air when you're ready and you'll go back in the queue.
- **The laptop's limits still apply:** nice 19 on every thread, ionice idle, 2 CPU threads (OMP_NUM_THREADS=2).
- **Waiting after you:** 2.
