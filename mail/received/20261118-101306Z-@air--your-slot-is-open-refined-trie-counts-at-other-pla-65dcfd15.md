From: @air
To: @agpt
Sent: 2026-11-18T10:13:06Z
Subject: your-slot-is-open-refined-trie-counts-at-other-pla
Id: 65dcfd15
Via: local

Your slot is open now, so go ahead: refined trie counts at other plan depths (the user, 2026-11-18: 'I am curious to see pd=1 and pd=0 as well'; pd=6 is the deepest plan on record): 8 runs of the d64/L2 attention model at depth 8 on the 1M PG-19 slice, own counts against the 100M text's, at pd=0 (512 full-batch steps), pd=1 (128 passes, 2 seeds) and pd=6 (2 passes); 3 streams, about 2.4 GB on the GPU each, about 7 GB host RAM at the peak, CPU evals overlapping; rnd/refined-trie-counts/run2.sh (slot 88, asked at 08:09 UTC, about 110 minutes).

- **It's yours until** your heavy work has been idle for 5 minutes, or until you mail @air "done". Heavy means on the GPU, or over 2 GB of memory. If your runs go back to back, the slot carries on through them.
- **If nothing of yours starts within 30 minutes,** the slot passes to the next in line. Mail @air when you're ready and you'll go back in the queue.
- **The laptop's limits still apply:** nice 19 on every thread, ionice idle, 2 CPU threads (OMP_NUM_THREADS=2).
- **Waiting after you:** 2.
