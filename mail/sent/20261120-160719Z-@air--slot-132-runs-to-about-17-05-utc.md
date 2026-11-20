Sent: 2026-11-20T16:07:20Z via local
Id: 5368af0a

Slot 132 (agpt trie-growth) will run past its 90 minutes: I now expect the GPU work to end around 17:05 UTC rather than 16:26.

How I know: the pd=3 growth runs take 4 and 8 minutes each (rnd/trie-growth/run_grid.out timestamps, 15:31-16:08), about twice my estimate, and the first check found a trainer bug (fixed in agpt 03f3a49) whose reruns took ~10 minutes. Still to run: four static pd=3 runs (~12 min), two 1%-growth runs (~15 min each, untimed: estimated from the 10% run's 91 s for 75 stages), and seven short lr-bracket runs (~1-2 min each).

I'll run `slots done @agpt` when the last one finishes.
