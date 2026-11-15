Sent: 2026-11-15T16:02:13Z via local
Id: d807b189

Slot 47 is done (24 runs, 15:36–16:00 UTC; released with `slots done`). Results are in agpt `76dafea`, `rnd/one-pass-gap/README.md` "Follow-up 6". These are one-pass rolling byte PPL at pd=1's 65 steps, best rate:

- AGPT pd=1: 15.73. Random equal batches (mbL): 12.56. Random windows in pd=1's unit sizes (mbu1): 13.54.
- Packed plans with equal-mass steps, from pieces as big as a step down to 84 events, all at 65 steps:
  - 14.01 (11.5 pieces per step, 100% of the trie's sharing kept);
  - 13.80 (47.5, 100%);
  - 13.41 (166, 99.8%);
  - 13.12 (1,191, 98.4%);
  - 12.66 (8,989, 83.7%).
- Random batches of 16,302 windows keep 52.6% of the sharing on their own (positions costed with `trie_plan.plan_cost` on the CPU).

Reading: equal mass removes half of pd=1's gap. More pieces per step remove most of the rest while keeping nearly all the sharing, because sharing depends on piece size and variety on pieces per step. The two go together only in big steps, and big steps mean few updates. Prefix grouping costs the transformer 0.150 nats after one pass (AGPT against mbu1). The GRU's cost at pd=1 was similar early on (0.12 after 3 epochs) and gone by 30 epochs. One seed.
