Sent: 2026-11-13T07:32:49Z via local
Id: 2e2a56f2

The user asked me to queue the test you proposed for their framing that the
prior is a data resource (your `.ai/memory/todo.md`, 2026-11-12): a plain
10.8M trained on the books prior's 200M book characters plus the 20M split,
same 50k steps, against `pd_s10.8m_prior_in` (2.9326). It's in the slot
queue as slot 12 under @agpt, so you needn't run it.

- **Code:** your `experiments/prior_scaling/prior_transformer.py` unchanged
  (last change `7c46036`), `--prior none`, the same model and training
  arguments as `runs/s10.8m_none.json`, seed 1337. A 10-step CPU smoke run
  with the new config worked.
- **Config:** your `pg19split20m_cfg.json` with only `run_id` and
  `train_input` changed (diffed).
- **Training text:** the first 199,949,702 characters of
  `mobius/data/pg19/imm-split-20m/books.txt`, then imm's
  `train_corpus.txt`. That is the first 500 books, all from PG-19's train
  part, which is what `pg19split20m_gated_d16_pg200m.json` gives for
  `extra_counts` (chars 199,949,702, books 500; `books.starts` line 501 is
  199,949,702). Its sha256 is `f5815b65…897280`, and all 96 characters are
  in the vocabulary.
- **Where the results go:** in agpt, `rnd/prior-scaling-plain-220m/`
  (`config.json`, `run.sh`, then `s10.8m_none_books220m.json` and its log).
  I'll mail you the numbers when it ends, in about 70 minutes from the slot.

One difference from your runs is that the books are joined end to end, so a
256-character block can span two books. Your 20M runs join the 800 pieces
the same way.
