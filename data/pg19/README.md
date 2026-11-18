# PG-19 splits (from mobius)

Hard links to mobius's PG-19 splits (`data/pg19/` in the mobius project,
built by its `scripts/pg19-split.cr` and `scripts/pg19-more.cr`): text
normalized to 96 symbols (printable ASCII and `\n`), training pieces of
25,000 characters from separate train books, and 50 held-out pieces of 5,000
from test books, the same in every split. Each split's `manifest.json`
records the SHA-256 of its training and held-out text. Not committed; the
runs record each file's SHA-256 in `meta.json` and their configs.

- `imm-split/`: 4,750,000 training characters (190 pieces).
- `imm-split-20m/`: 20,000,000 (800 pieces; starts with `imm-split`'s).
- `imm-split-100m/`: 100,000,000 (the 20M's pieces, then 205 whole books).
- `a1m/train_corpus.txt`: the first 1,000,000 characters of `imm-split`'s
  training text (40 pieces), SHA-256
  `bf7c1e86a2205b6e14bf59b54d94aae3f2a0ebdef994a9e8319783431bbe3340`.
