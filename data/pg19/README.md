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
- `imm-split-10g-dedup/`: 10,000,000,000 training characters, the 100M split's text
  and then its continuation: the 20M split's pieces, then 25,390 whole train books
  in the seeded order (the last cut at the mark), normalized as mobius's
  `pg19-normal.cr`. Built here, not by mobius, with
  `rnd/doped-counts-depth16/tools/build_pg19_big.py` (pg19-more.cr reads its output
  back into memory to hash it). Past the 100M split only books mobius counts for
  the 20M split (`imm-split-20m/books.names`, its overlap check against every
  piece) are used, and 54 more that share 64-character strings with a held-out
  piece are left out (`rnd/doped-counts-depth16/tools/excluded_books.txt`). Without
  that, the held-out pieces 20 and 23 occur whole in the text (other editions of
  their test books). The 100M split itself (pg19-more.cr does no overlap check)
  holds 5 books mobius's check leaves out; its held-out overlap is at most 0.2% of
  a piece's 64-character strings. Its first 100,000,000 characters are the 100M
  split byte for byte; `manifest.json` lists the appended and excluded books and
  the SHA-256 of the whole text (`95b084f2…`) and of its first 1e8 and 1e9
  characters. The held-out text and vocabulary are hard links to the 20M split's.
