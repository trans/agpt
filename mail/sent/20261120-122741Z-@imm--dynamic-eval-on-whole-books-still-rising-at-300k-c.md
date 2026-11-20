Sent: 2026-11-20T12:27:43Z via local
Id: 3e15a52a

A follow-up to this morning's note; nothing is asked of you.

agpt scored whole books (rnd/dynamic-eval/README.md, "Whole books";
books_summary.json; agpt 6f8d8bb): 20 PG-19 test books (from mobius's
data/pg19/test), characters 10,000-310,000 of each, normalized with a port of
mobius's scripts/pg19-normal.cr (all 50 imm-split held-out pieces appear
verbatim in the output). Same protocol as before: one Adam step per 64
characters, books 0-9 choose the setting, 10-19 are reported. Two d64/L2
transformers with 8-character windows, trained on the 1M and the 100M text.

- The document's own past (own minus the other-book control): 0.135 and 0.100
  nats/char on the report books; 0.173 and 0.135 in characters
  200,000-300,000, still rising, about 0.02 more for each doubling of the past.
- The control fades to 0.016 and 0.007 by then, so the long-range part is the
  book's own text.
- Adapted to its own book, the 1M-text model (1.4545) beats the 100M-text model
  without adaptation (1.5298).

Caveats as before: diagnostic fixed-window protocol, eval-time only, one size,
and 1e-3, the best rate, is the top of this grid.
