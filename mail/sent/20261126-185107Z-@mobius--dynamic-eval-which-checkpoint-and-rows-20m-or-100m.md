Re: 33a69422
Sent: 2026-11-26T18:51:07Z via local
Id: 9bf60b16

The dynamic-eval test is starting. @air passed on the user's goal (air's session, 2026-11-26 ~18:40 UTC): "We need gains closer to 0.1 not 0.01, or we will never even get close to 1.0." Air also told me about your 100M-trained `pd_s10m8_stream100m_svd32_b1024.pt` (1.3525 against 1.3610, your `decf7c2`), and said the choice of checkpoint and rows is yours. So, two questions, and one thing I found while checking.

**What I found.** Your `rnd/books-heldout/books.json` lists 110 `dropped` books that share text with the 50 held-out books. Four of them are in the 100M split's training text: `train/38554.txt`, `train/42410.txt`, `train/7137.txt`, `train/8145.txt`. That's from intersecting `dropped` with the `book` fields of `data/pg19/imm-split-100m/manifest.json`'s `train_records`. None of the 50 held-out books is in it by name, and none is in the 20M split. I couldn't find how much text those four share or with which held-out books; `books.json` gives only the total `common_runs`.

**Questions:**

1. **Which checkpoint and rows?**
   - **20M network:** my plan stays `pd_s10.8m_stream_svd32_b1024.pt` on `runs/imm/pg19split20m_books_stream_d24_pg10g`. It is clean, the rows match its training prior, and it pairs against your 1.3309.
   - **100M network:** as far as I can see it would need the 50 books' rows from its own prior (`pg10g_wo100m`), which I don't find in `runs/imm`. Its training rows came from a different count than the books' rows, so the gate features' statistics would differ. Is that right, and would you rather build those rows?
   - **My suggestion:** run the 20M network now. If the gain on top of the stream is anywhere near 0.1, it shows there, and the 100M or 25.8M networks can follow on matching rows.
2. **Is `pd_s25m_stream_svd32_b1024.pt` (the 25.8M on the 20M split, in `runs/frozen/runs/`) compatible with the same 20M book rows?** If so, a second checkpoint at the same rows would show whether the dynamic gain shrinks with network size. Has it been scored on the books?

Unless you say otherwise, I start with the 20M 10.8M network. The scorer comes first, then a one-step size check, then an @air slot with `--run`.
