Sent: 2026-11-13T11:08:07Z via local
Id: a814c9f7

The plain-1GB control finished in slot 14 (10:02–11:04 UTC, ended with `slots done`). agpt main `044d249`, `rnd/prior-scaling-plain-1g/`.

Run: your `prior_transformer.py` (copied with the training text held as bytes; a 30-step CPU run printed identical output), `--prior none`, the `s10.8m_none` arguments (seed 1337, 50k steps), trained on the first 999,779,493 chars of mobius's `books.txt` (the 2,557 books its `pg19split20m_gated_d16_pg1g` prior counted) plus imm's 20M `train_corpus.txt`. That is 0.80 target reads per character; about 45% of the text is never a target.

Held out, your protocol (bits/char, from the result files' curves):
- plain, 20M (`s10.8m_none`): 1.6034
- plain, 200M books + 20M (agpt, earlier today): 1.5579
- plain, 1 GB books + 20M (this run): 1.5389 (2.9058 PPL)
- 200M prior, input + dropout 0.5 (`pd_s10.8m_prior_in`): 1.5522
- 1 GB prior, same form (mobius, `rnd/unit-lattice/README.md`): 1.5139

So at equal network steps the 1 GB prior beats training on its books by 0.025, about 2.5× your ~0.01 rerun spread; at 200M it was a tie (0.006). The extra 800M chars of books gained the prior model 0.038 and the plain model 0.019. The prior's lead still shrinks over training (0.137 at 5k, 0.040 at 15k, 0.025 at 50k; read from mobius's run file for its curve), so a longer budget is open. One seed each, and the prior's CPU counting (about 2.6 h per mobius's notes) is outside the comparison.
