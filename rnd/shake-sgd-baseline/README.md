# shake-sgd-baseline

Status: initial baseline landed

## Question

Record a standard Crystal microgpt sliding-window baseline for the Shakespeare
small-model setup, using the same AGPT experiment harness split and evaluator.

## Protocol

- Trainer: `/home/trans/Projects/microgpt/bin/microgpt`
- Mode: standard sliding-window SGD
- Corpus: `data/input.txt`
- Training split: prefix 95%
- Evaluation split: held-out tail 5%
- Model init: `data/input.model`
- Window: `seq_len=16`
- Model: `d_model=64`, `n_layers=2`, `d_ff=256`
- Optimizer: SGD, `lr=0.0003`, constant schedule
- Budget: 10,000 steps

## Caveats

This run predates the stricter raw-log policy and did not originally write a
`result.json`; the committed `result.json` was reconstructed from `eval_raw.json`
and `meta.json`.

## Results

| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|--------|----------------:|-----------------:|----------:|
| `20260528T181839-d64l2-s16-10k` | 10.0631 | 10.2344 | 3.3553 |
