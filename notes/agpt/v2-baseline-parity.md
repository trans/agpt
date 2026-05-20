# v2 Baseline Parity Check

This note captures the current Shakespeare `d16` baseline used to confirm that
`src/cudax/` still matches the legacy trainer contract.

## Training command

```sh
bin/agpt_train_v2 \
  --model data/input.random.model \
  --trie-dir /tmp/shake_baseline_d16_radix \
  --mode train-epoch \
  --epochs 10 \
  --lr 3e-3 \
  --lr-schedule warmup-cosine \
  --warmup-epochs 1 \
  --save /tmp/v2_typed_10.model
```

## Evaluation command

```sh
bin/perplexity \
  --model /tmp/v2_typed_10.model \
  --file data/input.txt \
  --seq-len 16 \
  --backend openblas \
  --max-positions 4096
```

## Expected result

Observed stable band on the current baseline:

- legacy trainer: roughly `8.07` to `8.54` PPL across repeated runs
- v2 trainer: roughly `8.35` PPL on repeat checks, within the same spread

If this baseline moves materially outside that band after a trainer change,
check the execution contract first:

- persistent compact K/V cache across chunks
- parent-before-child depth order inside each `pd=1` unit
- runtime-owned RoPE cache and cuBLAS handle
- unit-level optimizer firing, with chunks used only for memory slicing
