# Wrap-around corpus synthesis

Sample arbitrary-length token sequences from a depth-D radix trie by walking
root→leaf via mass-weighted child picks, then bridging back to the root via a
token sampled from the leaf's endpoint distribution and continuing. Trains a
window-mode SGD model on the synthesized corpus to test whether a depth-D
trie carries the predictive content needed for `seq_len > D` training.

Result: at d=32, the synthesized corpus is sufficient — held-out PPL on the
real corpus matches the SGD seq=128 ceiling. See `logs/synth-d32-10M-seq128-10k.ppl`.

## Artifacts (not in git — regenerate as needed)

| File                                    | Size  | Regen                                |
|-----------------------------------------|-------|--------------------------------------|
| `data/synth_wrap_d16.txt`               | 1.1 M | step 1 below (depth=16, 1M tokens)   |
| `data/synth_wrap_d32.txt`               | 1.1 M | step 1 below (depth=32, 1M tokens)   |
| `data/synth_wrap_d32_10M.txt`           |  10 M | step 1 below (depth=32, 10M tokens)  |
| `rnd/wrap-around/synth-d32-10M-seq128-10k.model` | 425 K | step 2 below              |

## Regeneration

Prereqs: a built radix trie at the desired depth. Build one with:
```sh
bin/agpt_build_index --corpus data/input.txt --max-depth 32
bin/agpt_build_radix --leveled /tmp/agpt_input_d32
# → /tmp/agpt_input_d32_radix/
```

### Step 1: synthesize the wrap-around corpus

```sh
# 1M tokens, d=16 (the smaller experiment)
bin/synth_wrap_corpus --trie-dir /tmp/agpt_input_d16_radix \
  --vocab-text data/input.txt --total-tokens 1000000 --seed 42 \
  --output data/synth_wrap_d16.txt

# 1M tokens, d=32 (the original prefix experiment)
bin/synth_wrap_corpus --trie-dir /tmp/agpt_input_d32_radix \
  --vocab-text data/input.txt --total-tokens 1000000 --seed 42 \
  --output data/synth_wrap_d32.txt

# 10M tokens, d=32 (the headline result — PPL 7.17 at seq=128/10k steps)
bin/synth_wrap_corpus --trie-dir /tmp/agpt_input_d32_radix \
  --vocab-text data/input.txt --total-tokens 10000000 --seed 42 \
  --output data/synth_wrap_d32_10M.txt
```

### Step 2: train the headline model (PPL 7.17)

```sh
cp data/input.random.model /tmp/synth_d32_10k.model
bin/microgpt data/synth_wrap_d32_10M.txt \
  --model /tmp/synth_d32_10k.model \
  --seq-len 128 --steps 10000 --lr 3e-4 \
  --d-model 64 --n-layers 2 --backend openblas --seed 42

# Score on real corpus:
bin/perplexity --model /tmp/synth_d32_10k.model --file data/input.txt \
  --max-positions 4096 --backend openblas
# → Perplexity: 7.1737
```

`bin/microgpt` and `bin/perplexity` come from the µGPT shard — build them
with `just build-microgpt-tools` if missing.
