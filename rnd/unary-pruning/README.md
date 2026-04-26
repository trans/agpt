# Mass-1 Unary-Path Pruning

The d=32 Shakespeare radix trie has 1.1M leaves, **99.99% of them mass-1**
— each represents a 32-char prefix that appeared exactly once in the corpus
and continues with a single deterministic next char. Each mass-1 leaf
carries an edge of ~23 chars (median 24).

Open question: how much do those mass-1 unary chains actually contribute
to model quality? Each one is a one-shot pattern (not a reusable
distribution to learn), so plausibly the model can't really benefit from
training on most of their content. If true, we can prune them aggressively
without hurting either PPL or generation quality.

## Variations under test

Pruning is applied at synth time in `bin/synth_wrap_corpus`. When the
walk reaches a mass-1 leaf, instead of emitting the full ~23-char edge
and bridging, we emit a truncated prefix of the edge and wrap there:

| Flag                       | Cut policy                                           | Effective edge content kept |
|----------------------------|------------------------------------------------------|---------------------------|
| baseline (no flag)         | emit full edge + bridge token                        | full 23-char tail         |
| `--space-cut`              | back up to LAST whitespace; if none, fall through    | head .. last-whitespace   |
| `--prune-mass1-space`      | back up to FIRST whitespace; if none, fall through   | head .. first-whitespace  |
| `--prune-mass1-head`       | emit only the head token of the edge                 | head                      |

`--prune-mass1-head` is the most aggressive — it discards the entire
unary tail. `--prune-mass1-space` keeps everything up to the first
word boundary inside the edge. `--space-cut` keeps everything up to
the LAST word boundary (least aggressive of the three).

## Test protocol

Same as the wrap-around `synth-d32-10M-seq128-10k` baseline:

1. `bin/synth_wrap_corpus --trie-dir /tmp/agpt_input_d32_radix --vocab-text data/input.txt --total-tokens 10000000 --seed N <FLAG> --output /tmp/p.txt`
2. `bin/microgpt /tmp/p.txt --model /tmp/p.model --seq-len 128 --steps 10000 --lr 3e-4 --d-model 64 --n-layers 2 --backend openblas --seed 42`
3. `bin/perplexity --model /tmp/p.model --file data/input.txt --max-positions 4096 --backend openblas`
4. Sample generation from the model for visual quality check.

Reference numbers (from `rnd/wrap-around/`):

| config       | PPL mean | seeds       |
|--------------|---------:|-------------|
| baseline     |     7.04 | 42,44,46,48 |
| --space-cut  |     7.16 | 42,44,46    |

## Hypothesis

If mass-1 unary tails carry no useful training signal, --prune-mass1-head
should land within the noise band of baseline (~±0.10 PPL). If they DO
carry signal, head-pruning should hurt substantially (the model loses
most of its char-by-char training data).

--prune-mass1-space sits in between — keeps the head of each unary chain
but drops the long tail.

## Results

Multi-seed PPL on `data/input.txt` (4096 positions, openblas):

| Config                | Seeds        | Mean PPL | Range       |
|-----------------------|--------------|---------:|-------------|
| **--prune-mass1-space** | 42,44,46,48 | **6.95** | 6.77–7.17   |
| baseline              | 42,44,46,48  |     7.04 | 6.93–7.13   |
| --space-cut           | 42,44,46     |     7.16 | 7.00–7.37   |
| --prune-mass1-head    | 42, 44       |     7.53 | 7.50–7.56   |

Per-seed:

| seed | baseline | --space-cut | --prune-mass1-space | --prune-mass1-head |
|-----:|---------:|------------:|--------------------:|-------------------:|
|   42 |   7.1297 |      7.1205 |              6.7728 |             7.5582 |
|   44 |   6.9731 |      7.3691 |              6.8602 |             7.5029 |
|   46 |   7.1125 |      7.0047 |              6.9957 |                  — |
|   48 |   6.9345 |           — |              7.1694 |                  — |

### Findings

1. **--prune-mass1-head is confidently worse: +0.49 PPL vs baseline,
   ~5× noise floor.** Mass-1 unary chains *do* carry training signal
   — losing all but the head costs the model real ground.

2. **--prune-mass1-space is essentially baseline-equivalent: −0.09 mean,
   within natural variance.** The distributions overlap (baseline
   6.93–7.13, prune-mass1-space 6.77–7.17). The seed=42 −0.36 single-
   point gap was the lucky end of the prune distribution.

3. **The useful signal in a mass-1 unary chain is concentrated in
   roughly the first word past the head.** Keeping head + first-word
   (= first-whitespace cut) is statistically indistinguishable from
   keeping the full ~23-char tail. Keeping head alone loses 0.5 PPL.

4. **--space-cut (cuts at LAST whitespace, much later than
   --prune-mass1-space) is mildly worse on PPL.** The longer kept tail
   doesn't help; the very-late cut may even hurt slightly.

### Interpretation

Mass-1 paths look statistically irrelevant individually (one-shot
patterns, no distribution to learn from), but in aggregate they
provide useful char-by-char co-occurrence training data — at least
for the first word's worth of content. Past the first word, the
extra chars are essentially noise: discarding them doesn't hurt
(--prune-mass1-space ≈ baseline) and the model trained on fuller
content (baseline) doesn't gain from the extra context either.

Practical implication for the trie format: at d=32 with mass-1
dominance, a builder that **prunes mass-1 paths past the first
whitespace inside their unary chain** would shrink the trie
substantially without measurable PPL cost. Approximate savings:
average kept content per leaf drops from ~23 chars (full edge) to
~13.6 chars (head .. first whitespace), roughly 40% smaller.

### Generation quality (seed=42)

Sampling 500 chars at temperature 0.8 from the seed=42 model of
each variant:

  baseline:           "Bome One have the one acceds rice thou heartental..."
                      (JULIZAPENTHELA, plowferse — glued chunks of
                      disparate prefixes; pre-bug-fix baseline was
                      worse, fixed baseline still has some)
  --space-cut:        "...Sextlevemes a meave his, I sholds brithat..."
                      (word-shaped invented words)
  --prune-mass1-space: "...cartied here like I deat prace... mament
                      Edpeay shall a contele have lives, comeints."
                      (word-shaped, similar to --space-cut quality)
  --prune-mass1-head:  "...cantre me... ladalsen the se fall fen oin
                      welffor... tacanctingse wall ofith..."
                      (still word-shaped despite worse PPL — but more
                      fragmented, shorter coherent runs)

Generation quality differences are subtle past the bug-fix; all four
variants produce broadly Shakespeare-like word morphology.
