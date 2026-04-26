# Gutenberg 5M — wrap-around scaling test

Goal: validate that the wrap-around-via-synth-corpus approach scales beyond
Shakespeare to a 5M-character corpus.

## Setup

- Corpus: `data/gutenberg_5m.txt` — 5,000,000 chars, vocab 65 (same as
  Shakespeare; 5 Project Gutenberg books normalized)
- Trie: d=32 radix
- Synth: 10M wrap-around tokens, seed=42
- Train: 10k SGD steps at seq=128, lr=3e-4, d_model=64, n_layers=2, openblas
- Eval: PPL on `data/gutenberg_5m.txt`, 4096 positions, openblas

## Builder note

The original `bin/agpt_build_radix` (leveled→radix conversion) **OOMed** at
this scale on a 16 GB box, even with reduced cache settings. The leveled
trie itself builds fine (3.9 GB on disk in 2 minutes), but converting it
to radix exceeds memory because the per-depth children indices materialize
all 5M+ records of each cached depth.

Used the new `bin/agpt_build_radix_corpus` (corpus → radix per subtree)
instead. Bypasses the leveled intermediate; processes one root-character
subtree at a time. Memory bounded by ~1/65 of the trie. See
`src/agpt/corpus_radix_builder.cr`.

## Result

```
Build radix (per-subtree):  32.7 s   703 MB
Synth 10M wrap-around:       9.3 s    10 MB
Train 10k steps seq=128:   3:36 m    424 KB checkpoint
PPL eval (4096 positions): 28.4 s
                           -------
Total pipeline:            ~5 min
```

**PPL on `data/gutenberg_5m.txt`: 6.7807** (NLL 1.914 nats, 2.76 bpc)

For reference, Shakespeare 1M at d=32 with the same recipe got mean PPL
7.04 (across 4 seeds). Different test sets, so not directly comparable —
but the scaling shows wrap-around remains effective at 5× the corpus,
and the new builder makes it tractable.

## Depth sweep

| d  | Train loss | PPL    | radix_count | total_edge_chars | Build time |
|----|-----------:|-------:|------------:|-----------------:|-----------:|
| 32 |     1.9869 | **6.7807** |   7,539,820 |      113,838,759 |     32.7 s |
| 48 |     2.0085 |     7.3899 |   7,549,407 |      193,704,450 |     22.7 s |

Past d=32, the trie adds essentially **no new branching content**
(+9,587 radix nodes = +0.13%) — but **+70% edge chars** of pure mass-1
unary tails that the synth pipeline emits without proportional
structural signal. At a 10k-step training budget the model can't
absorb the extra synthetic material productively, and PPL on real text
actually rises by 0.61.

Consistent with the bayesian-bloom paper's *D\** concept (`§5`): past
the corpus's optimal branching depth, additional layers are mostly
noise. For Gutenberg 5M with this training recipe, **optimal d ≤ 32**.

## Compact char-trie (added 2026-04-26)

The d=48 build wouldn't have fit on a 16 GB box with the original
Hash-of-class char-trie node (~150 B/node) — it OOMed at the
high-frequency letter subtrees. Replacing the per-subtree char trie
with a struct-of-arrays representation (4 × Int32 per node = 16 B raw)
cut memory by ~10× and let d=48 build in 22.7 s. The same change made
d=32 about **2.4× faster** on Shakespeare (5.7 s → 2.4 s) thanks to
better cache behavior.

The compact-trie code is in `src/agpt/corpus_radix_builder.cr` (class
`CompactCharTrie`). It uses first-child + next-sibling linked lists
keyed by token, with O(branching) lookup at each level — fast in
practice because deep-trie branching is mostly 1.

## Generation sample (seed=42, temperature 0.8)

```
Illustration:

                    'CAlPTER I, Eno, and dey. Thal the beatfong have face det
knocter with he poly, so of this, and his of the cas them a bespe am or inter
and atlef we whild all the havor the her in and dust yould I
mist about and that as was dong nome a rew houndight of the who as ban, and for bemorg day py mad the sthing the mustem the gring thing the brealightle apearss the prould; and not the cally, and apon.

'Whable auddeng of belop dar sup mout the brestain, blet you's to lown thand -all
```

Recognizably 19th-century novel structure: chapter heading, narrative
paragraphs, single-quote dialogue. Words are invented but the morphology
and corpus shape are learned.

## Reproduce

```sh
just build-agpt-build-radix-corpus
just build-synth-wrap-corpus
just build-microgpt-tools

# Build radix from corpus (no leveled intermediate)
bin/agpt_build_radix_corpus --corpus data/gutenberg_5m.txt --max-depth 32 \
    --out /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus

# Synth + train + ppl
bin/synth_wrap_corpus --trie-dir /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus \
    --vocab-text data/gutenberg_5m.txt \
    --total-tokens 10000000 --seed 42 \
    --output /tmp/synth_g5m.txt

cp data/input.random.model /tmp/g5m.model
bin/microgpt /tmp/synth_g5m.txt --model /tmp/g5m.model \
    --seq-len 128 --steps 10000 --lr 3e-4 \
    --d-model 64 --n-layers 2 --backend openblas --seed 42

bin/perplexity --model /tmp/g5m.model --file data/gutenberg_5m.txt \
    --max-positions 4096 --backend openblas
```
