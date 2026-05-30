# Dense Causal Profile Diagnostic

Status: diagnostic, not an AGPT trie-depth metric.

Command shape:

```bash
python3 src/tools/agpt_ppl.py \
  --model <run>/checkpoint.model \
  --file <run>/heldout_corpus.txt \
  --vocab-file data/input.txt \
  --d 16 \
  --mode depth_profile \
  --device cpu \
  --batch-size 512
```

This evaluates the same held-out tail used by the current experiment harness:
`[16, 55755)` within `heldout_corpus.txt`, or 55,739 targets.

Important: this is a dense causal-LM diagnostic. It is not measuring AGPT
trie-depth probabilities. In the CUDAX compact-radix loss, non-endpoint
characters predict the next character inside a compact edge, while edge
endpoints predict the empirical branch distribution. A normal dense causal
window does not preserve that state semantics.

## Summary

The aggregate rolling PPL regression is caused by poor dense causal positions
near the start of each 16-token window. This explains why lm-eval rolling can
disagree with fixed/deep-only PPL, but it does not prove that shallow AGPT trie
nodes are bad.

| Run | position 1 | position 2 | position 3 | position 4 | position 8 | position 16 |
|---|---:|---:|---:|---:|---:|---:|
| `64x1` | 42.2151 | 14.8355 | 9.8066 | 7.8721 | 7.1792 | 7.2611 |
| `64x3` | 52.7152 | 30.4405 | 10.9533 | 8.0519 | 6.8955 | 6.8702 |
| `64x6` | 102.9235 | 27.1748 | 10.8716 | 7.7212 | 6.6992 | 6.7255 |
| `16x6` | 37.5784 | 18.0817 | 8.9655 | 7.2764 | 6.8420 | 6.8604 |
| `256x6` | 131.2872 | 27.4028 | 12.1160 | 7.5369 | 6.7136 | 6.7341 |

## Interpretation

These values should be interpreted as normal causal-LM position scores, not
tree-walk scores. A trie-aware evaluator is needed to compare AGPT against a
stochastic parrot that walks the same compact radix state machine.
