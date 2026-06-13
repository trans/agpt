# Count Backoff Gate

## Question

Can a count-only model learn when to trust a deeper context instead of backing
off, using only local prefix statistics?

This is a non-neural baseline for the proposed pointwise gate:

```text
r_d = m_d / (m_d + T_d)
g_d = KL(p_d || p_back(d))
h_d = H(p_d) / log(V)
l_d = d / D
w_d = sigmoid(theta . [r_d, g_d, h_d, l_d, 1])
```

Feature labels:

- `m_d`: count mass for the depth-`d` context.
- `T_d`: number of distinct next-token types seen after that context.
- `r_d`: Witten-Bell-style reliability; high when the context has substantial
  mass relative to its branching factor.
- `p_d`: empirical next-token distribution for the depth-`d` context.
- `p_back(d)`: backoff distribution for the suffix context at depth `d - 1`.
- `g_d`: distributional gain over backoff, measured as
  `KL(p_d || p_back(d))`.
- `h_d`: normalized entropy of the context distribution; lower means the
  context is more decisive.
- `V`: vocabulary size.
- `D`: configured maximum context depth.
- `l_d`: normalized depth, so the gate can learn depth-dependent trust.
- `theta`: the five learned shared gate parameters.
- `w_d`: learned trust weight for the depth-`d` context. `w_d = 1` means use
  the empirical context distribution; `w_d = 0` means use backoff.

The count tables are built from the first 95% of
`data/.splits/4fa9aec1db6b3aea/train_corpus.txt`. The final 5% of that train
split is used only to fit the 5-parameter gate. Final reporting uses
`data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt`.

## Implementation

Tool: `src/tools/agpt_count_gate.py`

The learned gate recursively mixes empirical context MLE with a backoff
distribution. Gate features are fixed statistics of each context; gradients only
update the five shared gate parameters. `g_d` uses a Witten-Bell backoff
distribution as the reference distribution.

`target_backoff_oracle` is intentionally reported only as a diagnostic. It is
not a valid language model because it backs off until the true target has been
seen.

## Results

Heldout fixed-skip PPL, natural-log perplexity:

| Max depth | Unigram | Witten-Bell | Learned gate | Target-backoff oracle |
|---:|---:|---:|---:|---:|
| 4 | 27.243 | 4.235 | 4.265 | 3.814 |
| 6 | 27.244 | 4.526 | 3.924 | 3.196 |
| 8 | 27.245 | 5.433 | 3.923 | 2.952 |
| 16 | 27.247 | 7.216 | 4.049 | 2.827 |

Entropy-delta extension:

```text
e_delta(d) = H(p_back(d)) / log(V) - H(p_d) / log(V)
```

Positive `e_delta(d)` means the child context is sharper than its backoff
context. Adding this as a sixth gate feature produced a small improvement:

| Max depth | Base learned gate | + entropy delta |
|---:|---:|---:|
| 6 | 3.924 | 3.921 |
| 8 | 3.923 | 3.923 |

Suffix-stat extension:

For the same context string, the suffix-side table counts what appears before
the string rather than after it. For context `abcdef`, prefix stats estimate
`P(next | abcdef)`, while suffix stats estimate `P(prev | abcdef)`. Suffix
backoff drops the newest token (`abcdef -> abcde`), mirroring a prefix trie over
the reversed corpus.

The `suffix_stats` feature group adds:

- `suffix_mass_norm`: log-normalized suffix-side count mass.
- `suffix_reliability`: `m / (m + T)` for the suffix-side distribution.
- `suffix_entropy_norm`: normalized entropy of preceding-token distribution.
- `suffix_kl_gain`: KL gain over suffix-side backoff.
- `suffix_entropy_delta`: suffix-side entropy drop relative to suffix backoff.

This helps materially:

| Max depth | Base gate | + entropy delta | + suffix stats | + both |
|---:|---:|---:|---:|---:|
| 6 | 3.924 | 3.921 | 3.892 | 3.887 |
| 8 | 3.923 | 3.923 | 3.861 | 3.860 |

Run files:

- `result-depth4-100ep.json`
- `result-depth6-100ep.json`
- `result-depth8-100ep.json`
- `result-100ep.json`
- `result-depth6-entropy-delta-100ep.json`
- `result-depth8-entropy-delta-100ep.json`
- `result-depth6-suffix-stats-100ep.json`
- `result-depth8-suffix-stats-100ep.json`
- `result-depth6-entropy-delta-suffix-stats-100ep.json`
- `result-depth8-entropy-delta-suffix-stats-100ep.json`
- `result-depth16-entropy-delta-suffix-stats-100ep.json`
- `result-depth16-entropy-delta-suffix-stats-100ep-sidecar-d16-top16.json`

## Neural Target Sidecar

The count-gate script can export a frozen `AGTS` v1 target sidecar keyed by
`substrings.bin` ids. The v2 CUDA trainer reads this through
`experimental.target_sidecar` and uses it as local soft target counts for each
radix endpoint. This is deliberately a target-smoothing experiment first: the
model does not receive suffix stats as input metadata.

For the depth-16 Shakespeare training trie, using entropy-delta plus suffix
features and retaining the top 16 tokens per substring produced:

- heldout fixed count-gate PPL: `3.900`
- substrings: `1,738,923`
- sidecar entries: `27,822,768`
- sidecar scale: `1,000,000`

The matching smoke artifacts were built in `/tmp`:

- `/tmp/agpt_d16_train_w16_position_data`
- `/tmp/agpt_d16_count_gate_suffix_top16.agts`

Trainer plan validation loaded the sidecar and matched all non-root prefix
radix nodes (`1,528,495 / 1,528,496`).

Neural target replacement and mixture checks:

| target interface | run | fixed-token PPL | rolling byte PPL | train wall |
|---|---|---:|---:|---:|
| raw trie targets | `rnd/lightning-agpt/20260612T024242-d64l2-depth16-lightning-stream-u20000-q10k-r6-cosfloor10` | 4.9776 | 5.3037 | 727s |
| 100% sidecar targets | `20260612T162513-d64l2-depth16-lightning-u20k-q10k-r6-sidecar-top16-rerun1` | 6.0046 | 6.6985 | 810s |
| 20% sidecar mixture | `20260612T165141-d64l2-depth16-lightning-u20k-q10k-r6-sidecar-mix020` | 4.9845 | 5.3667 | 865s |

The sidecar is valuable as a count/calibration diagnostic, but replacing neural
targets with it is too blunt. A 20% mixture mostly recovers the raw-target
result but still does not beat it. Keep this as a separate calibration /
target-shaping research thread, not as part of the core stochastic batching
claim.

## Findings

The learned pointwise gate is a strong count baseline. At depth 6-8 it reaches
about `3.92` heldout fixed PPL, below our recent attention AGPT numbers and well
below plain Witten-Bell in the same script.

Plain Witten-Bell gets worse as max depth increases here: `4.235` at depth 4,
`5.433` at depth 8, and `7.216` at depth 16. The learned gate corrects much of
that sparse-depth calibration failure, but depth 16 is still worse than depth
6-8.

The valid split and heldout split are not identical in difficulty. For example,
the depth-8 learned gate reports valid PPL `4.898` and heldout fixed PPL
`3.923`. So the absolute heldout number should be cross-checked against the KN
baseline using the same split convention, but the within-script comparison is
clean: the gate beats its Witten-Bell reference on the same counts and heldout.

The depth profile says the model is not blindly trusting long contexts. For
depth 8, average gate weight on heldout falls from about `0.66` at depth 1 to
about `0.17` at depth 8. For depth 16, it falls to about `0.05` at depth 16.

The feature table is open to additional observations. `entropy_delta` is the
first check: it is directionally positive but small, suggesting most of this
signal was already recoverable from absolute entropy, KL gain, reliability, and
depth.

Suffix-side statistics are a stronger addition. The best run so far is depth 8
with both prefix entropy delta and suffix stats at `3.860` heldout fixed PPL.
This suggests the reverse tree contains useful calibration information about
whether a context is an anchored phrase-like unit or a high-variance fragment,
even though the suffix distribution is not directly predicting the next token.

## Next Checks

- Compare this exact split and scoring convention against a proper
  interpolated or modified Kneser-Ney implementation.
- Try a gate fitted on multiple validation slices instead of one train tail, to
  reduce split-specific calibration.
- Use this count gate as a non-neural prior or calibration target for stochastic
  AGPT rather than adding entropy directly to the neural loss.
