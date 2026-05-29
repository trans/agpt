# Shakespeare H=0 Depth Profile

Date: 2026-05-28

This note records a quick radix-trie profile for the Shakespeare corpus. The goal was to quantify how quickly deeper trie context turns into deterministic/unary structure.

Source trie:

```text
/tmp/shake_d16_radix
version=2
radix_count=1,607,928
depth_files=17
total_edge_chars=9,313,896
```

Definitions:

- `H=0` means the node has exactly one observed continuation.
- `endpoint_H0` means a stored radix endpoint has one continuation.
- `absorbed_H0` means an original trie node hidden inside a radix-compressed edge. These are unary by construction.
- `total_nodes` is the uncompressed trie-node count represented at that depth.
- `mass=1` means the deterministic node was observed once.

## H=0 By Depth

| depth | endpoint_nodes | endpoint_H0 | absorbed_H0 | total_H0 | total_nodes | pct_H0 | endpoint_H0_mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62 | 0 | 3 | 3 | 65 | 4.62% | 0 |
| 2 | 1,139 | 0 | 264 | 264 | 1,403 | 18.82% | 0 |
| 3 | 7,624 | 0 | 3,932 | 3,932 | 11,556 | 34.03% | 0 |
| 4 | 25,187 | 0 | 25,525 | 25,525 | 50,712 | 50.33% | 0 |
| 5 | 51,508 | 0 | 89,513 | 89,513 | 141,021 | 63.47% | 0 |
| 6 | 71,524 | 0 | 211,787 | 211,787 | 283,311 | 74.75% | 0 |
| 7 | 78,836 | 0 | 368,513 | 368,513 | 447,349 | 82.38% | 0 |
| 8 | 75,929 | 0 | 533,726 | 533,726 | 609,655 | 87.55% | 0 |
| 9 | 64,226 | 0 | 686,235 | 686,235 | 750,461 | 91.44% | 0 |
| 10 | 49,624 | 0 | 809,292 | 809,292 | 858,916 | 94.22% | 0 |
| 11 | 36,324 | 0 | 900,926 | 900,926 | 937,250 | 96.12% | 0 |
| 12 | 25,759 | 0 | 965,630 | 965,630 | 991,389 | 97.40% | 0 |
| 13 | 17,832 | 0 | 1,010,023 | 1,010,023 | 1,027,855 | 98.27% | 0 |
| 14 | 12,554 | 0 | 1,039,914 | 1,039,914 | 1,052,468 | 98.81% | 0 |
| 15 | 8,738 | 0 | 1,060,686 | 1,060,686 | 1,069,424 | 99.18% | 0 |
| 16 | 1,081,061 | 1,074,942 | 0 | 1,074,942 | 1,081,061 | 99.43% | 1,096,911 |

Total:

```text
H=0 nodes:        8,780,911
total trie nodes: 9,313,896
pct H=0:          94.28%
```

## H=0 Mass Buckets

| depth | nodes | H=0 | pct_H0 | H0_mass1 | pct_H0_mass1 | H0_mass2_5 | H0_mass6_20 | H0_mass21p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65 | 3 | 4.62% | 1 | 33.33% | 0 | 0 | 2 |
| 2 | 1,403 | 264 | 18.82% | 77 | 29.17% | 53 | 56 | 78 |
| 3 | 11,556 | 3,932 | 34.03% | 1,782 | 45.32% | 1,029 | 566 | 555 |
| 4 | 50,712 | 25,525 | 50.33% | 15,291 | 59.91% | 6,561 | 2,363 | 1,310 |
| 5 | 141,021 | 89,513 | 63.47% | 62,302 | 69.60% | 20,327 | 5,089 | 1,795 |
| 6 | 283,311 | 211,787 | 74.75% | 162,282 | 76.63% | 40,588 | 7,202 | 1,715 |
| 7 | 447,349 | 368,513 | 82.38% | 303,728 | 82.42% | 56,133 | 7,331 | 1,321 |
| 8 | 609,655 | 533,726 | 87.55% | 463,946 | 86.93% | 62,952 | 5,895 | 933 |
| 9 | 750,461 | 686,235 | 91.44% | 620,152 | 90.37% | 61,186 | 4,239 | 658 |
| 10 | 858,916 | 809,292 | 94.22% | 752,582 | 92.99% | 53,451 | 2,801 | 458 |
| 11 | 937,250 | 900,926 | 96.12% | 855,086 | 94.91% | 43,696 | 1,828 | 316 |
| 12 | 991,389 | 965,630 | 97.40% | 929,912 | 96.30% | 34,266 | 1,220 | 232 |
| 13 | 1,027,855 | 1,010,023 | 98.27% | 982,598 | 97.28% | 26,415 | 845 | 165 |
| 14 | 1,052,468 | 1,039,914 | 98.81% | 1,019,043 | 97.99% | 20,164 | 580 | 127 |
| 15 | 1,069,424 | 1,060,686 | 99.18% | 1,044,633 | 98.49% | 15,531 | 427 | 95 |
| 16 | 1,081,061 | 1,074,942 | 99.43% | 1,062,378 | 98.83% | 12,167 | 327 | 70 |

Total:

```text
total nodes:       9,313,896
H=0 nodes:         8,780,911  (94.28%)
H=0 mass=1 nodes:  8,275,793  (94.25% of H=0)
H=0 mass 2..5:       454,519
H=0 mass 6..20:       40,769
H=0 mass 21+:          9,830
```

## Interpretation

By depth 4, more than half of represented trie nodes are already deterministic. By depth 8, almost 88% are deterministic. By depth 10, over 94% are deterministic.

Most deterministic nodes are also low-support singletons. At depth 4, about 60% of H=0 nodes are mass=1. By depth 8 this is about 87%, and by depth 16 it is about 99%.

This means increasing trie depth mostly adds singleton memorization surface rather than well-estimated branching structure. Any sequence-length or multi-position extension should account for this support collapse; blindly increasing depth is unlikely to provide proportionate new statistical signal.
