# Harmonic-filter diagnostic results

Data: /tmp/shake_position_data
Corpus: data/input.txt
d=16  head_dim=16  base=10000.0
W (chord window): 64
n_pairs sampled: 20000 on-path, 20000 off-path

Frequencies per dim-pair:
  pair 0: ω=1.00000  period=6.3  cycles_in_W=10.19
  pair 1: ω=0.31623  period=19.9  cycles_in_W=3.22
  pair 2: ω=0.10000  period=62.8  cycles_in_W=1.02
  pair 3: ω=0.03162  period=198.7  cycles_in_W=0.32
  pair 4: ω=0.01000  period=628.3  cycles_in_W=0.10
  pair 5: ω=0.00316  period=1986.9  cycles_in_W=0.03
  pair 6: ω=0.00100  period=6283.2  cycles_in_W=0.01
  pair 7: ω=0.00032  period=19869.2  cycles_in_W=0.00

## E3 (no shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +0.1270  +1.0869  +2.6995  +4.2129  +2.6188
off-path +0.0512  +1.2609  +2.8011  +4.2543  +2.6975
gap (on - off): -0.1017    IQR avg: 3.0597    separation: -0.033
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -0.3993  -0.0021  +0.0095  +0.0385  +0.0230
off-path -0.2936  -0.0091  +0.0034  +0.0157  +0.0041
gap (on - off): +0.0061    IQR avg: 0.0327    separation: +0.187
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +0.1111  +0.8714  +2.2940  +3.5742  +2.2058
off-path +0.1067  +1.0832  +2.3757  +3.6139  +2.2905
gap (on - off): -0.0816    IQR avg: 2.6168    separation: -0.031
```

## E4 (with depth shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +0.1013  +1.0613  +2.6909  +4.2096  +2.6099
off-path +0.0512  +1.2531  +2.7956  +4.2546  +2.6971
gap (on - off): -0.1048    IQR avg: 3.0749    separation: -0.034
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -0.4287  -0.0063  +0.0054  +0.0244  +0.0114
off-path -0.3502  -0.0117  +0.0014  +0.0135  +0.0010
gap (on - off): +0.0040    IQR avg: 0.0280    separation: +0.142
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +0.1110  +0.8708  +2.2949  +3.5749  +2.2061
off-path +0.1062  +1.0829  +2.3745  +3.6142  +2.2908
gap (on - off): -0.0796    IQR avg: 2.6177    separation: -0.030
```

## E4-norm (unit chords + shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +1.6886  +4.3913  +5.4583  +6.3683  +5.3756
off-path +1.4793  +4.1075  +5.0235  +5.9094  +5.0129
gap (on - off): +0.4348    IQR avg: 1.8894    separation: +0.230
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -2.9975  -0.5295  +0.5391  +1.3705  +0.4408
off-path -2.9896  -0.7983  +0.1120  +0.9710  +0.0966
gap (on - off): +0.4271    IQR avg: 1.8346    separation: +0.233
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +3.8664  +3.9931  +3.9988  +3.9998  +3.9934
off-path +3.8054  +3.9907  +3.9978  +3.9997  +3.9914
gap (on - off): +0.0009    IQR avg: 0.0079    separation: +0.120
```
