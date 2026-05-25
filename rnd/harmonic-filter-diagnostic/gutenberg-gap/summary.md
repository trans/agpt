# Harmonic-filter diagnostic results

Data: /tmp/gut_position_data
Corpus: data/gutenberg_5m.txt
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
on-path  +0.1000  +1.3099  +3.0991  +4.3861  +2.8352
off-path +0.0512  +1.3440  +3.1489  +4.4034  +2.8698
gap (on - off): -0.0498    IQR avg: 3.0678    separation: -0.016
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -0.3992  -0.0012  +0.0070  +0.0298  +0.0293
off-path -0.4002  -0.0065  +0.0035  +0.0135  +0.0048
gap (on - off): +0.0035    IQR avg: 0.0254    separation: +0.138
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +0.1111  +1.1142  +2.6254  +3.7214  +2.3829
off-path +0.1057  +1.1320  +2.6620  +3.7414  +2.4364
gap (on - off): -0.0365    IQR avg: 2.6083    separation: -0.014
```

## E4 (with depth shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +0.1532  +1.2993  +3.0908  +4.3875  +2.8229
off-path +0.0521  +1.3407  +3.1415  +4.4028  +2.8684
gap (on - off): -0.0507    IQR avg: 3.0751    separation: -0.016
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -1.1758  -0.0056  +0.0036  +0.0196  +0.0143
off-path -0.4013  -0.0095  +0.0013  +0.0104  +0.0006
gap (on - off): +0.0023    IQR avg: 0.0225    separation: +0.103
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +0.1111  +1.1136  +2.6242  +3.7228  +2.3832
off-path +0.1052  +1.1322  +2.6614  +3.7415  +2.4367
gap (on - off): -0.0371    IQR avg: 2.6093    separation: -0.014
```

## E4-norm (unit chords + shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +1.2858  +4.3645  +5.4129  +6.3545  +5.3453
off-path +1.4382  +4.1126  +5.0310  +5.9256  +5.0205
gap (on - off): +0.3819    IQR avg: 1.9015    separation: +0.201
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -2.9951  -0.5570  +0.4772  +1.3705  +0.4056
off-path -2.9862  -0.8078  +0.1038  +0.9931  +0.0994
gap (on - off): +0.3734    IQR avg: 1.8642    separation: +0.200
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +3.8268  +3.9936  +3.9990  +3.9998  +3.9938
off-path +3.7861  +3.9914  +3.9982  +3.9997  +3.9919
gap (on - off): +0.0008    IQR avg: 0.0073    separation: +0.107
```
