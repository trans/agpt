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
on-path  +3.7646  +4.6607  +4.6799  +4.7789  +4.8183
off-path +1.8431  +4.6305  +4.6690  +4.7085  +4.6727
gap (on - off): +0.0109    IQR avg: 0.0981    separation: +0.111
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -0.7163  +0.0020  +0.0118  +0.0732  +0.1384
off-path -2.7465  -0.0167  +0.0047  +0.0268  +0.0084
gap (on - off): +0.0072    IQR avg: 0.0574    separation: +0.125
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +3.8799  +3.9602  +3.9631  +3.9693  +3.9643
off-path +3.8358  +3.9592  +3.9628  +3.9679  +3.9625
gap (on - off): +0.0003    IQR avg: 0.0089    separation: +0.035
```

## E4 (with depth shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +2.7219  +4.6338  +4.6691  +4.7250  +4.6968
off-path +1.8748  +4.6103  +4.6647  +4.7026  +4.6484
gap (on - off): +0.0044    IQR avg: 0.0917    separation: +0.048
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -1.9963  -0.0100  +0.0047  +0.0326  +0.0366
off-path -2.6869  -0.0194  +0.0030  +0.0229  +0.0033
gap (on - off): +0.0017    IQR avg: 0.0424    separation: +0.039
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +3.8461  +3.9588  +3.9627  +3.9691  +3.9617
off-path +3.7953  +3.9577  +3.9624  +3.9677  +3.9599
gap (on - off): +0.0002    IQR avg: 0.0102    separation: +0.023
```

## E4-norm (unit chords + shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +1.2951  +4.3963  +5.3933  +6.3417  +5.3614
off-path +1.5248  +4.1841  +5.1178  +6.0011  +5.1088
gap (on - off): +0.2755    IQR avg: 1.8812    separation: +0.146
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -2.9919  -0.5000  +0.4853  +1.4043  +0.4461
off-path -3.0000  -0.6974  +0.2353  +1.0678  +0.2126
gap (on - off): +0.2500    IQR avg: 1.8347    separation: +0.136
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +3.8659  +3.9920  +3.9987  +3.9998  +3.9911
off-path +3.7958  +3.9897  +3.9981  +3.9997  +3.9891
gap (on - off): +0.0005    IQR avg: 0.0089    separation: +0.059
```
