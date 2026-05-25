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
on-path  +3.5834  +4.6628  +4.6767  +4.7580  +4.8006
off-path +2.3551  +4.6384  +4.6695  +4.7008  +4.6695
gap (on - off): +0.0072    IQR avg: 0.0788    separation: +0.091
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -0.8628  +0.0026  +0.0091  +0.0599  +0.1221
off-path -2.0962  -0.0108  +0.0047  +0.0219  +0.0055
gap (on - off): +0.0044    IQR avg: 0.0450    separation: +0.098
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +3.8962  +3.9606  +3.9628  +3.9681  +3.9641
off-path +3.8387  +3.9599  +3.9627  +3.9670  +3.9625
gap (on - off): +0.0001    IQR avg: 0.0073    separation: +0.019
```

## E4 (with depth shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +2.6254  +4.6386  +4.6675  +4.7183  +4.7049
off-path +1.8431  +4.6173  +4.6635  +4.6928  +4.6484
gap (on - off): +0.0040    IQR avg: 0.0776    separation: +0.051
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -2.0080  -0.0065  +0.0041  +0.0278  +0.0430
off-path -2.7465  -0.0144  +0.0026  +0.0160  +0.0016
gap (on - off): +0.0016    IQR avg: 0.0324    separation: +0.048
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +3.8577  +3.9591  +3.9624  +3.9676  +3.9619
off-path +3.8144  +3.9582  +3.9623  +3.9665  +3.9602
gap (on - off): +0.0001    IQR avg: 0.0084    separation: +0.014
```

## E4-norm (unit chords + shift)

### dim-pairs: all
```
            min       p25     median   p75     mean
on-path  +1.5267  +4.4692  +5.4649  +6.3663  +5.4120
off-path +1.2867  +4.2121  +5.1655  +6.0385  +5.1362
gap (on - off): +0.2994    IQR avg: 1.8617    separation: +0.161
```

### dim-pairs: useful (pairs 0-2)
```
            min       p25     median   p75     mean
on-path  -2.9944  -0.4364  +0.5422  +1.4139  +0.4849
off-path -2.9995  -0.6836  +0.2666  +1.1069  +0.2294
gap (on - off): +0.2756    IQR avg: 1.8204    separation: +0.151
```

### dim-pairs: noise (pairs 4-7)
```
            min       p25     median   p75     mean
on-path  +3.8770  +3.9937  +3.9988  +3.9998  +3.9924
off-path +3.8158  +3.9911  +3.9983  +3.9997  +3.9902
gap (on - off): +0.0005    IQR avg: 0.0074    separation: +0.067
```
