"""Events per node of the 1M slice's depth-<=8 trie when its contexts are
counted in a larger text that contains the slice. Bytes are the symbols."""
import sys, numpy as np
a = np.frombuffer(open(sys.argv[1], 'rb').read(), dtype=np.uint8)
big_paths = sys.argv[2:]
D = 8
def keys(t, d, lo, hi):          # packed length-d substrings starting at lo..hi-1
    k = np.zeros(hi - lo, dtype=np.uint64)
    for j in range(d):
        k |= t[lo + j:hi + j].astype(np.uint64) << np.uint64(8 * j)
    return k
A = [np.unique(keys(a, d, 0, a.size - d + 1)) for d in range(1, D + 1)]
nodes = sum(x.size for x in A)
print(f"A: {a.size} chars, {nodes} nodes depth<=8 ({[x.size for x in A]}), sharing {8 * a.size / nodes:.1f}x")
for p in big_paths:
    t = np.frombuffer(open(p, 'rb').read(), dtype=np.uint8)
    cov = np.zeros(D, dtype=np.int64); tot = np.zeros(D, dtype=np.int64)
    C = 10_000_000
    for d in range(1, D + 1):
        n = t.size - d + 1
        for lo in range(0, n, C):
            hi = min(n, lo + C)
            k = keys(t, d, lo, hi)
            i = np.searchsorted(A[d - 1], k)
            i[i == A[d - 1].size] = 0
            cov[d - 1] += int((A[d - 1][i] == k).sum()); tot[d - 1] += hi - lo
    print(f"{p}: {t.size} chars; events in A's nodes {cov.sum()} of {tot.sum()} ({cov.sum() / tot.sum():.1%}); "
          f"sharing {cov.sum() / nodes:.0f}x; by depth covered {[f'{c / x:.0%}' for c, x in zip(cov, tot)]}", flush=True)
