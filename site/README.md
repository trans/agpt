# AGPT project page

This is a self-contained static draft for the public AGPT showcase. It has no
build step, third-party assets, analytics, or runtime dependencies.

Preview it from the repository root:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000/site/`. The entire `site/` directory can be
published by a static host. Source links point to `github.com/trans/agpt` and
will resolve for visitors once the corresponding commits are pushed.

The page intentionally treats “one epoch and done” as the goal. The numerical
cards cite the [stochastic AGPT](../rnd/stochastic-agpt/README.md) and
[gradient population](../rnd/gradient-population/README.md) records. The paper
is labeled as a draft because its empirical section is under revision.
The advantage section cites the [per-depth branching
counts](../notes/trie-structure/shakespeare-h0-depth-profile.md), [radix node
counts](../rnd/sparsity-profile/README.md), and [depth-124
profile](../notes/seq-len-extension/d124-radix-feasibility.md). These describe
character-level structure in Tiny Shakespeare. Radix node records nearly
plateau after depth 16, while edge-character storage still grows sharply. The
50-token question for all recorded writing is an unmeasured working estimate,
not a storage bound or a measured end-to-end speedup.
