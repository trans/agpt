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
The advantage section also cites the [branching-depth
profile](../rnd/trie-attention-framing/findings.md) and [radix node
counts](../rnd/sparsity-profile/README.md); these describe corpus structure,
not a measured end-to-end speedup or a sublinear bound on total trie storage.
