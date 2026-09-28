# AGPT project page

This is a self-contained static draft for the public AGPT showcase. It has no
build step, third-party assets, analytics, or runtime dependencies.

Preview it from the repository root:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000/site/`. GitHub Pages publishes the contents of
`site/` at `https://trans.github.io/agpt/` through
[`pages.yml`](../.github/workflows/pages.yml) whenever the site changes on
`main`. Source links point to `github.com/trans/agpt`.

The social preview is `og-image.png`, generated from the editable
`og-image.svg` with `rsvg-convert -w 1200 -h 630 -o site/og-image.png site/og-image.svg`
from the repository root.

The page intentionally treats “one epoch and done” as the goal. The numerical
cards cite the [stochastic AGPT](../rnd/stochastic-agpt/README.md) and
[gradient population](../rnd/gradient-population/README.md) records. The paper
is labeled as a draft because its empirical section is under revision.
The advantage section cites the [per-depth branching
counts](../notes/trie-structure/shakespeare-h0-depth-profile.md), [radix node
counts](../rnd/sparsity-profile/README.md), and [depth-124
profile](../notes/seq-len-extension/d124-radix-feasibility.md). These describe
character-level structure in Tiny Shakespeare. Radix node records nearly
plateau after depth 16. The current exact representation still stores token
labels for unary edges; [tail-pruning](../rnd/unary-pruning/README.md) and
[root-wrap](../rnd/wrap-around/README.md) synthesis experiments point toward
ways to avoid retaining all long tails, but do not demonstrate memory savings
in the AGPT trainer. The 50-token question for all recorded writing is an
unmeasured working estimate, not a storage bound or a measured end-to-end
speedup.
