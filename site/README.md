# AGPT project page

This is the public AGPT showcase. The published HTML, CSS and JavaScript have
no runtime dependencies or analytics. The experiment pages are generated from
the front matter and Markdown in `rnd/*/README.md`.

Preview it from the repository root:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000/site/`. GitHub Pages publishes the contents of
`site/` at `https://trans.github.io/agpt/` through
[`pages.yml`](../.github/workflows/pages.yml) whenever the site changes on
`main`. Source links point to `github.com/trans/agpt`.

To rebuild the experiment index and detail pages from the repository root:

```sh
python3 -m pip install -r site/requirements-build.txt
python3 src/tools/rnd_front_matter.py validate
python3 src/tools/build_experiment_pages.py
```

The generator checks front-matter schema, uses only nonignored README records,
and writes `site/experiments/index.html` plus one page per record. It omits
the local scratch directories `rnd/_smoke` and `rnd/pd6-canonical-eval`.
Headline caveat badges use their linked run's `result.json` or `meta.json`
date, falling back to the front-matter `updated` date. A separate, quieter
line summarizes older orchestrator runs across each directory.
The Markdown renderer is a build dependency only. Commit regenerated pages
along with any changed README summaries so GitHub Pages publishes them.

The social preview is `og-image.png`, generated from the editable
`og-image.svg` with `rsvg-convert -w 1200 -h 630 -o site/og-image.png site/og-image.svg`
from the repository root.

The page intentionally treats “one epoch and done” as the goal. Section 03
summarizes the structural context limit, update-cadence tradeoff and speed
question; detailed runs live in the experiment pages. The paper
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
