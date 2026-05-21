# Archived sub-agent worktree work

Two abandoned sub-agent worktrees were dropped on 2026-05-20. Their
uncommitted/untracked content was salvaged here. The branches and
worktrees themselves are gone.

## a995434a-trie-columnar-refactor.patch

**Source worktree:** `microgpt/.claude/worktrees/agent-a995434a/` (microgpt-side, now removed)
**Branch (gone):** `worktree-agent-a995434a` (in microgpt's .git, base commit `a34ced0` from 2026-04-11)
**Base commit message:** "Add BFS trie-walk AGPT trainer with incremental backward"

**What the patch contains.** A real refactor of AGPT's trie internals:
turns `TrieNode` from a data-owning class into a thin value-type
façade (`struct TrieNode`) over columnar storage in `TrieCorpus`. The
node is now just `(corpus, id)` and all getters forward into parallel
arrays held by the corpus. ~400 lines across `src/agpt/trie_node.cr`
and `src/agpt/trie_corpus.cr` (current line counts are below).

**Caveat: applies to PRE-SPLIT layout.** The patch is rooted at paths
`src/agpt/trie_*.cr` which, at the time, lived inside the microgpt
repo. After commit `bdd8e82` ("Remove AGPT from µGPT: split into
separate project"), AGPT moved to its own repo (`/home/trans/Projects/agpt`)
and `src/agpt/trie_node.cr` + `src/agpt/trie_corpus.cr` now live there.
Whether this refactor has been applied (or partially applied) to the
post-split agpt-repo versions of these files has not been checked.

**To revisit.** Compare the patch's `+` blocks against the current
`/home/trans/Projects/agpt/src/agpt/trie_node.cr` and `trie_corpus.cr`.
If the columnar-storage refactor hasn't been done, the patch is a
reasonable starting point — paths align since AGPT kept its `src/agpt/`
layout post-split. If it has been done independently, the patch is
just historical.

```
$ wc -l a995434a-trie-columnar-refactor.patch
574
```

## ab1d1cb4-files/

**Source worktree:** `microgpt/.claude/worktrees/agent-ab1d1cb4/` (microgpt-side, now removed)
**Branch (gone):** `worktree-agent-ab1d1cb4` (in microgpt's .git, base commit `7518aeb` from 2026-03-23)
**Base commit message:** "Port connection rules: single-port replace, multi-port list collect"

The base commit is UI/construction-kit work (port connection rules for
the Svelte frontend, see `project_construction_kit.md` in user memory).
The worktree had no modifications to tracked files but two untracked
files preserved here:

- `convergence_analysis.py` — Python convergence analysis script, ~10 KB
- `convergence.cr` — Crystal convergence tool, ~9 KB

Both are standalone, not yet integrated with anything. Likely
exploratory tooling for convergence behavior in the trainer. Whether
they're worth integrating is unknown; they're 8+ weeks old.

## Why these aren't committed as code

Each piece is half-done sub-agent work on stale base commits. Applying
either to current HEAD requires real review:
- understanding what was being built and whether the design is still right
- checking for conflicts with intervening main commits
- replacing or merging with any equivalent work already on main

That's a deliberate restoration task, not a mechanical apply. These
files live in `notes/` so they're preserved without polluting the
build.
