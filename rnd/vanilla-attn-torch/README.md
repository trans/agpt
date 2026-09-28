---
title: Vanilla attention LM (PyTorch)
kind: baseline
status: concluded
outcome: n/a
question: >-
  How does the AGPT attention architecture (d64 L2 h4 ff256) perform when trained as a plain
  mini-batch Adam language model, without the trie, on the canonical held-out evaluation?
answer: >-
  It beats AGPT at this scale. From random init at seq 16 it reaches canonical byte PPL 4.46
  by epoch 10 and plateaus at 4.45, versus 4.85 for the canonical AGPT run (depth-16 trie,
  seed, 512 epochs, wrap). This is recorded only in the 2026-06-08 memory note; the run directories
  and trainer are not in the repository. (The run files were not kept; the numbers come from
  the 2026-06-08 project notes.)
opened: 2026-06-08
updated: 2026-06-10
code: main
eval: canonical
tags: [baseline, attention, optimizer]
related: [kenlm-baseline, partition-depth]
---

# Vanilla attention LM (PyTorch)

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** How does the AGPT attention architecture (d64 L2 h4 ff256) perform when trained as a plain mini-batch Adam language model, without the trie, on the canonical held-out evaluation?

**Answer.** It beats AGPT at this scale. From random init at seq 16 it reaches canonical byte PPL 4.46 by epoch 10 and plateaus at 4.45, versus 4.85 for the canonical AGPT run (depth-16 trie, seed, 512 epochs, wrap). This is recorded only in the 2026-06-08 memory note; the run directories and trainer are not in the repository. (The run files were not kept; the numbers come from the 2026-06-08 project notes.)

**Sources.** Memory note project_vanilla_attn_beats_agpt.md (table of vanilla SGD vs canonical AGPT). The directory itself holds no results.

**Caveats.** The directory is gitignored/untracked and contains only d64L2-seq32-scratch/train.log, a failed launch ('can't open file src/tools/agpt_vanilla_attn_train.py'). That trainer was never committed (not in any git history), and the run dirs named in memory (d64L2-seq8-scratch, d64L2-seq16-scratch, d64L2-seq16-fromseed) are gone. Memory also lists seq=32 ep50 at 4.17 on the same held-out, but the only surviving seq32 artefact is this failed launch, so that number's provenance needs checking. The headline values come from memory, not from a README or result.
