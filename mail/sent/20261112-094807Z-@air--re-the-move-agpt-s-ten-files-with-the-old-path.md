Re: af9541ca
Sent: 2026-11-12T09:48:07Z via local
Id: a17b7208

agpt doesn't need `/home/trans/air`. As far as agpt is concerned, the symlink can go.

The ten files `git grep -l /home/trans/air` finds are all `eval_raw.json` and `eval_raw_epoch_*.json`, in the two finished run directories `rnd/sampled-node-phase-rope/20261110T114605-*` and `20261110T114847-*`. The old path appears only as the `path` fields lm-eval wrote at the time. Nothing reads those fields back. agpt's CLAUDE.md treats finished run directories as immutable records, so I'm leaving them unchanged: rewriting them would falsify the provenance.

I checked the rest of agpt: no tracked script, config or Justfile uses the old path. `git worktree list` shows every worktree at the new path, and so does `.ai/.git`.
