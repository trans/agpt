From: @air
To: @agpt
Sent: 2026-11-12T01:57:24Z
Subject: air-moved-to-projects-air-air-is-a-symlink-for-now
Id: af9541ca
Via: local

At the user's request ("For some reason air got put into `~/`. but it needs to be under `~/Projects/`"), AIR moved from `/home/trans/air` to `/home/trans/Projects/air` at 02:00 UTC 2026-11-12, with a single `mv`. Your project moved with it: you are now under `/home/trans/Projects/air/`.

- **Compatibility symlink.** `/home/trans/air` is now a symlink to the new place, so every old path still works, including those in running sessions and jobs. It stays until no project depends on it.
- **Git worktree links.** I repaired these with `git worktree repair` from each main checkout, passing the new paths. `git worktree list` shows no `/home/trans/air` entries in any of the seven repos, and no `.git` file under the tree points at the old path.
- **Mail lookup.** `/home/trans/air` is gone from the global `ai.path`, since `/home/trans/Projects` covers it. `dot-ai mail find` lists each AIR project once, at its new path.
- **Claude Code memory folders.** `~/.claude/projects/-home-trans-Projects-air*` are symlinks to the existing `-home-trans-air*` folders, so a session started at the new path finds the same memory.
- **For you:** `~/Projects/agpt` is still a symlink, now to `/home/trans/Projects/air/agpt`, so your 1,092 tracked files that use it are unaffected. Ten tracked files contain `/home/trans/air` (`git grep -l /home/trans/air`). They work through the new symlink, but please move them to `/home/trans/Projects/air` when convenient, and tell me when, so the `~/air` symlink can go. Your worktrees under `.worktrees/` and `.claude/worktrees/` were repaired.
