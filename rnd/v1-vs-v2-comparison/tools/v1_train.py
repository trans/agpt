#!/usr/bin/env python3
"""v1-trainer wrapper that accepts the orchestrator's standard flags and
translates them to bin/agpt_train's CLI.

The v1 trainer needs a pre-built radix trie (--trie-dir), unlike v2
which builds it on the fly under --mode train-growth. This wrapper
maps the orchestrator's --corpus to the canonical radix dir
(/tmp/<basename>_d{growth_max_depth}_static_radix or similar) and
builds it if missing.

Drops v2-only flags silently (--mode, --growth-*, --chunk-queries).
Keeps everything else.

Lives under rnd/v1-vs-v2-comparison/tools/ per the per-experiment-tools
convention; this is not a general-purpose v1 wrapper.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def resolve_radix_dir(corpus_path: str, depth: int) -> str:
    """Map a corpus path to its pre-built radix dir. Errors if not found
    (rather than auto-building, since the bin/agpt_build_radix CLI takes
    --leveled and other flags we don't want to guess at here).
    """
    basename = Path(corpus_path).stem
    candidates = [
        f"/tmp/{basename}_d{depth}_radix",
        f"/tmp/{basename}_d{depth}_static_radix",
    ]
    for c in candidates:
        if Path(c, "meta.bin").exists():
            print(f"v1_train: using radix at {c}", file=sys.stderr)
            return c
    raise FileNotFoundError(
        f"v1_train: no radix found for {corpus_path} at d={depth}. "
        f"Build it first: bin/agpt_build_radix --leveled <leveled-dir> "
        f"--out <radix-dir>  (or pre-stage at one of: {candidates})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Orchestrator's standard flags
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--save", required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--optimizer", default="rmsprop")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--rmsprop-beta", type=float, default=0.999)
    p.add_argument("--momentum-beta", type=float, default=0.9)
    p.add_argument("--lr-schedule", default="warmup-cosine")
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--partition-depth", type=int, default=1)
    p.add_argument("--chunk-queries", type=int, default=50000)
    p.add_argument("--accumulate", action="store_true")
    p.add_argument("--no-accumulate", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--anc-grad", action="store_true",
                   help="redundant (default-on); kept for compat")
    p.add_argument("--ablate-anc-grad", action="store_true")
    p.add_argument("--mass-weight", default=None,
                   help="off|log|sqrt|linear — v1's per-query mass weighting")

    # v2-only flags: accept and drop
    for flag in ["--mode", "--growth-frontiers", "--growth-divisions",
                 "--growth-max-depth", "--growth-min-epochs",
                 "--growth-epoch-ramp", "--growth-epoch-schedule",
                 "--growth-final-frontier", "--growth-train-frac",
                 "--rope-position-mode", "--position-data",
                 "--pos-sample-seed"]:
        p.add_argument(flag, default=None, help=argparse.SUPPRESS)

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"v1_train: WARN: unknown flags dropped: {unknown}", file=sys.stderr)

    # Decide depth — orchestrator usually passes via --growth-max-depth
    depth = int(getattr(args, "growth_max_depth", None) or 16)

    # Resolve radix dir
    radix_dir = resolve_radix_dir(args.corpus, depth)

    # Construct v1 CLI
    cmd = [
        "bin/agpt_train",
        "--model", args.model,
        "--trie-dir", radix_dir,
        "--save", args.save,
        "--epochs", str(args.epochs),
        "--optimizer", args.optimizer,
        "--lr", str(args.lr),
        "--rmsprop-beta", str(args.rmsprop_beta),
        "--momentum-beta", str(args.momentum_beta),
        "--lr-schedule", args.lr_schedule,
        "--warmup-epochs", str(args.warmup_epochs),
        "--partition-depth", str(args.partition_depth),
        "--chunk-queries", str(args.chunk_queries),
    ]
    if args.accumulate:
        cmd.append("--accumulate")
    if args.no_accumulate:
        cmd.append("--no-accumulate")
    if args.quiet:
        cmd.append("--quiet")
    if args.anc_grad:
        cmd.append("--anc-grad")
    if args.ablate_anc_grad:
        cmd.append("--ablate-anc-grad")
    if args.mass_weight:
        cmd.extend(["--mass-weight", args.mass_weight])

    print(f"v1_train: invoking: {' '.join(cmd)}", file=sys.stderr)
    sys.stderr.flush()
    rc = subprocess.run(cmd).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
