#!/usr/bin/env python3
"""Run lm-evaluation-harness against an AGPT HF model directory.

This is the canonical AGPT evaluator. Uses `lm-evaluation-harness` so
results are directly comparable to published numbers for any task that
harness supports.

Importing `agpt_hf` registers our model with HF's auto classes, so any
HF directory written by `agpt_hf.py convert` is loadable here with
no `trust_remote_code` dance.

Usage:
    # Evaluate on a local text file (Shakespeare PPL is the default
    # bring-up test).
    python3 src/tools/agpt_lm_eval.py \\
        --hf-dir /tmp/agpt_hf_test \\
        --text-file data/input.txt \\
        --task-name shakespeare_ppl

    # Evaluate on a built-in lm-eval task (downloads the dataset).
    python3 src/tools/agpt_lm_eval.py \\
        --hf-dir /tmp/agpt_hf_test \\
        --builtin-task wikitext
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

# Importing agpt_hf registers AGPT with HF's auto classes; lm-eval will
# then load our model directories transparently.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import agpt_hf  # noqa: F401,E402  — side effect: HF registration

from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402


SHAKESPEARE_TASK_TEMPLATE = {
    "task": "TASK_NAME_PLACEHOLDER",
    "dataset_path": "text",
    "dataset_kwargs": {
        "data_files": {"test": "DATA_FILE_PLACEHOLDER"},
        "sample_by": "document",
    },
    "test_split": "test",
    "output_type": "loglikelihood_rolling",
    "doc_to_text": "",
    "doc_to_target": "{{text}}",
    "metric_list": [
        {"metric": "word_perplexity"},
        {"metric": "byte_perplexity"},
        {"metric": "bits_per_byte"},
    ],
    "metadata": {"version": 1.0},
}


def _write_local_text_task(
    task_name: str, text_file: str | Path, dest_dir: Path
) -> str:
    """Materialize an lm-eval task YAML pointing at a local text file."""
    task_path = dest_dir / f"{task_name}.yaml"
    config = dict(SHAKESPEARE_TASK_TEMPLATE)
    config["task"] = task_name
    config["dataset_kwargs"] = {
        "data_files": {"test": str(Path(text_file).resolve())},
        "sample_by": "document",
    }
    task_path.write_text(yaml.safe_dump(config))
    return task_name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--hf-dir",
        required=True,
        help="HF model directory (produced by `agpt_hf.py convert`)",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--text-file",
        help="Local plain-text corpus. Will compute PPL on the whole file.",
    )
    src.add_argument(
        "--builtin-task",
        help="Name of a built-in lm-eval task (e.g. wikitext).",
    )
    p.add_argument(
        "--task-name",
        default="local_text_ppl",
        help="Name of the task when using --text-file (cosmetic).",
    )
    p.add_argument(
        "--batch-size", type=int, default=1, help="lm-eval batch size."
    )
    p.add_argument(
        "--device", default="cpu", help="cpu or cuda."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of documents (useful for smoke tests).",
    )
    p.add_argument(
        "--out-json",
        help="Optional path to write the full lm-eval result JSON to.",
    )
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tdir:
        tdir_path = Path(tdir)
        if args.text_file:
            task = _write_local_text_task(
                args.task_name, args.text_file, tdir_path
            )
            include_path = str(tdir_path)
        else:
            task = args.builtin_task
            include_path = None

        model_args = f"pretrained={args.hf_dir},dtype=float32"
        task_manager = (
            TaskManager(include_path=include_path) if include_path else None
        )
        result = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=[task],
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            task_manager=task_manager,
        )

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result["results"], indent=2))
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
