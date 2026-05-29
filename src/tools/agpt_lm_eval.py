#!/usr/bin/env python3
"""Run AGPT evaluation protocols from one entrypoint.

This tool reports the two PPL protocols we currently care about:

  - lm-eval rolling PPL via `lm-evaluation-harness`
  - AGPT fixed-token PPL via the independent PyTorch reference model

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
import math
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
import agpt_ppl  # noqa: E402

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


def _write_local_chunks_task(
    task_name: str, chunk_files: list[Path], dest_dir: Path
) -> str:
    """Materialize an lm-eval task YAML pointing at a list of chunk files.

    Each chunk file becomes a separate test document. lm-eval scores them
    independently under `loglikelihood_rolling` and aggregates per-token NLL
    into the overall PPL, which avoids the boundary-context contamination that
    a concatenated heldout would introduce.
    """
    task_path = dest_dir / f"{task_name}.yaml"
    config = dict(SHAKESPEARE_TASK_TEMPLATE)
    config["task"] = task_name
    config["dataset_kwargs"] = {
        "data_files": {"test": [str(Path(c).resolve()) for c in chunk_files]},
        "sample_by": "document",
    }
    task_path.write_text(yaml.safe_dump(config))
    return task_name


def _resolve_chunks(chunks_dir: Path) -> tuple[list[Path], dict | None]:
    """Resolve a chunks directory into a sorted list of chunk files + manifest.

    Convention follows `bin/agpt_carve`'s output layout:
      <chunks_dir>/heldout_chunks/chunk_NN.txt  (preferred: chunks_dir is the
                                                 carve output root)
      <chunks_dir>/chunk_NN.txt                 (or chunks_dir IS the per-chunk
                                                 directory itself)

    If a `manifest.json` is found adjacent (either layout), it is returned for
    provenance recording in the result JSON.
    """
    chunks_dir = Path(chunks_dir).resolve()
    # Case A: chunks_dir IS the per-chunk directory (chunk_NN.txt at top level).
    direct = sorted(chunks_dir.glob("chunk_*.txt"))
    # Case B: chunks_dir is the carve output root with a heldout_chunks/ subdir.
    subdir = chunks_dir / "heldout_chunks"
    nested = sorted(subdir.glob("chunk_*.txt")) if subdir.exists() else []

    chunks = direct or nested
    if not chunks:
        raise ValueError(
            f"No chunk_*.txt files found in {chunks_dir} or {subdir}"
        )

    # Locate a manifest.json for provenance: adjacent to the chunks themselves
    # (carve output root) or two levels up. Optional; carve always writes one.
    manifest_path = chunks_dir / "manifest.json"
    if not manifest_path.exists() and nested:
        manifest_path = chunks_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = None
    return chunks, manifest


def _extract_lm_eval_metrics(results: dict, task_name: str) -> dict:
    """Flatten the lm-eval task metrics we show in experiment summaries."""
    task_results = results.get(task_name)
    if task_results is None and results:
        task_results = next(iter(results.values()))
    if not isinstance(task_results, dict):
        return {}

    out = {}
    mapping = {
        "word_perplexity,none": "lm_eval_rolling_word_perplexity",
        "byte_perplexity,none": "lm_eval_rolling_byte_perplexity",
        "bits_per_byte,none": "lm_eval_rolling_bits_per_byte",
    }
    for src, dest in mapping.items():
        val = task_results.get(src)
        if isinstance(val, (int, float)):
            out[dest] = float(val)
    return out


def _run_fixed_token_ppl(args: argparse.Namespace) -> dict:
    """Compute AGPT fixed-window token PPL.

    Source: --text-file (single corpus) or --chunks-dir (multi-chunk carve).
    For chunks-dir, each chunk is scored independently with its own fixed
    window pass; per-chunk NLL totals are summed and the aggregate PPL is
    exp(total_nll / total_tokens). No chunk-boundary contamination.
    """
    if not (args.text_file or args.chunks_dir):
        raise ValueError("fixed-token PPL requires --text-file or --chunks-dir")
    if not args.agpt_model:
        raise ValueError("fixed-token PPL requires --agpt-model")
    if not args.vocab_file:
        raise ValueError("fixed-token PPL requires --vocab-file")

    cfg, sd = agpt_ppl.load_model(args.agpt_model)
    d_window = args.fixed_context or cfg["seq_len"]
    device = args.fixed_device or args.device
    char_to_id, vocab_size = agpt_ppl.build_vocab(args.vocab_file)
    model = agpt_ppl.AGPTModel(cfg, sd, device=device).to(device)

    # Gather sources: one path (text_file) or many (chunks_dir).
    if args.text_file:
        sources = [Path(args.text_file)]
    else:
        sources, _ = _resolve_chunks(Path(args.chunks_dir))

    total_nll = 0.0
    total_tokens = 0
    per_chunk: list[dict] = []
    for src_path in sources:
        text = src_path.read_text(encoding="utf-8", errors="replace")
        tokens = [char_to_id[c] for c in text if c in char_to_id]
        if not tokens:
            continue
        ppl, n, start, stop = agpt_ppl.fixed_window_ppl(
            model,
            tokens,
            d_window,
            args.fixed_max_positions,
            device,
            args.fixed_batch_size,
        )
        if n > 0:
            # Reconstruct sum-NLL from per-token PPL so we can pool across chunks.
            chunk_nll = math.log(ppl) * n
            total_nll += chunk_nll
            total_tokens += n
            per_chunk.append({
                "path": str(src_path),
                "perplexity": ppl,
                "tokens_evaluated": n,
                "target_start": start,
                "target_stop": stop,
            })

    if total_tokens == 0:
        raise ValueError("fixed-token PPL: no tokens evaluated across all sources")
    agg_ppl = math.exp(total_nll / total_tokens)
    # target_start/target_stop are token offsets inside a single source array.
    # For a single source we report them at top level (preserves the previous
    # shape). For multi-source they live per-chunk only — the values from
    # separate sources don't combine into a continuous range.
    single_source = len(per_chunk) == 1
    return {
        "protocol": "agpt_fixed_token",
        "perplexity": agg_ppl,
        "tokens_evaluated": total_tokens,
        "target_start": per_chunk[0]["target_start"] if single_source else None,
        "target_stop": per_chunk[0]["target_stop"] if single_source else None,
        "context_tokens": d_window,
        "vocab_size": vocab_size,
        "model_seq_len": cfg["seq_len"],
        "per_chunk": per_chunk if not single_source else None,
    }


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
        "--chunks-dir",
        help=(
            "Directory of chunk_NN.txt files produced by `bin/agpt_carve` "
            "(or the carve output root, which has heldout_chunks/ inside). "
            "Scores each chunk as an independent document and aggregates "
            "per-token NLL — no chunk-boundary contamination, plus per-chunk "
            "breakdown if --log-samples is enabled."
        ),
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
    p.add_argument(
        "--log-samples",
        action="store_true",
        help=(
            "Pass log_samples=True to lm-eval to capture per-document metrics. "
            "Useful with --chunks-dir to get per-chunk PPL alongside the "
            "aggregate."
        ),
    )
    p.add_argument(
        "--agpt-model",
        help="Native .model checkpoint. Enables AGPT fixed-token PPL.",
    )
    p.add_argument(
        "--vocab-file",
        help="Vocab source for --agpt-model fixed-token PPL.",
    )
    p.add_argument(
        "--fixed-context",
        type=int,
        default=None,
        help="Fixed-token context length. Defaults to the .model seq_len.",
    )
    p.add_argument(
        "--fixed-max-positions",
        type=int,
        default=0,
        help="Cap fixed-token evaluated targets (0 = all).",
    )
    p.add_argument(
        "--fixed-batch-size",
        type=int,
        default=256,
        help="Batch size for fixed-token PPL.",
    )
    p.add_argument(
        "--fixed-device",
        default=None,
        help="Device for fixed-token PPL. Defaults to --device.",
    )
    args = p.parse_args()

    if args.agpt_model and not (args.text_file or args.chunks_dir):
        p.error("--agpt-model fixed-token PPL requires --text-file or --chunks-dir")
    if args.agpt_model and not args.vocab_file:
        p.error("--agpt-model fixed-token PPL requires --vocab-file")

    chunks_manifest = None
    with tempfile.TemporaryDirectory() as tdir:
        tdir_path = Path(tdir)
        if args.text_file:
            task = _write_local_text_task(
                args.task_name, args.text_file, tdir_path
            )
            include_path = str(tdir_path)
        elif args.chunks_dir:
            chunk_files, chunks_manifest = _resolve_chunks(Path(args.chunks_dir))
            task = _write_local_chunks_task(
                args.task_name, chunk_files, tdir_path
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
            log_samples=args.log_samples,
        )

    lm_results = result["results"]
    fixed = _run_fixed_token_ppl(args) if args.agpt_model else None
    metrics = _extract_lm_eval_metrics(lm_results, args.task_name)
    if fixed:
        metrics["agpt_fixed_token_perplexity"] = fixed["perplexity"]

    # Per-chunk breakdown if log_samples was requested. lm-eval exposes
    # per-document loglikelihoods in result["samples"][<task>] as a list
    # of dicts; we report each chunk's PPL alongside the aggregate.
    chunk_breakdown = None
    if args.log_samples and args.chunks_dir and "samples" in result:
        samples = result["samples"].get(args.task_name, [])
        chunk_breakdown = []
        for i, s in enumerate(samples):
            # Each sample has 'resps' = [[(loglikelihood, ...)]] and
            # 'doc' with the actual text. Compute per-chunk byte-PPL from
            # the loglikelihood (sum NLL over chars).
            try:
                resps = s.get("resps", [])
                ll = float(resps[0][0][0]) if resps and resps[0] and resps[0][0] else None
            except (TypeError, IndexError, ValueError):
                ll = None
            doc_text = s.get("doc", {}).get("text", "") if isinstance(s.get("doc"), dict) else ""
            n_bytes = len(doc_text.encode("utf-8"))
            entry = {"chunk_index": i, "n_bytes": n_bytes}
            if ll is not None and n_bytes > 0:
                # rolling-PPL convention: ll is sum of token log-probs over the
                # document (natural log). byte_perplexity = exp(-ll / n_bytes).
                entry["byte_perplexity"] = math.exp(-ll / n_bytes)
                entry["log_likelihood"] = ll
            chunk_breakdown.append(entry)

    output = {
        "schema": "agpt_eval.v1",
        "metrics": metrics,
        "lm_eval": lm_results,
        "agpt_fixed": fixed,
    }
    if chunks_manifest is not None:
        output["chunks_manifest"] = chunks_manifest
    if chunk_breakdown is not None:
        output["per_chunk"] = chunk_breakdown

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
