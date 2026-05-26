#!/usr/bin/env python3
"""β-aware byte_perplexity evaluator for the harmonic-bias prototype.

The canonical orchestrator evaluator (`src/tools/agpt_lm_eval.py`) runs
the HF wrapper through lm-evaluation-harness. That path has no slot for
β, so the bias-trained model's β contribution is silently dropped at
eval. This tool is a copy of agpt_lm_eval.py's intent but does its own
forward loop using HarmonicBiasModel, so β IS exercised at eval.

It writes the SAME JSON shape lm-evaluation-harness produces, so the
orchestrator's parse_eval_json picks up word_perplexity / byte_perplexity
/ bits_per_byte exactly as for the canonical evaluator.

This tool lives under rnd/harmonic-bias-prototype/tools/ (not src/tools/)
because it's experiment-specific. If the bias direction proves out and
graduates to canonical, β goes into the .model format and this tool
folds back into the canonical evaluator.

Usage:
    python3 rnd/harmonic-bias-prototype/tools/lm_eval_with_bias.py \\
        --hf-dir HF_DIR --text-file CORPUS.txt \\
        --beta-path STATE.beta.pt \\
        --task-name shake_bias_ppl \\
        --out-json result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_SRC_TOOLS = _REPO_ROOT / "src" / "tools"
for p in (str(_HERE), str(_SRC_TOOLS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import agpt_hf  # noqa: F401,E402  — side effect: HF registration
from agpt_hf import AGPTForCausalLM  # noqa: E402
from agpt_ppl import build_vocab  # noqa: E402
from bias import HarmonicBiasModel, precompute_chords, byte_perplexity_pytorch  # noqa: E402


def tokenize_corpus(corpus_path: str, vocab_path: str | None = None) -> torch.Tensor:
    """Tokenize corpus_path using vocab from vocab_path (the full training
    corpus). Critical: building vocab from the eval slice alone gives a
    different char→id mapping than training used, scrambling the model.
    """
    char_to_id, _ = build_vocab(vocab_path or corpus_path)
    text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    return torch.tensor(
        [char_to_id.get(c, 0) for c in text], dtype=torch.long,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--text-file", required=True,
                   help="Eval corpus (already the held-out slice).")
    p.add_argument("--beta-path", default=None,
                   help=".beta.pt file written by train.py --save-beta. "
                        "If omitted, looks at <hf-dir>/../checkpoint.beta.pt "
                        "(the orchestrator-convention location).")
    p.add_argument("--task-name", default="bias_ppl",
                   help="Cosmetic key in the output JSON.")
    p.add_argument("--out-json", default=None,
                   help="Where to write the result JSON (defaults to stdout only).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on eval tokens (after d_window) — for smoke.")
    # Flags the orchestrator passes for the canonical evaluator; ignore here.
    p.add_argument("--agpt-model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--vocab-file", default=None,
                   help="Vocab corpus (must match training). Required.")
    p.add_argument("--fixed-context", default=None, help=argparse.SUPPRESS)
    p.add_argument("--fixed-max-positions", default=None, help=argparse.SUPPRESS)
    p.add_argument("--fixed-batch-size", default=None, help=argparse.SUPPRESS)
    p.add_argument("--fixed-device", default=None, help=argparse.SUPPRESS)
    p.add_argument("--builtin-task", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"loading HF model {args.hf_dir}", file=sys.stderr)
    hf_model = AGPTForCausalLM.from_pretrained(args.hf_dir).to(device)
    beta_path = args.beta_path
    if beta_path is None:
        beta_path = str(Path(args.hf_dir).resolve().parent / "checkpoint.beta.pt")
    print(f"loading β sidecar {beta_path}", file=sys.stderr)
    beta_state = torch.load(beta_path, map_location="cpu", weights_only=False)
    n_freq = int(beta_state["n_freq"])
    window_W = int(beta_state["window_W"])
    d_window = int(beta_state["d_window"])
    beta = beta_state["beta"]
    print(f"  n_freq={n_freq} window_W={window_W} d_window={d_window} "
          f"β shape={tuple(beta.shape)}", file=sys.stderr)

    model = HarmonicBiasModel(hf_model, n_freq=n_freq).to(device)
    with torch.no_grad():
        model.beta.copy_(beta.to(device))

    print(f"tokenizing {args.text_file} (vocab from {args.vocab_file or args.text_file})", file=sys.stderr)
    ids = tokenize_corpus(args.text_file, vocab_path=args.vocab_file)
    if args.limit:
        ids = ids[: d_window + args.limit].contiguous()
    print(f"  {ids.numel()} tokens", file=sys.stderr)

    print(f"precomputing chords on eval corpus", file=sys.stderr)
    chord_table = precompute_chords(
        ids, d_window, window_W, n_freq,
    ).to(device)

    print(f"running PyTorch eval WITH bias", file=sys.stderr)
    m_bias = byte_perplexity_pytorch(
        model, ids, chord_table, d_window, device,
        batch_size=args.batch_size, use_bias=True,
    )
    print(f"running PyTorch eval WITHOUT bias (same model, β masked)", file=sys.stderr)
    m_nobias = byte_perplexity_pytorch(
        model, ids, chord_table, d_window, device,
        batch_size=args.batch_size, use_bias=False,
    )

    # Emit in the orchestrator's NEW metrics-block shape (matches Codex's
    # agpt_lm_eval.py output). `agpt_fixed_token_perplexity` is the
    # canonical row in the experiment table; with_bias=True is what
    # actually answers the prototype's question. We also include the
    # no-bias number so the table can show both columns if wanted.
    metrics = {
        "agpt_fixed_token_perplexity": m_bias["byte_perplexity"],
        "agpt_fixed_token_perplexity_no_bias": m_nobias["byte_perplexity"],
        "bits_per_byte": m_bias["bits_per_byte"],
        "bits_per_byte_no_bias": m_nobias["bits_per_byte"],
        "n_scored": m_bias["n_scored"],
    }
    result = {
        "protocol": "agpt_fixed_token_with_bias",
        "d_window": d_window,
        "metrics": metrics,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)


if __name__ == "__main__":
    main()
