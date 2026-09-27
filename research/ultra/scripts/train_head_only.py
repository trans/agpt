from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, read_text, sorted_samples
from agpt_ultra.head_only import train_head_only
from agpt_ultra.model import TinyCharRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only the output head with trie natural-gradient updates.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--step-scale", default="1.0")
    parser.add_argument("--max-step-scale", type=float, default=1.0)
    parser.add_argument("--trust-radius", type=float, default=None)
    parser.add_argument("--line-search-steps", type=int, default=0)
    parser.add_argument("--max-cg-iter", type=int, default=None)
    parser.add_argument("--cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.step_scale = args.step_scale if args.step_scale == "auto" else float(args.step_scale)
    torch.manual_seed(args.seed)

    text = read_text(args.input)
    vocab = CharVocab.from_text(text)
    samples = sorted_samples(text, block_size=args.block_size, stride=args.stride)
    model = TinyCharRNN(vocab.size, n_embd=args.embedding_size, n_hidden=args.hidden_size)

    results = train_head_only(
        model,
        vocab,
        samples,
        epochs=args.epochs,
        damping=args.damping,
        step_scale=args.step_scale,
        max_cg_iter=args.max_cg_iter,
        cg_tolerance=args.cg_tolerance,
        max_step_scale=args.max_step_scale,
        line_search_steps=args.line_search_steps,
        trust_radius=args.trust_radius,
    )
    for epoch, result in enumerate(results, start=1):
        print(
            f"epoch={epoch:03d} "
            f"before_loss={result.before_loss:.4f} "
            f"after_loss={result.after_loss:.4f} "
            f"cg_iters={result.cg_iterations} "
            f"step_scale={result.step_scale} "
            f"suggested_step_scale={result.suggested_step_scale} "
            f"eta_quad={result.eta_quad} "
            f"eta_trust={result.eta_trust} "
            f"rho_train={result.train_improvement_ratio}"
        )


if __name__ == "__main__":
    main()
