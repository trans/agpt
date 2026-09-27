from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, read_text, sorted_samples
from agpt_ultra.eval import evaluate_trie_loss, make_sample_split
from agpt_ultra.flat_ops import samples_to_flat_trie
from agpt_ultra.hybrid import freeze_embeddings, hybrid_epoch
from agpt_ultra.model import TinyCharRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU cell with trie loss and output head with trie Fisher.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--body-lr", type=float, default=1e-3)
    parser.add_argument("--head-damping", type=float, default=1.0)
    parser.add_argument("--head-step-scale", default="0.5")
    parser.add_argument("--max-head-step-scale", type=float, default=1.0)
    parser.add_argument("--head-trust-radius", type=float, default=None)
    parser.add_argument("--line-search-steps", type=int, default=0)
    parser.add_argument("--line-search-target", choices=["train", "val"], default="train")
    parser.add_argument("--skip-before-loss", action="store_true")
    parser.add_argument("--max-cg-iter", type=int, default=None)
    parser.add_argument("--cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--update-embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.head_step_scale = args.head_step_scale if args.head_step_scale == "auto" else float(args.head_step_scale)
    torch.manual_seed(args.seed)

    text = read_text(args.input)
    vocab = CharVocab.from_text(text)
    split = make_sample_split(
        text,
        block_size=args.block_size,
        stride=args.stride,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    model = TinyCharRNN(vocab.size, n_embd=args.embedding_size, n_hidden=args.hidden_size)
    train_trie = samples_to_flat_trie(split.train_samples, vocab)
    val_trie = samples_to_flat_trie(split.val_samples, vocab)
    before_val = evaluate_trie_loss(model, vocab, val_trie)
    print(
        f"samples train={len(split.train_samples)} "
        f"val={len(split.val_samples)} "
        f"vocab={vocab.size}"
    )
    print(
        f"epoch=000 "
        f"train_loss=nan "
        f"val_nll={before_val.nll_per_token:.4f} "
        f"val_ppl={before_val.perplexity:.2f}"
    )

    if not args.update_embeddings:
        freeze_embeddings(model)
    body_params = [*model.cell.parameters()]
    if args.update_embeddings:
        body_params.extend(model.embed.parameters())
    optimizer = torch.optim.AdamW(body_params, lr=args.body_lr)
    for epoch in range(1, args.epochs + 1):
        result = hybrid_epoch(
            model,
            vocab,
            split.train_samples,
            body_optimizer=optimizer,
            head_damping=args.head_damping,
            head_step_scale=args.head_step_scale,
            max_grad_norm=args.max_grad_norm,
            max_cg_iter=args.max_cg_iter,
            cg_tolerance=args.cg_tolerance,
            flat_trie=train_trie,
            max_head_step_scale=args.max_head_step_scale,
            line_search_steps=args.line_search_steps,
            line_search_flat_trie=val_trie if args.line_search_target == "val" else None,
            calibration_flat_trie=val_trie,
            trust_radius=args.head_trust_radius,
            compute_before_loss=not args.skip_before_loss,
            update_embeddings=args.update_embeddings,
        )
        val = evaluate_trie_loss(model, vocab, val_trie)
        before_loss = "None" if result.before_loss is None else f"{result.before_loss:.4f}"
        print(
            f"epoch={epoch:03d} "
            f"before_loss={before_loss} "
            f"after_body_loss={result.after_body_loss:.4f} "
            f"after_loss={result.after_loss:.4f} "
            f"cg_iters={result.head_result.cg_iterations} "
            f"step_scale={result.head_result.step_scale} "
            f"suggested_step_scale={result.head_result.suggested_step_scale} "
            f"eta_quad={result.head_result.eta_quad} "
            f"eta_trust={result.head_result.eta_trust} "
            f"rho_train={result.head_result.train_improvement_ratio} "
            f"rho_val={result.head_result.val_improvement_ratio} "
            f"val_nll={val.nll_per_token:.4f} "
            f"val_ppl={val.perplexity:.2f}"
        )


if __name__ == "__main__":
    main()
