from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.baseline import baseline_epoch, baseline_parameters
from agpt_ultra.data import CharVocab, read_text
from agpt_ultra.eval import evaluate_trie_loss, make_sample_split
from agpt_ultra.flat_ops import samples_to_flat_trie
from agpt_ultra.model import TinyCharRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU char model with AdamW on the trie objective.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    optimizer = torch.optim.AdamW(baseline_parameters(model), lr=args.lr)

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
    for epoch in range(1, args.epochs + 1):
        result = baseline_epoch(
            model,
            vocab,
            split.train_samples,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
            flat_trie=train_trie,
        )
        val = evaluate_trie_loss(model, vocab, val_trie)
        print(
            f"epoch={epoch:03d} "
            f"before_loss={result.before_loss:.4f} "
            f"after_loss={result.after_loss:.4f} "
            f"val_nll={val.nll_per_token:.4f} "
            f"val_ppl={val.perplexity:.2f}"
        )


if __name__ == "__main__":
    main()
