from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, read_text, sorted_samples
from agpt_ultra.model import TinyCharRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny sorted-sample character RNN.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    text = read_text(args.input)
    vocab = CharVocab.from_text(text)
    samples = sorted_samples(text, block_size=args.block_size, stride=args.stride)
    encoded = torch.tensor([vocab.encode(sample) for sample in samples], dtype=torch.long)

    model = TinyCharRNN(vocab.size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        ix = torch.randint(0, encoded.shape[0], (args.batch_size,))
        batch = encoded[ix]
        x = batch[:, :-1]
        y = batch[:, 1:]

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab.size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0:
            print(f"step={step:04d} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
