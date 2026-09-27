from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab, read_text
from agpt_ultra.eval import LossMetrics, split_text
from agpt_ultra.model import TinyCharTransformer
from agpt_ultra.run_ids import prefixed_path, resolve_run_id


@dataclass(frozen=True)
class AttentionSequenceRow:
    step: int
    train_loss: float
    val_nll: float | None
    val_ppl: float | None
    val_bpc: float | None
    runtime_sec: float
    eval_sec: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny causal attention char LM.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/attention_sequence.csv"))
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    return parser.parse_args()


def encode_text(vocab: CharVocab, text: str) -> torch.Tensor:
    return torch.tensor(vocab.encode(text), dtype=torch.long)


def sample_batch(encoded: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if encoded.numel() <= block_size:
        raise ValueError("encoded text must be longer than block_size")
    starts = torch.randint(0, encoded.numel() - block_size, (batch_size,), generator=generator)
    x = torch.stack([encoded[start : start + block_size] for start in starts])
    y = torch.stack([encoded[start + 1 : start + block_size + 1] for start in starts])
    return x, y


@torch.no_grad()
def evaluate_windows(
    model: TinyCharTransformer,
    encoded: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> LossMetrics:
    model.eval()
    device = next(model.parameters()).device
    total_tokens = encoded.numel() - block_size
    total_loss = 0.0
    starts = torch.arange(0, total_tokens, device=device)
    for offset in range(0, starts.numel(), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        x = torch.stack([encoded[start : start + block_size] for start in batch_starts])
        y = torch.stack([encoded[start + 1 : start + block_size + 1] for start in batch_starts])
        logits = model(x)
        total_loss += float(F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1), reduction="sum").item())
    tokens = int(total_tokens * block_size)
    nll = total_loss / tokens
    return LossMetrics(
        loss=total_loss,
        tokens=tokens,
        nll_per_token=nll,
        perplexity=float(torch.exp(torch.tensor(nll)).item()),
    )


def write_header(path: Path) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(asdict(AttentionSequenceRow(0, 0.0, None, None, None, 0.0, None)).keys()),
    )
    writer.writeheader()
    handle.flush()
    return writer, handle


def main() -> None:
    args = parse_args()
    args.run_id = resolve_run_id(args.run_id)
    args.output = prefixed_path(args.output, args.run_id)
    if args.checkpoint_output is None:
        args.checkpoint_output = args.output.with_suffix(".pt")
    else:
        args.checkpoint_output = prefixed_path(args.checkpoint_output, args.run_id)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    text = read_text(args.input)
    split = split_text(text, train_fraction=args.train_fraction)
    vocab = CharVocab.from_text(text)
    train_ids = encode_text(vocab, split.train_text)
    val_ids = encode_text(vocab, split.val_text)
    model = TinyCharTransformer(
        vocab.size,
        block_size=args.block_size,
        n_embd=args.embedding_size,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer, handle = write_header(args.output)
    print(
        f"train_chars={len(split.train_text)} val_chars={len(split.val_text)} steps={args.steps} "
        f"batch_size={args.batch_size} block_size={args.block_size} dim={args.embedding_size} "
        f"layers={args.layers} heads={args.heads} run_id={args.run_id} output={args.output}",
        flush=True,
    )
    try:
        for step in range(1, args.steps + 1):
            started = time.perf_counter()
            model.train()
            x, y = sample_batch(train_ids, args.batch_size, args.block_size, generator)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab.size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            should_eval = step == 1 or step == args.steps or (args.eval_every > 0 and step % args.eval_every == 0)
            eval_sec = None
            val_nll = None
            val_ppl = None
            val_bpc = None
            if should_eval:
                eval_started = time.perf_counter()
                metrics = evaluate_windows(model, val_ids, args.block_size, args.eval_batch_size)
                eval_sec = time.perf_counter() - eval_started
                val_nll = metrics.nll_per_token
                val_ppl = metrics.perplexity
                val_bpc = metrics.bits_per_char
            row = AttentionSequenceRow(
                step=step,
                train_loss=float(loss.item()),
                val_nll=val_nll,
                val_ppl=val_ppl,
                val_bpc=val_bpc,
                runtime_sec=time.perf_counter() - started,
                eval_sec=eval_sec,
            )
            writer.writerow(asdict(row))
            handle.flush()
            if should_eval:
                print(
                    f"step={step:05d} train_loss={row.train_loss:.4f} "
                    f"val_ppl={row.val_ppl:.3f} val_bpc={row.val_bpc:.3f} eval_sec={row.eval_sec:.2f}",
                    flush=True,
                )
    finally:
        handle.close()
    torch.save(
        {
            "args": vars(args),
            "vocab_chars": vocab.chars,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        args.checkpoint_output,
    )
    print(f"checkpoint={args.checkpoint_output}", flush=True)
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
