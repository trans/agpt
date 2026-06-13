from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agpt_ultra.data import CharVocab
from agpt_ultra.flat_trie import FlatTrie
from scripts.run_book_state_model import longest_suffix_node, trie_children


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a pure book trie-state checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--show-stats", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CharVocab, FlatTrie, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    vocab_chars = tuple(checkpoint["vocab_chars"])
    vocab = CharVocab(
        chars=vocab_chars,
        stoi={ch: index for index, ch in enumerate(vocab_chars)},
        itos={index: ch for index, ch in enumerate(vocab_chars)},
    )
    trie_data = checkpoint["trie"]
    trie = FlatTrie(
        vocab_size=int(trie_data["vocab_size"]),
        parents=trie_data["parents"].long(),
        tokens=trie_data["tokens"].long(),
        depths=trie_data["depths"].long(),
        counts=trie_data["counts"].float(),
        transition_counts=trie_data["transition_counts"].float(),
    )
    block_size = int(checkpoint["args"]["block_size"])
    return (
        checkpoint["states"].float(),
        checkpoint["weight"].float(),
        checkpoint["bias"].float(),
        vocab,
        trie,
        block_size,
    )


def sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator,
) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, k=top_k)
        probs = F.softmax(values, dim=0)
        sampled = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(indices[sampled].item())
    probs = F.softmax(logits, dim=0)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def main() -> None:
    args = parse_args()
    states, weight, bias, vocab, trie, block_size = load_checkpoint(args.checkpoint)
    children = trie_children(trie)
    logits_by_node = states @ weight.T + bias
    generator = torch.Generator().manual_seed(args.seed)

    output = args.prompt
    suffix_depths: list[int] = []
    for _ in range(args.tokens):
        context_text = output[-block_size:]
        context = [vocab.stoi[ch] for ch in context_text if ch in vocab.stoi]
        node_id = longest_suffix_node(context, children)
        suffix_depths.append(int(trie.depths[node_id].item()))
        token_id = sample_next(
            logits_by_node[node_id],
            temperature=args.temperature,
            top_k=args.top_k,
            generator=generator,
        )
        output += vocab.itos[token_id]

    print(output, end="" if output.endswith("\n") else "\n")
    if args.show_stats:
        if suffix_depths:
            mean_depth = sum(suffix_depths) / len(suffix_depths)
            max_depth = max(suffix_depths)
        else:
            mean_depth = 0.0
            max_depth = 0
        print(
            f"\n[stats] tokens={args.tokens} mean_suffix_depth={mean_depth:.2f} max_suffix_depth={max_depth}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
