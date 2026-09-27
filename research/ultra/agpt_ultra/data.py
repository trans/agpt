from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharVocab:
    chars: tuple[str, ...]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        chars = tuple(sorted(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(chars=chars, stoi=stoi, itos=itos)

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def make_samples(text: str, block_size: int, stride: int = 1) -> list[str]:
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if len(text) < block_size:
        raise ValueError("text is shorter than block_size")
    return [text[i : i + block_size] for i in range(0, len(text) - block_size + 1, stride)]


def sorted_samples(text: str, block_size: int, stride: int = 1) -> list[str]:
    return sorted(make_samples(text, block_size, stride))


def make_circular_samples(text: str, block_size: int, stride: int = 1) -> list[str]:
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if len(text) < block_size:
        raise ValueError("text is shorter than block_size")
    padded = text + text[:block_size]
    return [padded[i : i + block_size] for i in range(0, len(text), stride)]


def sorted_circular_samples(text: str, block_size: int, stride: int = 1) -> list[str]:
    return sorted(make_circular_samples(text, block_size, stride))
