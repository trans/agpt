from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class PrefixNode:
    token: str | None
    depth: int
    count: int = 0
    terminal_count: int = 0
    children: dict[str, "PrefixNode"] = field(default_factory=dict)

    @property
    def prefix_count(self) -> int:
        return self.count

    def child(self, token: str) -> "PrefixNode":
        node = self.children.get(token)
        if node is None:
            node = PrefixNode(token=token, depth=self.depth + 1)
            self.children[token] = node
        return node


@dataclass(frozen=True)
class StackEvent:
    common_prefix: int
    pop_count: int
    push_tokens: tuple[str, ...]
    sample: str


class PrefixTrie:
    def __init__(self) -> None:
        self.root = PrefixNode(token=None, depth=0)

    @classmethod
    def from_samples(cls, samples: Iterable[str]) -> "PrefixTrie":
        trie = cls()
        for sample in samples:
            trie.insert(sample)
        return trie

    def insert(self, sample: str) -> None:
        node = self.root
        node.count += 1
        for token in sample:
            node = node.child(token)
            node.count += 1
        node.terminal_count += 1

    def nodes(self) -> list[PrefixNode]:
        out: list[PrefixNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(reversed(list(node.children.values())))
        return out

    def node_count(self) -> int:
        return len(self.nodes())


def longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def stack_events(sorted_samples: list[str]) -> list[StackEvent]:
    events: list[StackEvent] = []
    previous = ""
    for sample in sorted_samples:
        common = longest_common_prefix(previous, sample)
        events.append(
            StackEvent(
                common_prefix=common,
                pop_count=len(previous) - common,
                push_tokens=tuple(sample[common:]),
                sample=sample,
            )
        )
        previous = sample
    return events
