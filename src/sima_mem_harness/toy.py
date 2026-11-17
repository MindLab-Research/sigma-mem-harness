"""Deterministic backend used by tests and the local smoke command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .backend import LogicalKVCache
from .memory import SigmaMemory


@dataclass(frozen=True)
class TraceCall:
    operation: str
    token_ids: tuple[int, ...]
    read_enabled: bool


class TraceBackend:
    """A tiny logical model that makes read/write gating observable."""

    def __init__(self, *, dim: int = 4, vocab_size: int = 1024) -> None:
        self.dim = dim
        self.vocab_size = vocab_size
        self.calls: list[TraceCall] = []
        self.write_calls: list[tuple[int, ...]] = []
        self.prepare_calls = 0
        self.fail_next_prepare = False

    def write_memory(self, token_ids: Sequence[int]) -> SigmaMemory:
        ids = tuple(int(token) for token in token_ids)
        self.write_calls.append(ids)
        features = torch.zeros(len(ids), self.dim)
        for row, token in enumerate(ids):
            features[row, token % self.dim] = 1.0
        return SigmaMemory.from_features(features)

    def prepare_read_state(self, memory: SigmaMemory) -> Any:
        self.prepare_calls += 1
        if self.fail_next_prepare:
            self.fail_next_prepare = False
            raise RuntimeError("injected reader-state preparation failure")
        return memory

    def append_native(self, cache, token_ids, *, read_state):
        previous = () if cache is None else cache.token_ids
        ids = tuple(previous) + tuple(int(token) for token in token_ids)
        self.calls.append(TraceCall("append_native", tuple(token_ids), read_state is not None))
        return LogicalKVCache(ids)

    def rebuild_native(self, token_ids, *, read_state):
        ids = tuple(int(token) for token in token_ids)
        self.calls.append(TraceCall("rebuild_native", ids, read_state is not None))
        return LogicalKVCache(ids)

    def generate(self, cache, *, read_state, max_new_tokens, eos_token_id, on_token_chunk: Callable | None = None):
        self.calls.append(TraceCall("generate", cache.token_ids, read_state is not None))
        output = tuple((900 + i) % self.vocab_size for i in range(max_new_tokens))
        if on_token_chunk is not None and output:
            for token in output:
                on_token_chunk((token,))
        return output


__all__ = ["TraceBackend", "TraceCall"]
