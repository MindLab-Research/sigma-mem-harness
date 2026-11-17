"""Reference adapter for the current DeepSeek-V4.1 Sigma implementation.

The existing ``SigmaV41.generate`` path is a correctness/reference path and
does not expose incremental paged KV yet. This adapter therefore keeps a
logical token cache and delegates the actual read to ``SigmaV41.generate``.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .backend import LogicalKVCache
from .memory import SigmaMemory


class SigmaV41ReferenceBackend:
    def __init__(self, model) -> None:
        self.model = model

    def write_memory(self, token_ids: Sequence[int]) -> SigmaMemory:
        return self.model.encode(list(token_ids), storage_device="cpu")

    def prepare_read_state(self, memory: SigmaMemory) -> SigmaMemory:
        return memory

    def append_native(self, cache, token_ids, *, read_state):
        previous = () if cache is None else cache.token_ids
        return LogicalKVCache(tuple(previous) + tuple(int(token) for token in token_ids))

    def rebuild_native(self, token_ids, *, read_state):
        return LogicalKVCache(tuple(int(token) for token in token_ids))

    def generate(self, cache, *, read_state, max_new_tokens, eos_token_id, on_token_chunk: Callable | None = None):
        memory = read_state if read_state is not None else self.model.writer.empty(device="cpu")
        output = self.model.generate(
            list(cache.token_ids),
            memory,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        generated = output.tolist()[len(cache.token_ids) :]
        # The reference model is not streaming yet. This callback still lets
        # the harness exercise write scheduling; a real serving backend should
        # invoke it incrementally during decode.
        if on_token_chunk is not None and generated:
            on_token_chunk(generated)
        return generated


__all__ = ["SigmaV41ReferenceBackend"]
