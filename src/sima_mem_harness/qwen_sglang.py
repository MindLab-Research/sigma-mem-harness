"""Reference backend for the original Qwen Σ-mem SGLang path.

This adapter intentionally keeps the harness' native cache logical. The
original delta-mem SGLang engine owns its own request KV cache; the adapter
feeds it the current logical token sequence and arms its per-session Sigma
reader through ``delta_set_sigma_memory``. It is useful for protocol and
write-schedule validation before wiring a serving-side paged-KV object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .backend import LogicalKVCache
from .memory import SigmaMemory


@dataclass(frozen=True)
class QwenSGLangReadState:
    session_id: str
    memory_version: int


class QwenSGLangBackend:
    """Wrap ``delta_mem_miles.sglang.engine_control`` for Sigma reads."""

    def __init__(self, engine: Any, writer: Any, *, session_id: str = "sima-harness") -> None:
        self.engine = engine
        self.writer = writer
        self.session_id = session_id
        self._memory_version = 0
        self._read_index = 0

        from delta_mem_miles.sglang.common import delta_read_rid

        self._delta_read_rid = delta_read_rid

    def write_memory(self, token_ids: Sequence[int]) -> SigmaMemory:
        try:
            return self.writer.encode(list(token_ids), storage_device="cpu")
        except TypeError:
            # Original delta-mem-miles writer API returns its configured device
            # directly; newer writer variants accept storage_device.
            return self.writer.encode(list(token_ids))

    def prepare_read_state(self, memory: SigmaMemory) -> QwenSGLangReadState:
        self._memory_version += 1
        self.engine.delta_set_sigma_memory(self.session_id, memory)
        return QwenSGLangReadState(self.session_id, self._memory_version)

    def append_native(self, cache, token_ids, *, read_state):
        previous = () if cache is None else cache.token_ids
        return LogicalKVCache(tuple(previous) + tuple(int(token) for token in token_ids))

    def rebuild_native(self, token_ids, *, read_state):
        return LogicalKVCache(tuple(int(token) for token in token_ids))

    def generate(
        self,
        cache,
        *,
        read_state,
        max_new_tokens,
        eos_token_id,
        on_token_chunk: Callable[[Sequence[int]], None] | None = None,
    ):
        sampling_params = {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "max_new_tokens": int(max_new_tokens),
        }
        rid = None
        if read_state is not None:
            rid = self._delta_read_rid(self.session_id, self._read_index)
            self._read_index += 1
        result = self.engine.generate(
            input_ids=list(cache.token_ids),
            sampling_params=sampling_params,
            rid=rid,
            cache_salt=f"sima-mem-harness:{self.session_id}",
        )
        generated = result.get("output_ids", result.get("token_ids", []))
        if generated and isinstance(generated[0], list):
            generated = generated[0]
        generated = tuple(int(token) for token in generated)
        if on_token_chunk is not None and generated:
            # The SGLang Engine API returns a completed response here. A
            # streaming engine can replace this with incremental callbacks.
            on_token_chunk(generated)
        return generated


__all__ = ["QwenSGLangBackend", "QwenSGLangReadState"]
