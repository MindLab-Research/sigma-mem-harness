"""Backend protocol for native KV plus Sigma-memory reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .memory import SigmaMemory


@dataclass(frozen=True)
class LogicalKVCache:
    """A tiny cache used by the reference adapter and tests."""

    token_ids: tuple[int, ...]


class SigmaBackend(Protocol):
    """Minimal model contract consumed by ``SigmaConversationHarness``."""

    def write_memory(self, token_ids: Sequence[int]) -> SigmaMemory:
        """Encode a completed turn without reading Sigma memory."""

    def prepare_read_state(self, memory: SigmaMemory) -> Any:
        """Convert readable memory into the model's per-layer read state."""

    def append_native(
        self,
        cache: Any | None,
        token_ids: Sequence[int],
        *,
        read_state: Any | None,
    ) -> Any:
        """Append tokens to native KV; None means Sigma reads are disabled."""

    def rebuild_native(self, token_ids: Sequence[int], *, read_state: Any | None) -> Any:
        """Rebuild native KV for the retained tail after compact."""

    def generate(
        self,
        cache: Any,
        *,
        read_state: Any | None,
        max_new_tokens: int,
        eos_token_id: int | None,
        on_token_chunk: Callable[[Sequence[int]], None] | None = None,
    ) -> Sequence[int]:
        """Generate from native KV and optionally stream generated chunks."""


__all__ = ["LogicalKVCache", "SigmaBackend"]
