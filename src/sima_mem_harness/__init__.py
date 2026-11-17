"""Minimal Sigma-memory conversation harness.

The harness deliberately keeps two paths separate:

* before the first context compaction, the backend receives ``read_state=None``
  and uses only its native KV cache;
* compaction retires old turns into Sigma memory, rebuilds the native cache
  from the recent tail, and enables the Sigma read path for later turns.
"""

from .backend import LogicalKVCache, SigmaBackend
from .harness import (
    ConversationTurn,
    GenerationResult,
    HarnessEvent,
    HarnessState,
    MemorySegment,
    MemoryWriteConfig,
    MemoryWriteMode,
    ReaderSnapshot,
    SigmaConversationHarness,
)
from .memory import SigmaMemory

__all__ = [
    "ConversationTurn",
    "GenerationResult",
    "HarnessEvent",
    "HarnessState",
    "LogicalKVCache",
    "MemorySegment",
    "MemoryWriteConfig",
    "MemoryWriteMode",
    "ReaderSnapshot",
    "SigmaBackend",
    "SigmaConversationHarness",
    "SigmaMemory",
]
