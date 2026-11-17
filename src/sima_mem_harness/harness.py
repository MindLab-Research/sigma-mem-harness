"""Conversation state machine for delayed Sigma-memory reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .backend import SigmaBackend
from .memory import SigmaMemory


@dataclass
class ConversationTurn:
    role: str
    token_ids: tuple[int, ...]
    memory: SigmaMemory
    segment_id: str = ""
    in_native_cache: bool = True
    readable: bool = False


@dataclass(frozen=True)
class MemorySegment:
    """One independently writable, persistable long-term-memory unit."""

    segment_id: str
    turn_index: int
    token_count: int
    memory: SigmaMemory
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReaderSnapshot:
    """Immutable reader state for one readable-memory version."""

    memory_version: int
    segment_ids: tuple[str, ...]
    state: Any


@dataclass(frozen=True)
class HarnessEvent:
    operation: str
    read_enabled: bool
    native_tokens: int
    readable_tokens: float
    compact_count: int
    memory_version: int
    reader_snapshot_version: int | None
    read_state_recomputed: bool = False


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    compacted: bool
    read_enabled: bool
    native_tokens: int
    memory_version: int


@dataclass
class HarnessState:
    turns: list[ConversationTurn] = field(default_factory=list)
    native_token_ids: tuple[int, ...] = ()
    native_cache: Any | None = None
    written_memory: SigmaMemory | None = None
    readable_memory: SigmaMemory | None = None
    memory_segments: list[MemorySegment] = field(default_factory=list)
    readable_segments: list[MemorySegment] = field(default_factory=list)
    read_state: Any | None = None
    reader_snapshot: ReaderSnapshot | None = None
    write_version: int = 0
    memory_version: int = 0
    compact_count: int = 0
    read_enabled: bool = False


class SigmaConversationHarness:
    """Use native KV until compaction, then enable Sigma reads.

    ``context_window_tokens`` is the native KV budget. ``compact_target_tokens``
    is the desired size of the recent native tail after a compact. Every
    completed turn is written exactly once. Before the first compact it only
    contributes to ``written_memory``; all backend calls receive
    ``read_state=None``. After compact, newly completed turns are added to
    ``readable_memory`` and the backend receives a fresh read state on the next
    turn.
    """

    def __init__(
        self,
        backend: SigmaBackend,
        *,
        context_window_tokens: int,
        compact_target_tokens: int | None = None,
        reserve_tokens: int = 0,
    ) -> None:
        if context_window_tokens < 1:
            raise ValueError("context_window_tokens must be positive")
        if reserve_tokens < 0 or reserve_tokens >= context_window_tokens:
            raise ValueError("reserve_tokens must be in [0, context_window_tokens)")
        target = compact_target_tokens
        if target is None:
            target = max(1, context_window_tokens // 2)
        if target < 1 or target >= context_window_tokens:
            raise ValueError("compact_target_tokens must be in [1, context_window_tokens)")
        self.backend = backend
        self.context_window_tokens = context_window_tokens
        self.compact_target_tokens = target
        self.reserve_tokens = reserve_tokens
        self.state = HarnessState()
        self.events: list[HarnessEvent] = []

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self.state.turns)

    @property
    def sigma_written_tokens(self) -> float:
        memory = self.state.written_memory
        return 0.0 if memory is None else float(memory.count.item())

    @property
    def sigma_readable_tokens(self) -> float:
        memory = self.state.readable_memory
        return 0.0 if memory is None else float(memory.count.item())

    def _add_written(self, part: SigmaMemory) -> None:
        if self.state.written_memory is None:
            self.state.written_memory = part
        else:
            self.state.written_memory = self.state.written_memory.add(part)

    @staticmethod
    def _sum_segments(segments: Sequence[MemorySegment]) -> SigmaMemory | None:
        memory = None
        for segment in segments:
            memory = segment.memory if memory is None else memory.add(segment.memory)
        return memory

    def _record_event(self, operation: str, *, read_state_recomputed: bool = False) -> None:
        snapshot_version = None
        if self.state.reader_snapshot is not None:
            snapshot_version = self.state.reader_snapshot.memory_version
        self.events.append(
            HarnessEvent(
                operation=operation,
                read_enabled=self.state.read_enabled,
                native_tokens=len(self.state.native_token_ids),
                readable_tokens=self.sigma_readable_tokens,
                compact_count=self.state.compact_count,
                memory_version=self.state.memory_version,
                reader_snapshot_version=snapshot_version,
                read_state_recomputed=read_state_recomputed,
            )
        )

    def _set_read_state(self) -> bool:
        if not self.state.read_enabled or self.state.readable_memory is None:
            self.state.read_state = None
            self.state.reader_snapshot = None
            return False
        snapshot = self.state.reader_snapshot
        if snapshot is not None and snapshot.memory_version == self.state.memory_version:
            self.state.read_state = snapshot.state
            return False
        read_state = self.backend.prepare_read_state(self.state.readable_memory)
        self.state.reader_snapshot = ReaderSnapshot(
            memory_version=self.state.memory_version,
            segment_ids=tuple(segment.segment_id for segment in self.state.readable_segments),
            state=read_state,
        )
        self.state.read_state = read_state
        return True

    def _rebuild_native(self) -> None:
        recomputed = self._set_read_state()
        self.state.native_cache = self.backend.rebuild_native(
            self.state.native_token_ids,
            read_state=self.state.read_state,
        )
        self._record_event("rebuild_native", read_state_recomputed=recomputed)

    def compact(self) -> None:
        """Retire the oldest native turns and enable Sigma reads."""

        native = [turn for turn in self.state.turns if turn.in_native_cache]
        native_tokens = sum(len(turn.token_ids) for turn in native)
        if not native:
            raise RuntimeError("cannot compact an empty conversation")

        evicted: list[ConversationTurn] = []
        # Keep a recent native tail. Evict at least one turn whenever compact
        # is requested, so the state transition is observable.
        while native and (native_tokens > self.compact_target_tokens or not evicted):
            turn = native.pop(0)
            evicted.append(turn)
            native_tokens -= len(turn.token_ids)

        evicted_ids = {turn.segment_id for turn in evicted}
        evicted_segments = [
            segment for segment in self.state.memory_segments if segment.segment_id in evicted_ids
        ]
        candidate_segments = list(self.state.readable_segments) + evicted_segments
        candidate_memory = self._sum_segments(candidate_segments)
        if candidate_memory is None:
            raise RuntimeError("compact selected no memory segments")

        # Prepare both the reader state and the rebuilt native cache before
        # mutating session state. A failed solve/rebuild leaves the old KV and
        # the old read gate intact.
        candidate_version = self.state.memory_version + 1
        candidate_read_state = self.backend.prepare_read_state(candidate_memory)
        candidate_snapshot = ReaderSnapshot(
            memory_version=candidate_version,
            segment_ids=tuple(segment.segment_id for segment in candidate_segments),
            state=candidate_read_state,
        )
        candidate_native_ids = tuple(token for turn in native for token in turn.token_ids)
        candidate_cache = self.backend.rebuild_native(
            candidate_native_ids,
            read_state=candidate_read_state,
        )

        for turn in evicted:
            turn.in_native_cache = False
            turn.readable = True
        self.state.readable_segments = candidate_segments
        self.state.readable_memory = candidate_memory
        self.state.read_enabled = True
        self.state.memory_version = candidate_version
        self.state.reader_snapshot = candidate_snapshot
        self.state.read_state = candidate_read_state
        self.state.compact_count += 1
        self.state.native_token_ids = candidate_native_ids
        self.state.native_cache = candidate_cache
        self._record_event("rebuild_native", read_state_recomputed=True)
        self._record_event("compact")

    def _ensure_capacity(self, prompt_tokens: int, max_new_tokens: int) -> bool:
        if prompt_tokens < 1:
            raise ValueError("a user turn must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        response_budget = max(max_new_tokens, self.reserve_tokens)
        required = len(self.state.native_token_ids) + prompt_tokens + response_budget
        if required <= self.context_window_tokens:
            return False
        if prompt_tokens + response_budget > self.context_window_tokens:
            raise ValueError("incoming prompt plus response exceeds context window")
        # Pick a tail size that leaves enough room for this prompt and answer.
        self.compact_target_tokens = min(
            self.compact_target_tokens,
            self.context_window_tokens - prompt_tokens - response_budget,
        )
        self.compact()
        required = len(self.state.native_token_ids) + prompt_tokens + response_budget
        if required > self.context_window_tokens:
            raise ValueError("compaction could not make room for the incoming prompt")
        return True

    def run_turn(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int = 32,
        eos_token_id: int | None = None,
    ) -> GenerationResult:
        """Run one user/assistant turn under the delayed-read policy."""

        prompt = tuple(int(token) for token in prompt_token_ids)
        compacted = self._ensure_capacity(len(prompt), max_new_tokens)

        # No compact: read_state is guaranteed to remain None. After compact,
        # the state is rebuilt with the readable Sigma statistic.
        self._set_read_state()
        self.state.native_cache = self.backend.append_native(
            self.state.native_cache,
            prompt,
            read_state=self.state.read_state,
        )
        self.state.native_token_ids += prompt
        self._record_event("append_user")

        generated = tuple(
            int(token)
            for token in self.backend.generate(
                self.state.native_cache,
                read_state=self.state.read_state,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
        )
        self.state.native_token_ids += generated
        self._record_event("generate")

        # Keep a complete user/assistant exchange as one eviction unit. The
        # two messages are still encoded independently, matching the writer's
        # per-message/window contract, then combined additively.
        user_memory = self.backend.write_memory(prompt)
        assistant_memory = self.backend.write_memory(generated)
        turn_memory = user_memory.add(assistant_memory)
        turn_index = len(self.state.turns) + 1
        segment_id = f"turn-{turn_index:06d}"
        segment = MemorySegment(
            segment_id=segment_id,
            turn_index=turn_index,
            token_count=len(prompt) + len(generated),
            memory=turn_memory,
            metadata={"kind": "conversation_turn"},
        )
        turn = ConversationTurn(
            role="turn",
            token_ids=prompt + generated,
            memory=turn_memory,
            segment_id=segment_id,
            in_native_cache=True,
            readable=self.state.read_enabled,
        )
        self.state.turns.append(turn)
        self.state.memory_segments.append(segment)
        self.state.write_version += 1
        self._add_written(turn_memory)
        if self.state.read_enabled:
            self.state.readable_segments.append(segment)
            self.state.readable_memory = self._sum_segments(self.state.readable_segments)
            self.state.memory_version += 1

        # New writes become readable on the next turn. Rebuilding here keeps
        # the cache/read-state contract explicit for a real backend; the toy
        # and reference adapters make this operation cheap.
        if self.state.read_enabled:
            self._rebuild_native()

        return GenerationResult(
            token_ids=generated,
            compacted=compacted,
            read_enabled=self.state.read_enabled,
            native_tokens=len(self.state.native_token_ids),
            memory_version=self.state.memory_version,
        )


__all__ = [
    "ConversationTurn",
    "GenerationResult",
    "HarnessEvent",
    "HarnessState",
    "MemorySegment",
    "ReaderSnapshot",
    "SigmaConversationHarness",
]
