"""Conversation state machine for delayed Sigma-memory reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

from .backend import SigmaBackend, SummaryBackend
from .memory import SigmaMemory

MemoryWriteMode = Literal["turn", "token_interval", "segment"]
CompactionMode = Literal["sigma", "summary", "hybrid", "recent_only"]


@dataclass(frozen=True)
class CompactionConfig:
    """How old context is represented after native KV compaction.

    ``sigma`` is the original design. ``summary`` and ``hybrid`` require a
    summary provider; the provider receives the old summary (if any) followed
    by newly evicted token ids and returns summary token ids. ``recent_only``
    is the no-memory ablation.
    """

    mode: CompactionMode = "sigma"
    summary_backend: SummaryBackend | None = None
    summary_max_tokens: int = 256
    write_sigma: bool | None = None

    def validate(self) -> None:
        if self.mode not in ("sigma", "summary", "hybrid", "recent_only"):
            raise ValueError("mode must be sigma, summary, hybrid, or recent_only")
        if self.summary_max_tokens < 1:
            raise ValueError("summary_max_tokens must be positive")
        if self.mode in ("summary", "hybrid") and self.summary_backend is None:
            raise ValueError(
                f"summary_backend is required in {self.mode} compaction mode"
            )

    @property
    def uses_sigma_read(self) -> bool:
        return self.mode in ("sigma", "hybrid")

    @property
    def writes_sigma_memory(self) -> bool:
        if self.write_sigma is not None:
            return self.write_sigma
        return self.mode in ("sigma", "hybrid")

    @property
    def uses_text_summary(self) -> bool:
        return self.mode in ("summary", "hybrid")


@dataclass(frozen=True)
class MemoryWriteConfig:
    """When completed conversation tokens become Sigma-memory writes."""

    mode: MemoryWriteMode = "turn"
    token_interval_tokens: int | None = None
    segment_capacity_tokens: int | None = None

    def validate(self) -> None:
        if self.mode not in ("turn", "token_interval", "segment"):
            raise ValueError("mode must be turn, token_interval, or segment")
        if self.mode == "token_interval" and (
            self.token_interval_tokens is None or self.token_interval_tokens < 1
        ):
            raise ValueError("token_interval_tokens must be positive in token_interval mode")
        if self.mode == "segment" and (
            self.segment_capacity_tokens is None or self.segment_capacity_tokens < 1
        ):
            raise ValueError("segment_capacity_tokens must be positive in segment mode")


@dataclass
class ConversationTurn:
    role: str
    token_ids: tuple[int, ...]
    memory: SigmaMemory | None
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
    summary_tokens: int = 0


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
    pending_turn_ids: list[str] = field(default_factory=list)
    pending_turn_tokens: dict[str, tuple[int, ...]] = field(default_factory=dict)
    summary_token_ids: tuple[int, ...] = ()


class SigmaConversationHarness:
    """Use native KV until compaction, then apply the configured representation.

    ``context_window_tokens`` is the native KV budget. ``compact_target_tokens``
    is the desired size of the recent native tail after a compact. The
    ``MemoryWriteConfig`` controls whether a turn is written at turn end, in
    generated-token intervals, or after a segment buffer reaches capacity.
    ``CompactionConfig`` selects Sigma state, text summary tokens, both, or a
    recent-only ablation. Before the first compact all Sigma writes are
    write-only and every backend call receives ``read_state=None``.
    """

    def __init__(
        self,
        backend: SigmaBackend,
        *,
        context_window_tokens: int,
        compact_target_tokens: int | None = None,
        reserve_tokens: int = 0,
        write_config: MemoryWriteConfig | None = None,
        compaction_config: CompactionConfig | None = None,
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
        self.write_config = write_config or MemoryWriteConfig()
        self.write_config.validate()
        self.compaction_config = compaction_config or CompactionConfig()
        self.compaction_config.validate()
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

    @property
    def pending_segment_tokens(self) -> int:
        return sum(len(self.state.pending_turn_tokens[turn_id]) for turn_id in self.state.pending_turn_ids)

    def _add_written(self, part: SigmaMemory) -> None:
        if self.state.written_memory is None:
            self.state.written_memory = part
        else:
            self.state.written_memory = self.state.written_memory.add(part)

    def _next_segment_id(self) -> str:
        return f"turn-{len(self.state.turns) + 1:06d}"

    def _materialize_turn(
        self,
        token_ids: tuple[int, ...],
        memory: SigmaMemory | None,
        *,
        segment_id: str | None = None,
    ) -> ConversationTurn:
        segment_id = segment_id or self._next_segment_id()
        turn_index = len(self.state.turns) + 1
        segment = None
        if memory is not None:
            segment = MemorySegment(
                segment_id=segment_id,
                turn_index=turn_index,
                token_count=len(token_ids),
                memory=memory,
                metadata={"kind": "conversation_turn"},
            )
        turn = ConversationTurn(
            role="turn",
            token_ids=token_ids,
            memory=memory,
            segment_id=segment_id,
            in_native_cache=True,
            readable=self.state.read_enabled,
        )
        self.state.turns.append(turn)
        if segment is not None:
            self.state.memory_segments.append(segment)
        self.state.write_version += 1
        if self.state.read_enabled and segment is not None:
            self.state.readable_segments.append(segment)
            self.state.readable_memory = self._sum_segments(self.state.readable_segments)
            self.state.memory_version += 1
        return turn

    def _queue_segment_turn(self, token_ids: tuple[int, ...]) -> None:
        segment_id = self._next_segment_id()
        turn = ConversationTurn(
            role="turn",
            token_ids=token_ids,
            memory=None,
            segment_id=segment_id,
            in_native_cache=True,
            readable=False,
        )
        self.state.turns.append(turn)
        self.state.pending_turn_ids.append(segment_id)
        self.state.pending_turn_tokens[segment_id] = token_ids

    def flush_pending_memory(self, *, rebuild: bool = True) -> int:
        """Write all buffered segment-mode turns and return their count."""

        if self.write_config.mode != "segment" or not self.state.pending_turn_ids:
            return 0
        pending = list(self.state.pending_turn_ids)
        for segment_id in pending:
            token_ids = self.state.pending_turn_tokens[segment_id]
            memory = None
            if self.compaction_config.writes_sigma_memory:
                memory = self.backend.write_memory(token_ids)
                self._add_written(memory)
            self._materialize_existing_pending_turn(segment_id, memory)
        self.state.pending_turn_ids.clear()
        self.state.pending_turn_tokens.clear()
        if rebuild and self.state.read_enabled:
            self._rebuild_native()
        return len(pending)

    def _materialize_existing_pending_turn(
        self,
        segment_id: str,
        memory: SigmaMemory | None,
    ) -> None:
        turn = next(turn for turn in self.state.turns if turn.segment_id == segment_id)
        turn.memory = memory
        turn.readable = self.state.read_enabled
        turn_index = self.state.turns.index(turn) + 1
        if memory is not None:
            segment = MemorySegment(
                segment_id=segment_id,
                turn_index=turn_index,
                token_count=len(turn.token_ids),
                memory=memory,
                metadata={"kind": "conversation_turn"},
            )
            self.state.memory_segments.append(segment)
        self.state.write_version += 1
        if self.state.read_enabled and memory is not None:
            self.state.readable_segments.append(segment)
            self.state.readable_memory = self._sum_segments(self.state.readable_segments)
            self.state.memory_version += 1

    @staticmethod
    def _sum_segments(segments: Sequence[MemorySegment]) -> SigmaMemory | None:
        memory = None
        for segment in segments:
            memory = segment.memory if memory is None else memory.add(segment.memory)
        return memory

    @staticmethod
    def _unique_segments(segments: Sequence[MemorySegment]) -> list[MemorySegment]:
        """Keep one immutable memory segment per conversation turn.

        A turn created after the first compact is already in
        ``readable_segments``. When that same turn is later evicted from the
        native tail, compact must not append it a second time.
        """

        unique: list[MemorySegment] = []
        seen: set[str] = set()
        for segment in segments:
            if segment.segment_id in seen:
                continue
            seen.add(segment.segment_id)
            unique.append(segment)
        return unique

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

    def _make_text_summary(
        self,
        evicted: Sequence[ConversationTurn],
        *,
        max_tokens: int,
    ) -> tuple[int, ...]:
        if not self.compaction_config.uses_text_summary:
            return ()
        backend = self.compaction_config.summary_backend
        if backend is None:  # validated in __init__, kept for type checkers
            raise RuntimeError("summary backend is not configured")
        source = self.state.summary_token_ids + tuple(
            token for turn in evicted for token in turn.token_ids
        )
        summary = backend.summarize(
            source,
            max_new_tokens=max_tokens,
        )
        return tuple(int(token) for token in summary)[:max_tokens]

    def compact(self) -> None:
        """Retire the oldest native turns and enable Sigma reads."""

        # A segment-mode buffer cannot be compacted until its raw turns have a
        # durable Sigma representation. This is the forced-write boundary.
        self.flush_pending_memory(rebuild=False)

        native = [turn for turn in self.state.turns if turn.in_native_cache]
        native_turn_tokens = sum(len(turn.token_ids) for turn in native)
        summary_budget = (
            min(self.compaction_config.summary_max_tokens, max(1, self.compact_target_tokens - 1))
            if self.compaction_config.uses_text_summary
            else 0
        )
        tail_budget = max(1, self.compact_target_tokens - summary_budget)
        if not native:
            raise RuntimeError("cannot compact an empty conversation")

        evicted: list[ConversationTurn] = []
        # Keep a recent native tail. Evict at least one turn whenever compact
        # is requested, so the state transition is observable.
        while native and (native_turn_tokens > tail_budget or not evicted):
            turn = native.pop(0)
            evicted.append(turn)
            native_turn_tokens -= len(turn.token_ids)

        candidate_summary = self._make_text_summary(
            evicted,
            max_tokens=summary_budget,
        )

        candidate_segments: list[MemorySegment] = []
        candidate_memory: SigmaMemory | None = None
        if self.compaction_config.uses_sigma_read:
            evicted_ids = {turn.segment_id for turn in evicted}
            evicted_segments = [
                segment
                for segment in self.state.memory_segments
                if segment.segment_id in evicted_ids
            ]
            candidate_segments = self._unique_segments(
                [*self.state.readable_segments, *evicted_segments]
            )
            candidate_memory = self._sum_segments(candidate_segments)
            if candidate_memory is None:
                raise RuntimeError("compact selected no memory segments")

        # Prepare both the reader state and the rebuilt native cache before
        # mutating session state. A failed solve/rebuild leaves the old KV and
        # the old read gate intact.
        candidate_version = self.state.memory_version + 1
        candidate_read_state = None
        candidate_snapshot = None
        if candidate_memory is not None:
            candidate_read_state = self.backend.prepare_read_state(candidate_memory)
            candidate_snapshot = ReaderSnapshot(
                memory_version=candidate_version,
                segment_ids=tuple(segment.segment_id for segment in candidate_segments),
                state=candidate_read_state,
            )
        candidate_native_ids = candidate_summary + tuple(
            token for turn in native for token in turn.token_ids
        )
        candidate_cache = self.backend.rebuild_native(
            candidate_native_ids,
            read_state=candidate_read_state,
        )

        for turn in evicted:
            turn.in_native_cache = False
            turn.readable = candidate_read_state is not None
        self.state.readable_segments = candidate_segments
        self.state.readable_memory = candidate_memory
        self.state.read_enabled = candidate_read_state is not None
        self.state.memory_version = candidate_version
        self.state.reader_snapshot = candidate_snapshot
        self.state.read_state = candidate_read_state
        self.state.summary_token_ids = candidate_summary
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
        end_of_dialogue: bool = False,
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

        mode = self.write_config.mode
        observed_generated: list[int] = []
        interval_buffer: list[int] = []
        written_parts: list[SigmaMemory] = []
        user_memory: SigmaMemory | None = None
        writes_sigma = self.compaction_config.writes_sigma_memory

        def write_interval_chunk(tokens: Sequence[int]) -> None:
            part = self.backend.write_memory(tokens)
            self._add_written(part)
            written_parts.append(part)

        def on_token_chunk(tokens: Sequence[int]) -> None:
            observed_generated.extend(int(token) for token in tokens)
            interval_buffer.extend(int(token) for token in tokens)
            interval = self.write_config.token_interval_tokens
            if interval is None:
                return
            while len(interval_buffer) >= interval:
                chunk = tuple(interval_buffer[:interval])
                del interval_buffer[:interval]
                write_interval_chunk(chunk)

        if mode == "token_interval" and writes_sigma:
            user_memory = self.backend.write_memory(prompt)
            self._add_written(user_memory)

        callback: Callable[[Sequence[int]], None] | None = (
            on_token_chunk if mode == "token_interval" and writes_sigma else None
        )
        generated = tuple(
            int(token)
            for token in self.backend.generate(
                self.state.native_cache,
                read_state=self.state.read_state,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                on_token_chunk=callback,
            )
        )
        if mode == "token_interval" and writes_sigma:
            # Reference backends may not stream yet. Process any suffix that
            # was returned without a callback, preserving the same schedule.
            if len(observed_generated) < len(generated):
                on_token_chunk(generated[len(observed_generated) :])
            if tuple(observed_generated) != generated:
                raise RuntimeError("backend token callback did not match generated output")
            if interval_buffer:
                write_interval_chunk(tuple(interval_buffer))
                interval_buffer.clear()
        self.state.native_token_ids += generated
        self._record_event("generate")

        full_turn_tokens = prompt + generated
        if mode == "turn":
            # Keep a complete user/assistant exchange as one eviction unit.
            turn_memory = None
            if writes_sigma:
                user_memory = self.backend.write_memory(prompt)
                assistant_memory = self.backend.write_memory(generated)
                self._add_written(user_memory)
                self._add_written(assistant_memory)
                turn_memory = user_memory.add(assistant_memory)
            self._materialize_turn(full_turn_tokens, turn_memory)
        elif mode == "token_interval":
            turn_memory = user_memory
            for part in written_parts:
                turn_memory = part if turn_memory is None else turn_memory.add(part)
            if turn_memory is None and writes_sigma:
                turn_memory = self.backend.write_memory(())
                self._add_written(turn_memory)
            self._materialize_turn(full_turn_tokens, turn_memory)
        else:
            self._queue_segment_turn(full_turn_tokens)
            if self.pending_segment_tokens >= (self.write_config.segment_capacity_tokens or 0):
                self.flush_pending_memory(rebuild=False)

        if end_of_dialogue:
            self.finish()

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
            summary_tokens=len(self.state.summary_token_ids),
        )

    def finish(self) -> None:
        """Force the configured mode to flush any buffered dialogue memory."""

        if self.write_config.mode == "segment":
            self.flush_pending_memory(rebuild=self.state.read_enabled)


__all__ = [
    "CompactionConfig",
    "CompactionMode",
    "ConversationTurn",
    "GenerationResult",
    "HarnessEvent",
    "HarnessState",
    "MemorySegment",
    "MemoryWriteConfig",
    "MemoryWriteMode",
    "ReaderSnapshot",
    "SigmaConversationHarness",
]
