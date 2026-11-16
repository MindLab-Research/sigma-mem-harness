"""Conversation state machine for delayed Sigma-memory reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .backend import SigmaBackend
from .memory import SigmaMemory


@dataclass
class ConversationTurn:
    role: str
    token_ids: tuple[int, ...]
    memory: SigmaMemory
    in_native_cache: bool = True
    readable: bool = False


@dataclass(frozen=True)
class HarnessEvent:
    operation: str
    read_enabled: bool
    native_tokens: int
    readable_tokens: float
    compact_count: int


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    compacted: bool
    read_enabled: bool
    native_tokens: int


@dataclass
class HarnessState:
    turns: list[ConversationTurn] = field(default_factory=list)
    native_token_ids: tuple[int, ...] = ()
    native_cache: Any | None = None
    written_memory: SigmaMemory | None = None
    readable_memory: SigmaMemory | None = None
    read_state: Any | None = None
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

    def _add_readable(self, part: SigmaMemory) -> None:
        if self.state.readable_memory is None:
            self.state.readable_memory = part
        else:
            self.state.readable_memory = self.state.readable_memory.add(part)

    def _record_event(self, operation: str) -> None:
        self.events.append(
            HarnessEvent(
                operation=operation,
                read_enabled=self.state.read_enabled,
                native_tokens=len(self.state.native_token_ids),
                readable_tokens=self.sigma_readable_tokens,
                compact_count=self.state.compact_count,
            )
        )

    def _set_read_state(self) -> None:
        if not self.state.read_enabled or self.state.readable_memory is None:
            self.state.read_state = None
            return
        self.state.read_state = self.backend.prepare_read_state(self.state.readable_memory)

    def _rebuild_native(self) -> None:
        self._set_read_state()
        self.state.native_cache = self.backend.rebuild_native(
            self.state.native_token_ids,
            read_state=self.state.read_state,
        )
        self._record_event("rebuild_native")

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
            turn.in_native_cache = False
            evicted.append(turn)
            native_tokens -= len(turn.token_ids)

        for turn in evicted:
            if not turn.readable:
                self._add_readable(turn.memory)
                turn.readable = True

        self.state.read_enabled = True
        self.state.compact_count += 1
        self.state.native_token_ids = tuple(token for turn in native for token in turn.token_ids)
        self._rebuild_native()
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

        for role, token_ids in (("user", prompt), ("assistant", generated)):
            part = self.backend.write_memory(token_ids)
            turn = ConversationTurn(
                role=role,
                token_ids=token_ids,
                memory=part,
                in_native_cache=True,
                readable=self.state.read_enabled,
            )
            self.state.turns.append(turn)
            self._add_written(part)
            if self.state.read_enabled:
                self._add_readable(part)

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
        )


__all__ = [
    "ConversationTurn",
    "GenerationResult",
    "HarnessEvent",
    "HarnessState",
    "SigmaConversationHarness",
]
