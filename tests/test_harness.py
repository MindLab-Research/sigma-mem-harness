import pytest

from sima_mem_harness import MemoryWriteConfig, SigmaConversationHarness
from sima_mem_harness.toy import TraceBackend


def test_before_compact_sigma_only_writes_and_never_reads():
    backend = TraceBackend()
    harness = SigmaConversationHarness(backend, context_window_tokens=32, compact_target_tokens=12)

    first = harness.run_turn([1, 2, 3], max_new_tokens=2)
    second = harness.run_turn([4, 5], max_new_tokens=2)

    assert first.compacted is False and second.compacted is False
    assert harness.state.read_enabled is False
    assert harness.sigma_written_tokens > 0
    assert harness.sigma_readable_tokens == 0
    assert all(not call.read_enabled for call in backend.calls)


def test_compact_moves_old_turns_to_sigma_and_enables_reads():
    backend = TraceBackend()
    harness = SigmaConversationHarness(backend, context_window_tokens=7, compact_target_tokens=5)

    harness.run_turn([1, 2, 3], max_new_tokens=1)
    result = harness.run_turn([4, 5, 6], max_new_tokens=1)

    assert result.compacted is True
    assert result.read_enabled is True
    assert harness.state.compact_count == 1
    assert harness.sigma_written_tokens >= harness.sigma_readable_tokens > 0
    assert any(call.operation == "generate" and call.read_enabled for call in backend.calls)
    assert any(turn.in_native_cache for turn in harness.turns)
    assert any(not turn.in_native_cache for turn in harness.turns)


def test_post_compact_turns_are_written_and_read_on_the_next_turn():
    backend = TraceBackend()
    harness = SigmaConversationHarness(backend, context_window_tokens=7, compact_target_tokens=4)

    harness.run_turn([1, 2, 3], max_new_tokens=1)
    harness.run_turn([4, 5, 6], max_new_tokens=1)
    readable_before = harness.sigma_readable_tokens
    harness.run_turn([7], max_new_tokens=1)

    assert harness.sigma_readable_tokens > readable_before
    assert harness.state.read_enabled is True
    assert backend.calls[-1].operation == "rebuild_native"
    assert backend.calls[-1].read_enabled is True


def test_compact_commits_segments_and_reader_snapshot_atomically():
    backend = TraceBackend()
    harness = SigmaConversationHarness(backend, context_window_tokens=7, compact_target_tokens=5)

    harness.run_turn([1, 2, 3], max_new_tokens=1)
    result = harness.run_turn([4, 5, 6], max_new_tokens=1)

    assert result.memory_version == 2
    assert [segment.segment_id for segment in harness.state.memory_segments] == [
        "turn-000001",
        "turn-000002",
    ]
    assert [segment.segment_id for segment in harness.state.readable_segments] == [
        "turn-000001",
        "turn-000002",
    ]
    assert harness.state.reader_snapshot is not None
    assert harness.state.reader_snapshot.memory_version == harness.state.memory_version
    assert harness.state.reader_snapshot.segment_ids == ("turn-000001", "turn-000002")
    assert backend.prepare_calls == 2


def test_compact_failure_keeps_native_state_and_read_gate_unchanged():
    backend = TraceBackend()
    harness = SigmaConversationHarness(backend, context_window_tokens=7, compact_target_tokens=5)
    harness.run_turn([1, 2, 3], max_new_tokens=1)
    previous_cache = harness.state.native_cache
    backend.fail_next_prepare = True

    with pytest.raises(RuntimeError, match="injected reader-state"):
        harness.run_turn([4, 5, 6], max_new_tokens=1)

    assert harness.state.compact_count == 0
    assert harness.state.read_enabled is False
    assert harness.state.memory_version == 0
    assert harness.state.native_cache == previous_cache
    assert len(harness.turns) == 1
    assert harness.turns[0].in_native_cache is True


def test_prompt_that_cannot_fit_is_rejected_before_model_calls():
    backend = TraceBackend()
    harness = SigmaConversationHarness(backend, context_window_tokens=8)
    try:
        harness.run_turn([1] * 7, max_new_tokens=2)
    except ValueError as exc:
        assert "exceeds context window" in str(exc)
    else:
        raise AssertionError("oversized prompt should be rejected")
    assert backend.calls == []


def test_token_interval_mode_writes_during_generation_and_forces_remainder():
    backend = TraceBackend()
    harness = SigmaConversationHarness(
        backend,
        context_window_tokens=64,
        write_config=MemoryWriteConfig(mode="token_interval", token_interval_tokens=2),
    )

    result = harness.run_turn([1, 2], max_new_tokens=5, end_of_dialogue=True)

    assert result.compacted is False
    assert harness.sigma_readable_tokens == 0
    assert backend.write_calls == [(1, 2), (900, 901), (902, 903), (904,)]
    assert harness.sigma_written_tokens == 7
    assert all(not call.read_enabled for call in backend.calls)


def test_segment_mode_buffers_turns_until_capacity_then_flushes():
    backend = TraceBackend()
    harness = SigmaConversationHarness(
        backend,
        context_window_tokens=64,
        write_config=MemoryWriteConfig(mode="segment", segment_capacity_tokens=6),
    )

    harness.run_turn([1, 2], max_new_tokens=1)
    assert backend.write_calls == []
    assert harness.pending_segment_tokens == 3
    harness.run_turn([3, 4], max_new_tokens=1)

    assert harness.pending_segment_tokens == 0
    assert len(backend.write_calls) == 2
    assert harness.sigma_written_tokens == 6
    assert harness.sigma_readable_tokens == 0


def test_segment_mode_forces_pending_writes_before_compact():
    backend = TraceBackend()
    harness = SigmaConversationHarness(
        backend,
        context_window_tokens=7,
        compact_target_tokens=5,
        write_config=MemoryWriteConfig(mode="segment", segment_capacity_tokens=100),
    )

    harness.run_turn([1, 2, 3], max_new_tokens=1)
    assert backend.write_calls == []
    result = harness.run_turn([4, 5, 6], max_new_tokens=1)

    assert result.compacted is True
    assert result.read_enabled is True
    assert len(backend.write_calls) == 1
    assert harness.sigma_readable_tokens == 4
