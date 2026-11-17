# sima-mem-harness

A minimal multi-turn conversation harness for the following Sigma-memory
protocol: **use only the model's native KV cache before compaction, and enable
Sigma-memory reads only after compaction**.

This repository provides conversation orchestration. It does not replace the
DeepSeek-V4.1 training or serving backend. It follows the Sigma-memory design
used by the project: the writer produces an additive second-moment statistic
`H`, compacted history is moved into readable memory, and the backend converts
that memory into the model's per-layer read state.

## Protocol

User/assistant history is written to Sigma memory according to the configured
write schedule:

```text
turn -> writer -> written_memory
```

Before the first compact, `written_memory` is write-only. Every backend call
receives `read_state=None`:

```text
recent conversation -> native KV -> model
history             -> Sigma writer only
```

When a new prompt plus its expected response would exceed the native context
window, the harness:

1. Evicts the oldest complete conversation turns while retaining a recent
   token tail.
2. Adds the evicted turns to `readable_memory`.
3. Rebuilds the native KV cache for the retained tail.
4. Starts passing `read_state` to the backend, enabling Sigma-memory reads.

Turns completed after compaction are also written to Sigma memory and become
readable on the next turn. `written_memory` and `readable_memory` are kept
separate so recent turns still present in native KV are not read twice.

The session state now keeps complete-turn `MemorySegment` objects, a
`memory_version`, and an immutable `ReaderSnapshot`. Reader states are reused
while the readable memory version is unchanged. Compaction prepares the new
reader state and rebuilt native cache before committing any eviction, so a
failed solve or rebuild leaves the previous KV and read gate intact.

## Write timing modes

The write schedule is configurable through `MemoryWriteConfig`:

```python
from sima_mem_harness import MemoryWriteConfig, SigmaConversationHarness

# Existing behavior: write user and assistant after every completed turn.
turn_end = MemoryWriteConfig(mode="turn")

# Write assistant prefixes every 128 generated tokens, then flush the remainder.
token_interval = MemoryWriteConfig(mode="token_interval", token_interval_tokens=128)

# Buffer completed turns until their combined token count reaches 4096.
segment_buffer = MemoryWriteConfig(mode="segment", segment_capacity_tokens=4096)

harness = SigmaConversationHarness(
    backend,
    context_window_tokens=8192,
    write_config=token_interval,
)
```

All three modes keep the same read gate: memory written before the first
compact remains write-only. `token_interval` requires a backend that invokes
`on_token_chunk` during decoding for true mid-generation writes; the reference
DeepSeek adapter currently invokes that callback after its non-streaming
reference generation completes. `segment` mode flushes its pending turns when
the configured capacity is reached, when `finish()` is called, or immediately
before compact.

## Local usage

```bash
python -m pip install -e '.[test]'
pytest -q
python -m sima_mem_harness.smoke
```

`TraceBackend` is a deterministic protocol-test backend. It does not pretend
to implement real KV caching; it records whether each append, rebuild, and
generation call received a Sigma read state.

## Connecting the current DeepSeek-V4.1 implementation

The current `delta_mem_miles/sigma/dpsk-v4.1` implementation is still a
reference generation path and does not expose incremental paged KV yet.
`SigmaV41ReferenceBackend` can wrap an already-constructed `SigmaV41` model:

```python
from sima_mem_harness import SigmaConversationHarness
from sima_mem_harness.v41 import SigmaV41ReferenceBackend

backend = SigmaV41ReferenceBackend(model)
harness = SigmaConversationHarness(
    backend,
    context_window_tokens=8192,
    compact_target_tokens=4096,
    reserve_tokens=512,
)
answer_ids = harness.run_turn(question_ids, max_new_tokens=512).token_ids
```

The reference adapter still recomputes the read sequence through
`SigmaV41.generate`. When a real vLLM, SGLang, or DeepSeek paged-KV backend is
ready, it only needs to implement the five methods in `SigmaBackend`; the
harness compaction/read gate remains unchanged.

## Validation status

The harness has been exercised end to end with the project's `SigmaV41`
implementation and its V4.1 tiny architecture fixture:

```text
first turn:  compacted=False, read_enabled=False
second turn: compacted=True, read_enabled=True
```

This confirms the complete writer, compaction, reader-state, and generation
path. The smoke uses the shared DeepSeek-style encoder as the writer; it does
not use a Qwen3-0.6B writer.

The full DeepSeek-V4.1 checkpoint has not been run through this new harness
yet. The TP8 checkpoint is available in the project environment, but the
server GPUs were occupied by the existing training job during this validation.
The harness therefore does not claim a real-checkpoint result until that run
is completed. The original Qwen setup (Qwen3-0.6B writer and Qwen3-4B reader)
was validated separately in `delta-mem-miles`, outside this harness.

## Future extensions

- Replace `LogicalKVCache` with a real paged-KV object.
- Connect compaction to the model's token eviction and block manager.
- Cache each layer's `S_l` in `prepare_read_state` to avoid repeated solves
  during decoding.
- Add full-history, no-memory, and compact-memory comparison evaluations.

See [EVALUATION.md](EVALUATION.md) for the harness comparison protocol,
quality suites, system metrics, and long-context matrix.
