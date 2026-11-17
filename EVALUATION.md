# Harness evaluation plan

The harness should be evaluated as a two-tier memory system: native KV for
recent, position-sensitive context and Sigma memory for compacted history. A
single final F1 number is not enough because a policy can improve recall while
adding unacceptable write or compaction latency.

## Policies to compare

Keep the model, reader adapter, tokenizer, training checkpoint, decoding
parameters, native-KV budget, and question set fixed. Compare only the harness
policy:

| Policy | Native KV | Sigma memory |
| --- | --- | --- |
| `native_only` | Recent window | Disabled |
| `global_sigma` | Recent tail after compact | One H over all retired turns |
| `segmented_sigma` | Recent tail after compact | One H per completed turn |
| `topk_segment_sigma` | Recent tail after compact | Select segments, then sum H |
| `decayed_segment_sigma` | Recent tail after compact | Segment selection plus time decay |

The writer is a separate factor. Run DeepSeek shared-encoder writer and
Qwen3-0.6B writer as independent ablations instead of changing them together
with the harness policy.

Write timing is another independent axis:

| Schedule | Parameter | Question |
| --- | --- | --- |
| turn end | none | Is delayed writing sufficient? |
| token interval | `n` generated tokens | Does partial writing reduce burst cost or improve durability? |
| segment buffer | capacity in tokens | What batching size gives the best write/latency tradeoff? |

All schedules must use the same first-compact read gate. Before the first
compact, interval and segment writes are write-only and cannot affect logits.

## Correctness gates

Every policy must pass these checks before quality measurements:

- pending memory never changes logits before the first compact;
- compact moves complete user/assistant turns atomically;
- a failed reader solve or KV rebuild leaves the old KV and read gate intact;
- a new turn is not readable by the generation that created it;
- the next turn sees the new memory version;
- adding segments in different orders gives the same additive H;
- repeated compact does not duplicate a segment;
- reader states are reused while the memory version is unchanged;
- restart preserves segment IDs, memory versions, and reader snapshots.

## Quality suites

Use controlled synthetic conversations first, then the existing LoCoMo/Metis
protocols.

1. **Recent recall:** the answer is in the retained native KV tail.
2. **Evicted recall:** the answer is only in a compacted segment.
3. **Distractor recall:** unrelated segments are added to measure interference.
4. **Updates and time:** facts are revised across turns; measure latest-fact and
   temporal-order accuracy.
5. **Multi-hop:** separate segments contain facts that must be combined.
6. **Negative controls:** shuffled memory, no memory, and irrelevant memory.

Report answer F1/EM together with:

```text
evicted-memory accuracy
recent-memory accuracy
correct-memory minus shuffled-memory
correct-memory minus no-memory
temporal accuracy
multi-hop accuracy
```

## Systems measurements

Log one record per turn and compact event:

```text
session_id, turn_id, input_tokens, output_tokens
native_tokens_before, native_tokens_after, compacted
evicted_segment_ids, write_ms, compact_ms, reader_solve_ms
prefill_ms, decode_ms, native_kv_bytes, H_bytes, S_bytes
gpu_peak_memory, cpu_memory, disk_bytes, memory_version
```

Report TTFT, decode tokens/sec, write latency, compact pause, reader-solve
latency, peak GPU memory, CPU/NVMe memory, compact frequency, and amortized
cost per turn. Plot quality against end-to-end latency and memory, and select
the Pareto frontier rather than the highest quality alone.

For write-schedule experiments, also report the number of writer calls per
turn, average and P95 write batch size, foreground generation time, background
write time, and the amount of unwritten data at a crash or forced `finish()`.

## Long-context matrix

Run the same conversations at 8K, 16K, 32K, 64K, 128K, 256K, and 1M history
tokens. Use native-KV budgets such as 4K, 8K, and 16K. Include online writes
that reuse Prefill activations, offline document writes, and restart-from-
persistent-memory runs.

A useful first matrix is:

```text
3 native-KV budgets × 4 harness policies × 3 history lengths × 3 seeds
```

The primary comparison is not whether Sigma memory replaces native KV. It is
whether a fixed recent KV tail plus compacted Sigma memory gives a better
quality/latency/memory tradeoff than a shorter native window or full-history
replay.
