# Harness evaluation plan

The harness should be evaluated as a two-tier memory system: native KV for
recent, position-sensitive context and Sigma memory for compacted history. A
single final F1 number is not enough because a policy can improve recall while
adding unacceptable write or compaction latency.

## Benchmark selection

The benchmark must be replayable as an online stream. A dataset that only gives
one long prompt does not exercise this harness: the harness needs complete
user/assistant turns, a native-KV budget smaller than the accumulated history,
and a later query whose evidence may have been evicted. Every external dataset
therefore needs a replay adapter that feeds its sessions or trajectories one
turn at a time and records whether the evidence turn was compacted.

### Recommended benchmark stack

| Priority | Benchmark | What it measures | Fit for this harness | Recommended use |
| --- | --- | --- | --- | --- |
| P0 | **Sigma-Harness Gate Suite** (new) | Exact read/write gates, eviction positions, updates, multi-hop facts, distractors, and abstention | Direct; full control over token lengths and compact timing | Required protocol and regression suite |
| P1 | [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) ([paper](https://arxiv.org/abs/2507.05257)) | Incremental multi-turn memory with accurate retrieval, test-time learning, long-range understanding, and conflict resolution | Very strong; the data is explicitly split into interaction chunks and follows an “inject once, query multiple times” design | First external benchmark for write timing and post-compact recall |
| P1 | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) ([cleaned data](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)) | 500 questions covering information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention | Strong for the text-only Qwen harness; sessions have timestamps and evidence-turn/session labels | Main conversational-memory benchmark |
| P1 | [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) ([paper](https://arxiv.org/abs/2605.12493)) | Agentic web/enterprise trajectories, static and dynamic state, workflow knowledge, environment gotchas, and premise awareness | Best modern long-history benchmark, but needs trajectory replay and multimodal support for screenshot questions | Second-stage agentic and ultra-long evaluation |
| P2 | [GoodAI LTM Benchmark](https://github.com/GoodAI/goodai-ltm-benchmark) | Dynamic memory upkeep and long-span needle tasks with configurable 2K–500K memory spans | Good stress test; older and partly generated, so weaker as a natural conversation measure | Context-length and compact-frequency stress |
| P2 | [InMind](https://github.com/imlrz/InMind) | Direct recall versus semantically indirect application of an earlier user fact | Good diagnostic for whether the reader can use memory when query/history lexical overlap is low | Reader and query-matching ablation |
| P2 | [Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) | LoCoMo-style multi-hop, temporal, common-sense, single-hop, adversarial, plus a cognitive implicit-recall category | Useful natural-dialogue diagnostic, but not the primary long-context benchmark | Secondary cognitive-recall check |
| P3 | [BEAM](https://arxiv.org/abs/2510.27246) | Long-term memory beyond million-token histories | Potentially useful for scale, but requires a careful streaming adapter and reproducible data setup | Later ultra-long stress test |

The first external run should use MemoryAgentBench and
`longmemeval_s_cleaned.json`. MemoryAgentBench is the closest match to our
incremental interaction design. LongMemEval-S is the strongest text-only
reference because each of its 500 instances contains timestamped sessions,
turn-level evidence labels, and answer-session labels. Its released S version
is roughly 115K tokens across about 40 sessions; the M version uses about 500
sessions and is better treated as a scale stress test. The `oracle` version,
which keeps only evidence sessions, is a useful upper-bound control rather than
an online harness workload.

LongMemEval-V2 is the most current benchmark in this list. It contains 451
curated questions, up to 500 web or enterprise trajectories per haystack, and
histories reaching roughly 115M tokens. Its official interface is a memory
backend that inserts trajectories and returns evidence items for a downstream
answerer. That interface is different from our online decoder and should not be
compared by raw leaderboard score. We should adapt each trajectory into a
turn-by-turn stream, then report both answer quality and the compact/read trace.
The current text-only Qwen backend cannot claim full V2 coverage until image
observations are supported.

GoodAI LTM is valuable for controlled scaling because the filler span can be
changed while the needles and scripts stay fixed. InMind isolates an important
Sigma failure mode: the relevant query may be semantically related to a fact
without sharing its words. Locomo-Plus adds a similar cognitive category to a
natural dialogue set, but its role here is diagnostic rather than the main
long-context result. ALFWorld trajectories can be used as an optional agent
state-continuity stress test; they are not a substitute for a memory benchmark
because task success often follows from the current observation and the
released archive is not a complete multimodal evaluation environment.

### Sigma-Harness Gate Suite

The self-defined suite is necessary even when external benchmarks are used. It
should contain short deterministic sessions with exact answers and explicit
metadata:

```text
recent_recall       answer remains in native KV
evicted_recall      answer is only in a compacted segment
knowledge_update    old fact is replaced by a later fact
temporal_reasoning  answer depends on event order or date
multi_hop           facts are split across multiple segments
distractor          unrelated segments surround the evidence
abstention          no segment contains the requested fact
```

Each instance records `fact_turn`, `query_turn`, `must_compact`, and the
expected evidence segment. The adapter rejects or separately reports an
instance when `must_compact=true` but no compact occurred. This suite is the
only reliable place to test the exact first-compact gate and all three write
schedules before spending model-inference time on a large benchmark.

The initial implementation is in
[`evaluation/gate-suite/`](evaluation/gate-suite/). It contains a JSONL case
file, a JSON schema, a deterministic runner, and a short usage guide. Run it
from the repository root with:

```bash
python evaluation/gate-suite/run_gate_suite.py
```

The current examples cover pre-compact write-only behavior, evicted-memory
read gating, repeated compact without duplicate segments, token-interval
remainder flushing, forced segment flush at `finish()`, and transactional
reader failure. The default matrix currently passes 12/12 TraceBackend runs.
The same cases can later be replayed with the Qwen or DeepSeek backend, but
their semantic answers must be reported separately from these protocol gates.

### Replay protocol for every external benchmark

For each history, sort sessions by their supplied timestamp and feed every
user/assistant turn through the harness. Keep the final benchmark question
outside the history and issue it as a new user turn. Set the native budget below
the full history, for example 4K, 8K, and 16K, and keep the output reserve
fixed. Store the benchmark evidence labels alongside the harness trace:

```text
question_id, history_tokens, native_budget, fact_session_ids
fact_turn_ids, evicted_before_query, compact_count, memory_version
read_enabled_at_query, reader_request_count, answer, gold_answer
```

The primary comparison is full-history versus native-only versus Sigma harness.
Report recent and evicted evidence separately. A correct answer without an
evicted evidence segment is not an evicted-memory result, and a `dmr-*` request
without a reader-state trace is not sufficient evidence that Sigma memory
affected the decoder.

### Compaction representation ablations

The harness exposes four compact representations through `CompactionConfig`:

| Mode | What remains in native context after compact | What represents evicted context |
| --- | --- | --- |
| `sigma` | Recent exact token tail | Sigma `H` and reader state |
| `summary` | Recent exact tail plus summary tokens | Text summary tokens |
| `hybrid` | Recent exact tail plus summary tokens | Text summary plus Sigma reader state |
| `recent_only` | Recent exact token tail | Nothing; no-memory control |

Run these modes with the same native budget, write schedule, reader checkpoint,
and question set. For `summary` and `hybrid`, hold the summary model and
`summary_max_tokens` fixed while comparing the compaction representation. The
summary provider receives the previous summary followed by newly evicted turns;
the resulting summary tokens are inserted before the retained tail and are not
written back into Sigma memory. Report summary generation latency and summary
token count separately from writer and reader costs.

The default summary provider is a separately loaded clean `Qwen/Qwen3-4B`
model. It is independent of the tuned Sigma reader and the Qwen3-0.6B writer,
and is loaded lazily at the first summary compact. Pin the same local model
path, prompt, tokenizer, decoding settings, and `summary_max_tokens` across all
summary and hybrid runs. A custom `SummaryBackend` can be injected for a
different summarizer, but it must be treated as a separate experimental factor.

## Minimal integration gates

Run both cases with the same reader checkpoint, tokenizer, decoding settings,
and one harness write schedule at a time. Then repeat them for `turn`,
`token_interval`, and `segment` to verify that the schedule does not change the
first-compact read gate.

### Case A: pre-compact write-only

Use a native context budget large enough that two short turns cannot trigger a
compact. The first turn can state a synthetic fact, for example:

```text
Remember that the project code is ALPHA-48271.
```

Run a second short turn and inspect both the harness state and backend trace.
This case passes only when all of the following hold:

- every result has `compacted=False` and `read_enabled=False`;
- `sigma_written_tokens > 0` while `sigma_readable_tokens == 0`;
- every append and generation call receives `read_state=None`;
- a Qwen SGLang delta metric sample reports `read_tokens == 0`;
- greedy output matches a fresh native-only control run.

The last check catches accidental memory influence even when the read-state
flag is not exposed by a serving wrapper.

### Case B: post-compact evicted recall

Use a small native budget, for example `context_window_tokens=64` and
`compact_target_tokens=16`. Run the fact turn, enough filler turns to exceed
the budget, and then ask:

```text
What is the project code? Answer only the code.
```

The fact turn must be in an evicted segment before the question is generated.
The protocol gate passes when:

- at least one compact occurs and `read_enabled=True` afterward;
- the fact segment has `in_native_cache=False` and appears in
  `readable_segments`;
- `memory_version` advances and the reader snapshot contains that segment;
- the question's backend generation receives a non-null read state;
- the Qwen SGLang delta metrics report `read_tokens > 0` and nonzero read
  strength or query-correction ratio.

Decode the answer and record exact-match recall as a separate semantic result.
An incorrect answer with positive read metrics means the wiring works but the
checkpoint or prompt needs a quality investigation; an answer without read
metrics is not evidence that Sigma memory was used.

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
