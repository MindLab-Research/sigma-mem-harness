# Sigma-Harness Gate Suite

The Gate Suite is the protocol test layer for `sigma-mem-harness`. It answers a
narrow question before any large benchmark is run:

> Given a sequence of turns and a native-KV budget, did the harness perform the
> exact write, compact, and read transitions that the Sigma-memory design
> requires?

This is separate from answer quality. A model can answer a question correctly
without reading Sigma memory, and it can read Sigma memory correctly while still
producing a poor answer. Keeping these two questions separate makes failures
localizable.

## Why this suite is needed

LongMemEval, MemoryAgentBench, and other external benchmarks tell us whether an
agent remembers information. They do not tell us whether a failed result came
from the benchmark adapter, the compact trigger, the writer, the reader, the
model, or the answer judge. They also do not guarantee that a particular sample
will overflow the native KV budget.

The Gate Suite controls the token IDs, turn lengths, compact budget, and write
schedule. It makes the first-compact rule observable:

```text
before first compact:
    native KV -> generation
    Sigma writer -> written H
    reader is disabled

after compact:
    old complete turns -> readable H -> reader state
    recent tail -> native KV
    native KV + Sigma reader -> generation
```

The suite uses the deterministic `TraceBackend`, so a failure points to the
harness state machine rather than model quality, tokenizer behavior, GPU
kernels, or serving latency. The same case file can later be replayed with a
real Qwen or DeepSeek backend as an integration smoke test.

## What is tested

| Case | What it protects | Main assertions |
| --- | --- | --- |
| `precompact_write_only` | The delayed-read contract | No compact, no reader state, no read generation, positive writes |
| `evicted_recall_gate` | The first compact read gate | Compact occurs, query sees a reader state, readable segments exist |
| `multi_compact_no_duplicate` | Long sessions and additive memory bookkeeping | Multiple compacts, unique readable segment IDs |
| `token_interval_remainder` | Mid-generation schedule correctness | Full `n`-token chunks and the final incomplete remainder are written |
| `segment_finish_flush` | Durability at dialogue end | A below-capacity segment is written by `finish()` |
| `reader_failure_atomicity` | Transactional compaction | A failed reader solve leaves the old KV and read gate unchanged |

These cases cover mechanics that a final QA score cannot isolate. More cases can
be added for snapshot reuse, stale memory versions, knowledge updates, and
restart persistence as those interfaces are implemented.

## Case format

Cases live in [`cases.jsonl`](cases.jsonl). A case contains tokenized user
prompts, a context budget, the schedules to run, and protocol expectations. A
small example is:

```json
{
  "case_id": "evicted_recall_gate",
  "turns": [[1, 2, 3], [4, 5, 6], [7]],
  "max_new_tokens": 1,
  "budgets": {
    "context_window_tokens": 7,
    "compact_target_tokens": 4,
    "reserve_tokens": 0
  },
  "schedules": ["turn", "token_interval", "segment"],
  "schedule_params": {
    "token_interval_tokens": 2,
    "segment_capacity_tokens": 100
  },
  "query_turn_index": 3,
  "expected": {
    "must_compact": true,
    "query_read_enabled": true,
    "unique_readable_segments": true
  }
}
```

The prompts are token IDs because protocol gates should not depend on a
particular tokenizer. A real-model adapter can translate text to token IDs and
reuse the same case metadata. The schema is documented in
[`schema.json`](schema.json).

The budget is chosen so that a query is legal by itself but the accumulated
native tail cannot fit it. This is deliberate: calling `compact()` manually
would only test the compaction method; allowing the budget check to trigger it
tests the real online path.

## Write schedules

The runner expands each case over its applicable schedules:

- `turn`: write the user and assistant parts at the end of the turn;
- `token_interval`: write the prompt and every configured `n` generated tokens,
  then flush the remainder;
- `segment`: queue complete turns until the configured capacity, a compact, or
  `finish()` forces a write.

The first-compact read gate is identical for all three schedules. The suite
also checks that a pre-compact schedule cannot change generation through
pending memory. The real-model evaluation should additionally report writer
batch boundaries because different boundaries can change the writer features;
that is a quality ablation, not a protocol failure.

## Running the suite

Run all cases and all applicable schedules from the repository root:

```bash
python evaluation/gate-suite/run_gate_suite.py
```

Run only one case:

```bash
python evaluation/gate-suite/run_gate_suite.py \
  --case evicted_recall_gate
```

Write a machine-readable report:

```bash
python evaluation/gate-suite/run_gate_suite.py \
  --output /tmp/sigma-gate-report.json
```

The current deterministic matrix has 12 runs and should report:

```text
12/12 runs passed
```

Each result includes compact count, memory version, readable segment IDs,
writer call count, pending segment tokens, generation read flags, and generated
token IDs. The generated tokens are deterministic test outputs; they are not a
semantic answer score.

## How to interpret failures

- `precompact_write_only` fails: memory may be leaking into generation before
  the first compact, or the case is accidentally overflowing its budget.
- `evicted_recall_gate` fails: the reader gate, compact trigger, or native-tail
  rebuild is wrong.
- `multi_compact_no_duplicate` fails: memory can be double-counted, which would
  corrupt the reader state on long sessions.
- `token_interval_remainder` fails: a crash or turn boundary can lose the tail
  of a generated response.
- `segment_finish_flush` fails: a successful dialogue can end with unwritten
  memory.
- `reader_failure_atomicity` fails: a failed solve can partially evict the
  native KV, leaving the session unrecoverable.

A passing Gate Suite does not prove that Sigma improves answer accuracy or
latency. It proves that the experiment is actually exercising the intended
protocol and that later quality results can be interpreted. Those questions are
covered by the external benchmark replay plan in
[`EVALUATION.md`](../../EVALUATION.md).

## Extending to real backends

The runner currently constructs `TraceBackend` intentionally. A real integration
runner should keep the same JSONL cases and replace only the backend factory:

```text
TraceBackend                 -> protocol regression
QwenSGLangBackend            -> Qwen adapter and dmr read requests
SigmaV41ReferenceBackend     -> DeepSeek reference path
paged-KV serving backend     -> production memory and latency evaluation
```

For a real backend, report both parts of the result:

```text
protocol trace: compact/read/write state, memory version, reader request
model result:   generated answer, exact match/F1, latency, memory footprint
```

A correct answer without a reader trace does not prove Sigma memory was used.
A reader trace with an incorrect answer means the wiring works and the reader,
writer, prompt, or checkpoint quality needs investigation.
