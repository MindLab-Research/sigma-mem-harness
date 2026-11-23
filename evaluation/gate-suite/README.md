# Sigma-Harness Gate Suite

This is the deterministic protocol suite for `sigma-mem-harness`. It uses
`TraceBackend`, so failures identify harness state-machine bugs rather than
model quality or serving noise.

The cases cover:

- no-overflow write-only behavior;
- first-compact evicted-memory read gating;
- repeated compact without duplicate segments;
- token-interval remainder flushing;
- forced `segment` flush at `finish()`;
- transactional reader failure handling.

Run every case under every applicable write schedule:

```bash
python evaluation/gate-suite/run_gate_suite.py
```

Run one case or write a JSON report:

```bash
python evaluation/gate-suite/run_gate_suite.py \
  --case evicted_recall_gate \
  --output /tmp/sigma-gate-report.json
```

The suite is intentionally separate from semantic benchmark evaluation. A real
Qwen/DeepSeek backend can reuse the JSONL cases after replacing `TraceBackend`,
but the result must then report both the protocol trace and the generated
answer.
