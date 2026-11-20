# Figure brief: Sigma-memory harness pipeline

## What this figure is

A double-column systems architecture figure showing the complete multi-turn Sigma-memory harness request path and its first-compact read gate. It explains how a user turn moves through the native KV path, how the configurable writer schedules produce an additive statistic `H`, and how compaction arms the trained reader for later turns.

## Components

The **Online turn path** lane contains **User turn** (external input), **Tokenizer** (existing model-side preprocessing), **Turn controller + capacity gate** (new harness orchestration), **Native KV tail** (the decoder's existing recent-context cache), **Reader-equipped decoder** (the configured reader model and its optional Sigma adapter), and **Assistant tokens** (the generated output). The **Write path** lane contains **Write timing** (the configurable `turn`, `token_interval`, or `segment` policy), **Sigma writer** (the independent frozen writer, Qwen3-0.6B in the current validation), and **Written memory H** (the additive statistic that is write-only before compaction). The **Compact/read path** lane contains **Compaction** (new harness state transition), **Readable memory H** (the evicted complete turns), and **Reader state** (the per-layer state prepared by the configured reader adapter). The decoder is drawn once in the online lane; the reader-state arrow enters it after the first compact.

## Flows and main path

① User turn → Tokenizer: raw user text becomes token ids. ② Tokenizer → Turn controller: the harness receives the turn and checks whether the prompt plus response budget fits the native KV budget. ③ Turn controller → Native KV tail → Reader-equipped decoder: recent context is generated from native KV; before the first compact, `read_state=None`. ④ Turn controller / Assistant tokens → Write timing → Sigma writer → Written memory H: the selected policy writes user and assistant tokens at turn end, in generated-token intervals, or after a segment buffer reaches capacity. ⑤ When capacity is exceeded, Written memory H → Compaction → Readable memory H: complete old turns are evicted and made readable. ⑥ Readable memory H → Reader state → Reader-equipped decoder: the reader state is armed and later generations use the `dmr-*` read path. The compact path also rebuilds the native KV tail. A feedback edge returns assistant tokens to the next turn.

The dashed boundary around the compact/read path means this path is conditional on overflow; it is not a second always-on model path. Solid arrows carry data, while a small legend explains the dashed compact trigger.

## The one point the figure argues

The native KV path is the fast path for recent context. Sigma memory is written independently, but it becomes a model input only after the first compact; the harness therefore separates writing from reading while preserving a recent native tail.

## Layout and style intent

Use the systems `pipeline` template at double-column width (504 pt), with three pale horizontal lanes and a left-to-right main path. Nest no more than one level. Use the blue-orange palette: neutral blue modules, pale yellow write lane, pale green compact/read lane, and one orange accent for the new turn controller. Use orthogonal connectors, a heavier main path, circled step markers ①–⑥, and a compact legend for solid data versus dashed compact-triggered flow. The closest template exemplars are `asplos26-132_fig6` for a setup/online split, `osdi24-050_fig6` for a compact controller with a cache hand-off, and `nsdi23-045_fig2` for numbered stages with labelled payloads.

## Assumptions to confirm

- The reader is shown as one decoder box with a Sigma adapter rather than a separate duplicate model, because the adapter is installed inside the reader.
- `Written memory H` is shown as a single logical store even though `segment` mode can retain per-turn `MemorySegment` objects.
- The diagram is an architecture overview for the README, so it uses current Qwen names only as a sublabel and keeps the harness protocol model-agnostic.

## Caption draft

**Sigma-memory harness pipeline.** Recent turns follow the native KV path. The configured writer updates `H` according to the selected write schedule, but the statistic remains write-only until compaction evicts complete turns; compaction then prepares reader state so subsequent decoder requests use Sigma memory alongside the retained native KV tail.
