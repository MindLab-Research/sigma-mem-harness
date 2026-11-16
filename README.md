# sima-mem-harness

最小的 Σ-mem 多轮对话 harness，用于验证 **compact 之前只使用主模型原生 KV，compact 之后才启用 Σ-mem 读取** 的会话协议。

这个仓库只实现会话编排，不替代 DeepSeek-V4.1 的训练或 serving backend。它复用了项目中的 Σ-mem 核心约定：writer 产生可加的二阶统计量 `H`，compact 时把被淘汰的历史转入可读 memory，reader 由 backend 将 memory 转成模型的 per-layer state。

## 协议

每个完成的 user/assistant turn 都会写一次 Σ-mem：

```text
turn -> writer -> written_memory
```

在第一次 compact 之前，`written_memory` 只用于写入统计，所有 backend 调用都收到 `read_state=None`：

```text
recent conversation -> native KV -> model
history            -> Sigma writer only
```

当新问题加上预计回答长度将超过 native context window 时，harness 会：

1. 从最老的 turn 开始淘汰，保留最近的 token tail；
2. 把淘汰 turn 的 memory 加到 `readable_memory`；
3. 重建 native KV tail；
4. 从这一刻开始给 backend 传 `read_state`，主模型才读取 Σ-mem。

compact 后新完成的 turn 也会写入 Σ-mem，并在下一轮读取。`written_memory` 与 `readable_memory` 分开，避免保留在 native KV 中的近期 turn 被重复读入。

## 本地运行

```bash
python -m pip install -e '.[test]'
pytest -q
python -m sima_mem_harness.smoke
```

`TraceBackend` 是一个确定性的协议测试 backend。它不假装实现真实 KV，只记录每次 append/rebuild/generate 是否收到 Sigma read state。

## 接入当前 DeepSeek-V4.1 实现

当前项目的 `delta_mem_miles/sigma/dpsk-v4.1` 仍是参考生成路径，尚未暴露增量 paged KV。`SigmaV41ReferenceBackend` 可以直接包住已经构造好的 `SigmaV41` 实例：

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

这个 reference adapter 仍由 `SigmaV41.generate` 重算 read 序列；后续接入真正的 vLLM/SGLang/DeepSeek paged-KV backend 时，只需实现 `SigmaBackend` 的五个方法，harness 的 compact/read gate 不变。

## 后续扩展

- 用实际 paged KV 对象替换 `LogicalKVCache`；
- compact 时接入当前模型的历史 token eviction 和 block manager；
- 在 `prepare_read_state` 中缓存每层 `S_l`，避免每个 decode step 重复求解；
- 加入 full-history、no-memory 和 compact-memory 三种对照评测。
