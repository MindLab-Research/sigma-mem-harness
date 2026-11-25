import json

from sigma_mem_harness import (
    CompactionConfig,
    DEFAULT_SUMMARY_MODEL,
    QwenSummaryBackend,
    QwenSummaryConfig,
    QwenSGLangConfig,
    QwenSigmaReaderConfig,
    QwenSigmaWriterConfig,
)


def test_summary_mode_defaults_to_clean_qwen4b_without_loading_weights():
    config = CompactionConfig(mode="summary")

    assert isinstance(config.summary_backend, QwenSummaryBackend)
    assert config.summary_backend.config.model_path == DEFAULT_SUMMARY_MODEL
    assert config.summary_backend.is_loaded is False


def test_qwen_summary_backend_returns_only_generated_ids():
    import torch

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return "old conversation"

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"].endswith("old conversation")
            assert kwargs["add_generation_prompt"] is True
            return torch.tensor([[11, 12]])

    class FakeModel:
        device = torch.device("cpu")

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [[11, 12]]
            assert kwargs["max_new_tokens"] == 2
            return torch.tensor([[11, 12, 21, 22]])

    backend = QwenSummaryBackend(
        QwenSummaryConfig(device="cpu"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    assert backend.summarize([1, 2], max_new_tokens=2) == (21, 22)


def test_reader_config_exports_spawned_rank_environment():
    config = QwenSigmaReaderConfig(model_path="reader", adapter_dir="adapter")

    env = config.environment()

    assert env["DELTA_MEM_RANK"] == "64"
    assert env["DELTA_MEM_ATTACH"] == "attention:o mlp:down"
    assert env["DELTA_MEM_SIGMA_DIM"] == "1024"


def test_checkpoint_config_selects_latest_adapter_and_saved_writer(tmp_path):
    root = tmp_path / "checkpoint"
    old = root / "model" / "iter_0009"
    latest = root / "model" / "iter_0010"
    old.mkdir(parents=True)
    latest.mkdir(parents=True)
    (old / "delta_mem_adapter.pt").write_bytes(b"old")
    (latest / "delta_mem_adapter.pt").write_bytes(b"latest")
    (latest / "sigma_writer_state.pt").write_bytes(b"state")
    (latest / "sigma_writer.json").write_text(
        json.dumps(
            {
                "model_path": "writer",
                "layer": 19,
                "chunk": 8192,
                "sink": 1,
                "weight": "surprisal",
            }
        )
    )

    config = QwenSGLangConfig.from_checkpoint("reader", str(root))

    assert config.reader.model_path == "reader"
    assert config.reader.adapter_dir == str(latest)
    assert config.writer.model_path == "writer"
    assert config.writer.state_path == str(latest / "sigma_writer_state.pt")


def test_qwen_config_json_round_trip(tmp_path):
    config = QwenSGLangConfig(
        reader=QwenSigmaReaderConfig(model_path="reader", adapter_dir="adapter"),
        writer=QwenSigmaWriterConfig(model_path="writer", state_path="state.pt"),
        engine_kwargs={"tp_size": 2},
    )
    path = tmp_path / "qwen.json"

    config.save_json(path)
    loaded = QwenSGLangConfig.from_json(path)

    assert loaded == config
