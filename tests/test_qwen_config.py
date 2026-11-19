import json

from sima_mem_harness import (
    QwenSGLangConfig,
    QwenSigmaReaderConfig,
    QwenSigmaWriterConfig,
)


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
