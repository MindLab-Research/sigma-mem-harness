"""Reference backend for the original Qwen Σ-mem SGLang path.

This adapter intentionally keeps the harness' native cache logical. The
original delta-mem SGLang engine owns its own request KV cache; the adapter
feeds it the current logical token sequence and arms its per-session Sigma
reader through ``delta_set_sigma_memory``. It is useful for protocol and
write-schedule validation before wiring a serving-side paged-KV object.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .backend import LogicalKVCache
from .memory import SigmaMemory


@dataclass(frozen=True)
class QwenSGLangReadState:
    session_id: str
    memory_version: int


@dataclass(frozen=True)
class QwenGenerationRecord:
    """Evidence that a generation used native-only or Sigma read mode."""

    read_enabled: bool
    rid: str | None
    input_tokens: int
    generated_tokens: int


@dataclass(frozen=True)
class QwenSigmaReaderConfig:
    """Configuration for the decoder and its trained Sigma reader adapter.

    The base decoder and adapter are deliberately separate fields: the
    adapter directory must match the decoder hidden size, while the writer is
    configured independently below.  This lets one harness select a Qwen3-4B
    or Qwen3-8B reader without changing the conversation state machine.
    """

    model_path: str
    adapter_dir: str | None = None
    rank: int = 64
    alpha: float = 16.0
    scaling: str = "rslora"
    num_state_heads: int = 32
    ffn_state_heads: int = 16
    attach: str = "attention:o mlp:down"
    write_mode: str = "sigma"
    read_mode: str = "frozen"
    sigma_dim: int = 1024
    sigma_ridge: str = "relative"
    sigma_ridge_init: float = 1.0
    sigma_ffn_query: str = "input"

    def environment(self) -> dict[str, str]:
        """Return the delta-mem environment consumed by spawned SGLang ranks."""

        return {
            "DELTA_MEM_RANK": str(self.rank),
            "DELTA_MEM_ALPHA": str(self.alpha),
            "DELTA_MEM_SCALING": self.scaling,
            "DELTA_MEM_NUM_STATE_HEADS": str(self.num_state_heads),
            "DELTA_MEM_FFN_STATE_HEADS": str(self.ffn_state_heads),
            "DELTA_MEM_ATTACH": self.attach,
            "DELTA_MEM_WRITE_MODE": self.write_mode,
            "DELTA_MEM_READ_MODE": self.read_mode,
            "DELTA_MEM_SIGMA_DIM": str(self.sigma_dim),
            "DELTA_MEM_SIGMA_RIDGE": self.sigma_ridge,
            "DELTA_MEM_SIGMA_RIDGE_INIT": str(self.sigma_ridge_init),
            "DELTA_MEM_SIGMA_FFN_QUERY": self.sigma_ffn_query,
        }

    def apply_environment(self) -> None:
        os.environ.update(self.environment())


@dataclass(frozen=True)
class QwenSigmaWriterConfig:
    """Configuration for the independent frozen Sigma-memory writer."""

    model_path: str
    layer: int = 19
    chunk: int = 8192
    sink: int = 1
    dtype: str = "bfloat16"
    weight: str = "surprisal"
    successor: float | None = None
    key_layer: int = 0
    split: str = "window"
    gate: int = 0
    state_path: str | None = None
    device: str = "cpu"

    def build(self) -> Any:
        """Construct the original project's ``SigmaWriter`` lazily."""

        from delta_mem_miles.sigma.writer import SigmaWriter, SigmaWriterConfig

        native = SigmaWriterConfig(
            model_path=self.model_path,
            layer=self.layer,
            chunk=self.chunk,
            sink=self.sink,
            dtype=self.dtype,
            weight=self.weight,
            successor=self.successor,
            key_layer=self.key_layer,
            split=self.split,
            gate=self.gate,
            state_path=self.state_path or "",
        )
        return SigmaWriter(native, device=self.device)


@dataclass(frozen=True)
class QwenSGLangConfig:
    """Complete, serializable model/writer/reader configuration.

    ``engine_kwargs`` is passed unchanged to the original
    ``build_delta_engine`` function, so deployment-specific SGLang settings
    remain configurable without adding them to the harness API.
    """

    reader: QwenSigmaReaderConfig
    writer: QwenSigmaWriterConfig
    session_id: str = "sigma-harness"
    local_files_only: bool = True
    engine_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            object.__setattr__(
                self,
                "engine_kwargs",
                {
                    "tp_size": 1,
                    "context_length": 128,
                    "mem_fraction_static": 0.40,
                    "disable_cuda_graph": True,
                    "attention_backend": "triton",
                    "decode_attention_backend": "triton",
                    "sampling_backend": "pytorch",
                    "log_level": "warning",
                },
            )

    @classmethod
    def from_checkpoint(
        cls,
        reader_model_path: str,
        checkpoint_dir: str,
        *,
        adapter_dir: str | None = None,
        writer_model_path: str | None = None,
        session_id: str = "sigma-harness",
        local_files_only: bool = True,
        engine_kwargs: Mapping[str, Any] | None = None,
    ) -> "QwenSGLangConfig":
        """Build a config from a copied checkpoint directory.

        The newest ``model/iter_*`` directory is selected automatically, and
        its ``sigma_writer.json`` and ``sigma_writer_state.pt`` are used to
        configure the writer.  ``writer_model_path`` can override the path
        recorded by training when the checkpoint is mounted elsewhere.
        """

        root = Path(checkpoint_dir)
        if adapter_dir is None:
            iterations = [
                path
                for path in (root / "model").glob("iter_*")
                if (path / "delta_mem_adapter.pt").is_file()
            ]
            if not iterations:
                raise FileNotFoundError(f"no adapter iteration found under {root / 'model'}")
            adapter = max(iterations, key=lambda path: int(path.name.removeprefix("iter_")))
        else:
            adapter = Path(adapter_dir)
        record_path = adapter / "sigma_writer.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else {}
        reader_record_path = adapter / "delta_mem_config.json"
        reader_record = (
            json.loads(reader_record_path.read_text())
            if reader_record_path.is_file()
            else {}
        )
        writer_path = writer_model_path or record.get("model_path")
        if not writer_path:
            raise ValueError(f"writer_model_path is required; no model_path in {record_path}")
        state_path = adapter / "sigma_writer_state.pt"
        heads = reader_record.get("heads_by_family", {})
        return cls(
            reader=QwenSigmaReaderConfig(
                model_path=reader_model_path,
                adapter_dir=str(adapter),
                rank=int(reader_record.get("rank", 64)),
                alpha=float(reader_record.get("alpha", 16.0)),
                scaling=str(reader_record.get("scaling", "rslora")),
                num_state_heads=int(reader_record.get("num_state_heads", 32)),
                ffn_state_heads=int(heads.get("mlp", 16)),
                sigma_ridge=str(reader_record.get("sigma_ridge", "relative")),
                sigma_ffn_query=str(reader_record.get("sigma_ffn_query", "input")),
            ),
            writer=QwenSigmaWriterConfig(
                model_path=str(writer_path),
                layer=int(record.get("layer", 19)),
                chunk=int(record.get("chunk", 8192)),
                sink=int(record.get("sink", 1)),
                weight=str(record.get("weight", "surprisal")),
                successor=record.get("successor"),
                key_layer=int(record.get("key_layer", 0)),
                split=str(record.get("split", "window")),
                # A trained gate is loaded from the saved state at serving
                # time; passing gate=64 together with state_path is invalid.
                gate=0 if state_path.is_file() else int(record.get("gate", 0)),
                state_path=str(state_path) if state_path.is_file() else None,
            ),
            session_id=session_id,
            local_files_only=local_files_only,
            engine_kwargs=engine_kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reader": self.reader.__dict__.copy(),
            "writer": self.writer.__dict__.copy(),
            "session_id": self.session_id,
            "local_files_only": self.local_files_only,
            "engine_kwargs": dict(self.engine_kwargs or {}),
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "QwenSGLangConfig":
        payload = json.loads(Path(path).read_text())
        return cls(
            reader=QwenSigmaReaderConfig(**payload["reader"]),
            writer=QwenSigmaWriterConfig(**payload["writer"]),
            session_id=payload.get("session_id", "sigma-harness"),
            local_files_only=bool(payload.get("local_files_only", True)),
            engine_kwargs=payload.get("engine_kwargs"),
        )

    def build(self) -> "QwenSGLangRuntime":
        """Construct tokenizer, writer, SGLang engine, and harness backend."""

        self.reader.apply_environment()
        from transformers import AutoTokenizer
        from delta_mem_miles.sglang.engine_control import build_delta_engine

        writer = self.writer.build()
        engine = build_delta_engine(
            self.reader.model_path,
            adapter_dir=self.reader.adapter_dir,
            delta_mem_rank=self.reader.rank,
            **dict(self.engine_kwargs or {}),
        )
        engine.assert_delta_modules_present()
        tokenizer = AutoTokenizer.from_pretrained(
            self.reader.model_path,
            local_files_only=self.local_files_only,
        )
        backend = QwenSGLangBackend(engine, writer, session_id=self.session_id)
        return QwenSGLangRuntime(
            config=self,
            engine=engine,
            writer=writer,
            tokenizer=tokenizer,
            backend=backend,
        )


@dataclass
class QwenSGLangRuntime:
    """Live objects produced by :meth:`QwenSGLangConfig.build`."""

    config: QwenSGLangConfig
    engine: Any
    writer: Any
    tokenizer: Any
    backend: "QwenSGLangBackend"

    def shutdown(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if shutdown is not None:
            shutdown()


class QwenSGLangBackend:
    """Wrap ``delta_mem_miles.sglang.engine_control`` for Sigma reads."""

    def __init__(self, engine: Any, writer: Any, *, session_id: str = "sigma-harness") -> None:
        self.engine = engine
        self.writer = writer
        self.session_id = session_id
        self._memory_version = 0
        self._read_index = 0
        self.prepare_read_calls = 0
        self.generation_calls: list[QwenGenerationRecord] = []

        from delta_mem_miles.sglang.common import delta_read_rid

        self._delta_read_rid = delta_read_rid

    def write_memory(self, token_ids: Sequence[int]) -> SigmaMemory:
        try:
            return self.writer.encode(list(token_ids), storage_device="cpu")
        except TypeError:
            # Original delta-mem-miles writer API returns its configured device
            # directly; newer writer variants accept storage_device.
            return self.writer.encode(list(token_ids))

    def prepare_read_state(self, memory: SigmaMemory) -> QwenSGLangReadState:
        self._memory_version += 1
        self.prepare_read_calls += 1
        self.engine.delta_set_sigma_memory(self.session_id, memory)
        return QwenSGLangReadState(self.session_id, self._memory_version)

    def append_native(self, cache, token_ids, *, read_state):
        previous = () if cache is None else cache.token_ids
        return LogicalKVCache(tuple(previous) + tuple(int(token) for token in token_ids))

    def rebuild_native(self, token_ids, *, read_state):
        return LogicalKVCache(tuple(int(token) for token in token_ids))

    def generate(
        self,
        cache,
        *,
        read_state,
        max_new_tokens,
        eos_token_id,
        on_token_chunk: Callable[[Sequence[int]], None] | None = None,
    ):
        sampling_params = {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "max_new_tokens": int(max_new_tokens),
        }
        rid = None
        if read_state is not None:
            rid = self._delta_read_rid(self.session_id, self._read_index)
            self._read_index += 1
        result = self.engine.generate(
            input_ids=list(cache.token_ids),
            sampling_params=sampling_params,
            rid=rid,
            cache_salt=f"sigma-mem-harness:{self.session_id}",
        )
        generated = result.get("output_ids", result.get("token_ids", []))
        if generated and isinstance(generated[0], list):
            generated = generated[0]
        generated = tuple(int(token) for token in generated)
        self.generation_calls.append(
            QwenGenerationRecord(
                read_enabled=read_state is not None,
                rid=rid,
                input_tokens=len(cache.token_ids),
                generated_tokens=len(generated),
            )
        )
        if on_token_chunk is not None and generated:
            # The SGLang Engine API returns a completed response here. A
            # streaming engine can replace this with incremental callbacks.
            on_token_chunk(generated)
        return generated


__all__ = ["QwenGenerationRecord", "QwenSGLangBackend", "QwenSGLangReadState"]
