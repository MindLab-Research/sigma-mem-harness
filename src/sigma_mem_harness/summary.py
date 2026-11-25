"""Text-summary backends used by compaction ablations.

The summary model is deliberately independent from the Sigma reader and
writer.  The default is a clean Qwen3-4B model; model weights are loaded only
when the first summary is requested so importing the harness and running the
protocol tests stays lightweight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


DEFAULT_SUMMARY_MODEL = "Qwen/Qwen3-4B"
DEFAULT_SUMMARY_PROMPT = (
    "Summarize the conversation below for future turns. Preserve concrete facts, "
    "names, dates, preferences, decisions, commitments, and unresolved questions. "
    "Resolve repeated information when possible. Output only a concise factual "
    "summary; do not describe this instruction and do not add guesses.\n\n"
    "Conversation:\n"
)


def _default_model_path() -> str:
    """Return the configured clean model, with a portable HF-id fallback."""

    return os.environ.get("SIGMA_SUMMARY_MODEL_PATH", DEFAULT_SUMMARY_MODEL)


@dataclass(frozen=True)
class QwenSummaryConfig:
    """Configuration for the independent clean Qwen summary model.

    ``model_path`` defaults to ``Qwen/Qwen3-4B`` rather than either tuned
    Sigma checkpoint.  ``local_files_only`` defaults to true because harness
    evaluation should use an explicitly available local model and avoid a
    hidden network download.  Set it to false when the model should be fetched
    from the Hugging Face hub.
    """

    model_path: str = field(default_factory=_default_model_path)
    local_files_only: bool = True
    device: str = "auto"
    dtype: str = "bfloat16"
    max_input_tokens: int = 32768
    system_prompt: str = DEFAULT_SUMMARY_PROMPT
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")
        if self.device not in ("auto", "cpu", "cuda") and not self.device.startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")


class QwenSummaryBackend:
    """Generate natural-language summaries with an independent Qwen model.

    The imports and model construction are lazy.  This keeps the core harness
    usable without Transformers installed and makes ``CompactionConfig`` cheap
    to construct.  Tests can inject ``tokenizer`` and ``model`` directly,
    while production runs normally pass only a ``QwenSummaryConfig``.
    """

    def __init__(
        self,
        config: QwenSummaryConfig | None = None,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or QwenSummaryConfig()
        self.tokenizer = tokenizer
        self.model = model
        self.load_count = 0

    @property
    def is_loaded(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def _load(self) -> None:
        if self.is_loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "QwenSummaryBackend requires transformers and torch; install the "
                "summary extra or inject tokenizer/model objects"
            ) from exc

        load_kwargs = dict(self.config.model_kwargs)
        load_kwargs.setdefault("local_files_only", self.config.local_files_only)
        if self.config.dtype != "auto":
            try:
                load_kwargs.setdefault("torch_dtype", getattr(torch, self.config.dtype))
            except AttributeError as exc:
                raise ValueError(f"unsupported torch dtype: {self.config.dtype}") from exc

        use_device_map = self.config.device == "auto" and torch.cuda.is_available()
        if use_device_map:
            load_kwargs.setdefault("device_map", "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=self.config.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **load_kwargs,
        )
        if not use_device_map:
            target = (
                "cuda"
                if self.config.device == "auto" and torch.cuda.is_available()
                else "cpu"
                if self.config.device == "auto"
                else self.config.device
            )
            self.model.to(target)
        self.model.eval()
        self.load_count += 1

    def _model_device(self) -> Any:
        import torch

        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return model_device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _encode_prompt(self, source_text: str) -> Any:
        """Apply Qwen's chat template, with a plain-text fallback."""

        messages = [{"role": "user", "content": self.config.system_prompt + source_text}]
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is not None:
            try:
                return apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    truncation=True,
                    max_length=self.config.max_input_tokens,
                    return_tensors="pt",
                )
            except TypeError:
                # Older Transformers releases do not know enable_thinking or
                # truncation in every Qwen template implementation.
                try:
                    return apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                except TypeError:
                    pass

        encoded = self.tokenizer(
            self.config.system_prompt + source_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        return encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids

    def summarize(self, token_ids: Sequence[int], *, max_new_tokens: int) -> Sequence[int]:
        """Return generated summary token ids, excluding the prompt tokens."""

        if max_new_tokens < 1 or not token_ids:
            return ()
        self._load()
        source_text = self.tokenizer.decode(
            [int(token) for token in token_ids],
            skip_special_tokens=True,
        )
        input_ids = self._encode_prompt(source_text).to(self._model_device())
        attention_mask = input_ids.new_ones(input_ids.shape)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        import torch

        with torch.inference_mode():
            output = self.model.generate(**generation_kwargs)
        generated = output[0, input_ids.shape[-1] :]
        return tuple(int(token) for token in generated.tolist())[:max_new_tokens]


__all__ = [
    "DEFAULT_SUMMARY_MODEL",
    "DEFAULT_SUMMARY_PROMPT",
    "QwenSummaryBackend",
    "QwenSummaryConfig",
]
