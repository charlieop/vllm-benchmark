"""Native Diffusers backend for Qwen-Image 2.1 image editing.

The 16 GB preset is deliberately conservative: the diffusion transformer uses
bitsandbytes NF4 while the bf16 Qwen3-VL text encoder and fp16 VAE are
CPU-offloaded between calls. NF4 components are loaded directly on their target
device because moving or Accelerate-offloading bitsandbytes modules is not a
portable operation. Image dimensions still determine activation memory, so the
preset cannot guarantee that every input fits in 16 GB.

Runtime dependencies are ``torch``, ``diffusers``, ``transformers``,
``accelerate``, ``bitsandbytes`` (for int8/NF4), and ``huggingface_hub``.
They are imported only when the first image is generated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .base import FatalModelError, GenerationOutput, SampleModelError

LargePrecision = Literal["bf16", "fp16", "int8", "nf4"]
VaePrecision = Literal["fp32", "bf16", "fp16"]


@dataclass(frozen=True)
class QwenConfig:
    """Configuration for a pinned Qwen-Image 2.1 Hugging Face checkpoint."""

    model_id: str
    revision: str
    cache_dir: str | Path
    transformer_precision: LargePrecision = "bf16"
    text_encoder_precision: LargePrecision = "bf16"
    vae_precision: VaePrecision = "fp32"
    transformer_cpu_offload: bool = False
    text_encoder_cpu_offload: bool = False
    vae_cpu_offload: bool = False
    device: str = "cuda"
    num_inference_steps: int = 40
    true_cfg_scale: float = 1.0
    negative_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be a non-empty Hugging Face repository ID")
        if not self.revision.strip() or self.revision == "main":
            raise ValueError("revision must pin an explicit checkpoint revision, not 'main'")
        if not str(self.cache_dir).strip():
            raise ValueError("cache_dir must be configured explicitly")
        if self.transformer_precision not in {"bf16", "fp16", "int8", "nf4"}:
            raise ValueError(f"unsupported transformer_precision: {self.transformer_precision}")
        if self.text_encoder_precision not in {"bf16", "fp16", "int8", "nf4"}:
            raise ValueError(f"unsupported text_encoder_precision: {self.text_encoder_precision}")
        if self.vae_precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"unsupported vae_precision: {self.vae_precision}")
        if self.transformer_cpu_offload and self.transformer_precision in {"int8", "nf4"}:
            raise ValueError("CPU offload is unsupported for a quantized transformer")
        if self.text_encoder_cpu_offload and self.text_encoder_precision in {"int8", "nf4"}:
            raise ValueError("CPU offload is unsupported for a quantized text encoder")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if self.true_cfg_scale < 0:
            raise ValueError("true_cfg_scale must be non-negative")
        if self.true_cfg_scale > 1 and not self.negative_prompt:
            raise ValueError("true_cfg_scale > 1 requires a non-empty negative_prompt")

    @classmethod
    def vram_16gb(
        cls, *, model_id: str, revision: str, cache_dir: str | Path, **overrides: Any
    ) -> "QwenConfig":
        """Return the documented 16 GB-oriented NF4/offload configuration."""
        values: dict[str, Any] = {
            "model_id": model_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "transformer_precision": "nf4",
            "text_encoder_precision": "bf16",
            "vae_precision": "fp16",
            "text_encoder_cpu_offload": True,
            "vae_cpu_offload": True,
        }
        values.update(overrides)
        return cls(**values)


class QwenBackend:
    """One-at-a-time native Qwen image-edit inference through Diffusers."""

    def __init__(self, config: QwenConfig):
        self.config = config
        self._pipe: Any | None = None
        self._torch: Any | None = None

    @staticmethod
    def _dtype(torch: Any, precision: str) -> Any:
        return {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "int8": torch.float16,
            "nf4": torch.bfloat16,
        }[precision]

    @staticmethod
    def _quant_config(config_type: Any, precision: str, torch: Any) -> Any | None:
        if precision == "int8":
            return config_type(load_in_8bit=True)
        if precision == "nf4":
            return config_type(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        return None

    @staticmethod
    def _quantized_device_map(precision: str, device: str) -> dict[str, str] | None:
        """Place BnB weights while loading; they must not be moved afterward."""
        return {"": device} if precision in {"int8", "nf4"} else None

    @staticmethod
    def _place_component(
        component: Any,
        *,
        precision: str,
        cpu_offload_enabled: bool,
        cpu_offload_fn: Any,
        device: str,
    ) -> None:
        if cpu_offload_enabled:
            cpu_offload_fn(component, execution_device=device)
        elif precision not in {"int8", "nf4"}:
            component.to(device)

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from accelerate import cpu_offload
            from diffusers import AutoModel as DiffusersAutoModel
            from diffusers import AutoencoderKLQwenImage21
            from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
            from diffusers import QwenImage21Pipeline
            from transformers import BitsAndBytesConfig as TransformersBnbConfig
            from transformers import Qwen3VLForConditionalGeneration
        except (ImportError, OSError) as exc:
            raise FatalModelError(
                "Qwen local inference requires torch, diffusers, transformers, accelerate, "
                "huggingface_hub, and bitsandbytes when using int8/NF4"
            ) from exc

        cfg = self.config
        common = {
            "revision": cfg.revision,
            "cache_dir": str(cfg.cache_dir),
            "low_cpu_mem_usage": True,
        }
        try:
            transformer_load = dict(common)
            transformer_device_map = self._quantized_device_map(
                cfg.transformer_precision, cfg.device
            )
            if transformer_device_map is not None:
                transformer_load["device_map"] = transformer_device_map
            transformer = DiffusersAutoModel.from_pretrained(
                cfg.model_id,
                subfolder="transformer",
                dtype=self._dtype(torch, cfg.transformer_precision),
                quantization_config=self._quant_config(
                    DiffusersBnbConfig, cfg.transformer_precision, torch
                ),
                **transformer_load,
            )
            text_encoder_load = dict(common)
            text_encoder_device_map = self._quantized_device_map(
                cfg.text_encoder_precision, cfg.device
            )
            if text_encoder_device_map is not None:
                text_encoder_load["device_map"] = text_encoder_device_map
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                cfg.model_id,
                subfolder="text_encoder",
                dtype=self._dtype(torch, cfg.text_encoder_precision),
                quantization_config=self._quant_config(
                    TransformersBnbConfig, cfg.text_encoder_precision, torch
                ),
                **text_encoder_load,
            )
            vae = AutoencoderKLQwenImage21.from_pretrained(
                cfg.model_id,
                subfolder="vae",
                dtype=self._dtype(torch, cfg.vae_precision),
                **common,
            )
            pipe = QwenImage21Pipeline.from_pretrained(
                cfg.model_id,
                transformer=transformer,
                text_encoder=text_encoder,
                vae=vae,
                dtype=self._dtype(torch, cfg.vae_precision),
                **common,
            )
            for component, offload, precision in (
                (transformer, cfg.transformer_cpu_offload, cfg.transformer_precision),
                (text_encoder, cfg.text_encoder_cpu_offload, cfg.text_encoder_precision),
                (vae, cfg.vae_cpu_offload, cfg.vae_precision),
            ):
                self._place_component(
                    component,
                    precision=precision,
                    cpu_offload_enabled=offload,
                    cpu_offload_fn=cpu_offload,
                    device=cfg.device,
                )
        except torch.OutOfMemoryError as exc:
            raise FatalModelError("GPU out of memory while loading Qwen-Image; settings were not changed") from exc
        except Exception as exc:
            raise FatalModelError(f"failed to load pinned Qwen checkpoint: {exc}") from exc
        self._torch = torch
        self._pipe = pipe

    def generate(
        self, image: Image.Image, prompt: str, system_prompt: str, seed: int | None
    ) -> GenerationOutput:
        self._load()
        assert self._pipe is not None and self._torch is not None
        if image.width % 32 or image.height % 32:
            raise SampleModelError(
                "Qwen-Image 2.1 requires width and height divisible by 32; "
                "preprocess this sample in the dataset loader"
            )
        full_prompt = f"{system_prompt.rstrip()}\n\n{prompt}" if system_prompt.strip() else prompt
        generator = None
        if seed is not None:
            generator = self._torch.Generator(device=self.config.device).manual_seed(seed)
        kwargs: dict[str, Any] = {
            "image": image,
            "height": image.height,
            "width": image.width,
            "prompt": full_prompt,
            "generator": generator,
            "num_inference_steps": self.config.num_inference_steps,
            "true_cfg_scale": self.config.true_cfg_scale,
        }
        if self.config.negative_prompt is not None:
            kwargs["negative_prompt"] = self.config.negative_prompt
        try:
            result = self._pipe(**kwargs)
            output = result.images[0]
        except self._torch.OutOfMemoryError as exc:
            raise FatalModelError(
                "GPU out of memory during Qwen-Image inference; settings were not changed"
            ) from exc
        serialized_config = asdict(self.config)
        serialized_config["cache_dir"] = str(self.config.cache_dir)
        return GenerationOutput(
            image=output,
            metadata={
                "model_id": self.config.model_id,
                "revision": self.config.revision,
                "seed": seed,
                "qwen_config": serialized_config,
            },
        )

    def close(self) -> None:
        self._pipe = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None


# Descriptive alias for callers that name backends after their model.
QwenImageBackend = QwenBackend
