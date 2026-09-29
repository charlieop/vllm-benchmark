from pathlib import Path

import pytest

from visionbench.models.base import SampleModelError
from visionbench.models.qwen import QwenBackend, QwenConfig


BASE = {"model_id": "Qwen/Qwen-Image-2.1", "revision": "abc123", "cache_dir": "/models"}


@pytest.mark.parametrize("precision", ["bf16", "fp16", "int8", "nf4"])
def test_large_component_precisions_are_independent(precision: str) -> None:
    config = QwenConfig(**BASE, transformer_precision=precision, text_encoder_precision="bf16")
    assert config.transformer_precision == precision
    config = QwenConfig(**BASE, transformer_precision="fp16", text_encoder_precision=precision)
    assert config.text_encoder_precision == precision


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_vae_precisions(precision: str) -> None:
    assert QwenConfig(**BASE, vae_precision=precision).vae_precision == precision


@pytest.mark.parametrize(
    ("field", "value"),
    [("transformer_precision", "fp8"), ("text_encoder_precision", "fp32"), ("vae_precision", "nf4")],
)
def test_unsupported_precisions_fail(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        QwenConfig(**BASE, **{field: value})


def test_checkpoint_and_cache_must_be_explicit() -> None:
    with pytest.raises(ValueError, match="revision"):
        QwenConfig(**{**BASE, "revision": "main"})
    with pytest.raises(ValueError, match="cache_dir"):
        QwenConfig(**{**BASE, "cache_dir": ""})


def test_true_cfg_requires_negative_prompt() -> None:
    with pytest.raises(ValueError, match="negative_prompt"):
        QwenConfig(**BASE, true_cfg_scale=2.0)
    config = QwenConfig(**BASE, true_cfg_scale=2.0, negative_prompt="blurry")
    assert config.negative_prompt == "blurry"


def test_16gb_preset_uses_nf4_and_component_offload() -> None:
    config = QwenConfig.vram_16gb(**BASE)
    assert config.transformer_precision == "nf4"
    assert config.text_encoder_precision == "bf16"
    assert config.vae_precision == "fp16"
    assert not config.transformer_cpu_offload
    assert config.text_encoder_cpu_offload
    assert config.vae_cpu_offload


@pytest.mark.parametrize(
    ("precision_field", "offload_field"),
    [
        ("transformer_precision", "transformer_cpu_offload"),
        ("text_encoder_precision", "text_encoder_cpu_offload"),
    ],
)
def test_quantized_components_reject_cpu_offload(precision_field: str, offload_field: str) -> None:
    with pytest.raises(ValueError, match="offload"):
        QwenConfig(**BASE, **{precision_field: "nf4", offload_field: True})


class _FakeTorch:
    float32 = "float32"
    bfloat16 = "bfloat16"
    float16 = "float16"


class _FakeBnb:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_quantization_configs_are_explicit() -> None:
    int8 = QwenBackend._quant_config(_FakeBnb, "int8", _FakeTorch)
    nf4 = QwenBackend._quant_config(_FakeBnb, "nf4", _FakeTorch)
    assert int8.kwargs == {"load_in_8bit": True}
    assert nf4.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }
    assert QwenBackend._quant_config(_FakeBnb, "fp16", _FakeTorch) is None


def test_quantized_components_are_placed_at_load_time() -> None:
    assert QwenBackend._quantized_device_map("nf4", "cuda") == {"": "cuda"}
    assert QwenBackend._quantized_device_map("int8", "cuda:1") == {"": "cuda:1"}
    assert QwenBackend._quantized_device_map("bf16", "cuda") is None


def test_quantized_component_is_not_moved_or_cpu_offloaded() -> None:
    class Component:
        def to(self, device):
            raise AssertionError("quantized component must not be moved with .to()")

    def offload(*args, **kwargs):
        raise AssertionError("quantized component must not use accelerate.cpu_offload")

    QwenBackend._place_component(
        Component(),
        precision="nf4",
        cpu_offload_enabled=False,
        cpu_offload_fn=offload,
        device="cuda",
    )


def test_unquantized_component_placement_paths() -> None:
    calls = []

    class Component:
        def to(self, device):
            calls.append(("to", device))

    component = Component()
    QwenBackend._place_component(
        component,
        precision="bf16",
        cpu_offload_enabled=False,
        cpu_offload_fn=None,
        device="cuda",
    )
    QwenBackend._place_component(
        component,
        precision="bf16",
        cpu_offload_enabled=True,
        cpu_offload_fn=lambda model, execution_device: calls.append(
            ("offload", model, execution_device)
        ),
        device="cuda",
    )
    assert calls == [("to", "cuda"), ("offload", component, "cuda")]


def test_constructor_does_not_import_or_load_heavy_dependencies(tmp_path: Path) -> None:
    backend = QwenBackend(QwenConfig(**{**BASE, "cache_dir": tmp_path}))
    assert backend._pipe is None
    backend.close()


def test_generate_uses_qwen_image_21_call_signature() -> None:
    calls = []

    class FakeGenerator:
        def __init__(self, *, device):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        class OutOfMemoryError(RuntimeError):
            pass

        Generator = FakeGenerator
        cuda = FakeCuda()

    class Result:
        images = ["generated-image"]

    class StrictQwen21Pipe:
        # Mirrors QwenImage21Pipeline's relevant keyword-only surface. In
        # particular, guidance_scale is intentionally not accepted.
        def __call__(
            self,
            *,
            image,
            height,
            width,
            prompt,
            generator,
            num_inference_steps,
            true_cfg_scale,
            negative_prompt=None,
        ):
            calls.append(locals())
            return Result()

    class InputImage:
        width = 64
        height = 64

    backend = QwenBackend(QwenConfig(**BASE, true_cfg_scale=1.0))
    backend._torch = FakeTorch
    backend._pipe = StrictQwen21Pipe()
    input_image = InputImage()
    output = backend.generate(input_image, "edit", "system", 7)

    assert output.image == "generated-image"
    assert calls[0]["prompt"] == "system\n\nedit"
    assert calls[0]["true_cfg_scale"] == 1.0
    assert calls[0]["height"] == 64
    assert calls[0]["width"] == 64
    assert calls[0]["generator"].seed == 7


def test_generate_rejects_dimensions_not_divisible_by_32() -> None:
    class InvalidImage:
        width = 65
        height = 64

    backend = QwenBackend(QwenConfig(**BASE))
    backend._pipe = object()
    backend._torch = object()
    with pytest.raises(SampleModelError, match="dataset loader"):
        backend.generate(InvalidImage(), "edit", "", None)
