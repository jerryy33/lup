import gc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from lup.load import get_layers, load, num_layers

GPT2 = "openai-community/gpt2"
HAS_CUDA = torch.cuda.is_available()
NEEDS_CUDA = [pytest.mark.gpu, pytest.mark.skipif(not HAS_CUDA, reason="no CUDA")]


@dataclass(slots=True)
class Target:
    name: str
    device: str
    upto: int
    has_tok: bool


@dataclass(slots=True)
class Stats:
    params: int
    bytes: int
    alloc: int
    peak: int
    secs: float


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("tiny", id="tiny-cpu"),
        pytest.param("gpt2", id="gpt2-gpu", marks=NEEDS_CUDA),
    ],
)
def target(request: pytest.FixtureRequest, tiny_dir: str) -> Target:
    match request.param:
        case "tiny":
            return Target(tiny_dir, "cpu", upto=2, has_tok=False)
        case _:
            return Target(GPT2, "cuda", upto=3, has_tok=True)


def param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def free(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def measure(target: Target, **kwargs: Any) -> Stats:
    free(target.device)
    cuda = target.device == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    start = perf_counter()
    model, _ = load(target.name, tokenizer=False, device=target.device, **kwargs)
    if cuda:
        torch.cuda.synchronize()
    secs = perf_counter() - start
    nbytes = param_bytes(model)
    stats = Stats(param_count(model), nbytes, nbytes, nbytes, secs)
    if cuda:
        stats.alloc = torch.cuda.memory_allocated() - base
        stats.peak = torch.cuda.max_memory_allocated() - base
    del model
    free(target.device)
    return stats


@pytest.fixture(scope="module")
def stats(target: Target) -> dict[str, Stats]:
    measure(target)
    out = {
        "full": measure(target),
        "upto": measure(target, upto=target.upto),
        "upto+head": measure(target, upto=target.upto, lm_head=True),
    }
    print(f"\n[{target.name} on {target.device}, upto={target.upto}]")
    print(f"{'variant':<10} {'params':>12} {'alloc MB':>10} {'peak MB':>10} {'load s':>8}")
    for key, s in out.items():
        mb, peak_mb = s.alloc / 2**20, s.peak / 2**20
        print(f"{key:<10} {s.params:>12,} {mb:>10.1f} {peak_mb:>10.1f} {s.secs:>8.3f}")
    return out


@contextmanager
def capture(layer: nn.Module, out: list[torch.Tensor]) -> Iterator[None]:
    def hook(module: nn.Module, args: tuple, output: Any) -> None:
        out.append(output[0] if isinstance(output, tuple) else output)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def layer_output(model: PreTrainedModel, idx: int, ids: torch.Tensor) -> torch.Tensor:
    out: list[torch.Tensor] = []
    with capture(get_layers(model)[idx], out), torch.no_grad():
        model(ids)
    return out[0]


def test_variants(target: Target) -> None:
    full, _ = load(target.name, tokenizer=False, device=target.device)
    n = num_layers(full)
    base, _ = load(target.name, tokenizer=False, device=target.device, upto=target.upto)
    head, _ = load(
        target.name, tokenizer=False, device=target.device, upto=target.upto, lm_head=True
    )
    assert target.upto < n
    assert hasattr(full, "lm_head")
    assert num_layers(base) == target.upto
    assert not hasattr(base, "lm_head")
    assert num_layers(head) == target.upto
    assert hasattr(head, "lm_head")
    for model in (full, base, head):
        assert not model.training
        assert not any(p.requires_grad for p in model.parameters())
        assert model.config.use_cache is False


def test_matches_full_forward(target: Target) -> None:
    kw = {"tokenizer": False, "device": target.device, "dtype": torch.float32}
    full, _ = load(target.name, **kw)
    base, _ = load(target.name, upto=target.upto, **kw)
    head, _ = load(target.name, upto=target.upto, lm_head=True, **kw)
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (2, 8), device=target.device)
    last = target.upto - 1
    ref = layer_output(full, last, ids)
    torch.testing.assert_close(layer_output(base, last, ids), ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(layer_output(head, last, ids), ref, atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        base_hs = base(ids).last_hidden_state
        head_out = head(ids, output_hidden_states=True)
        torch.testing.assert_close(head_out.hidden_states[-1], base_hs, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            head_out.logits, head.lm_head(head_out.hidden_states[-1]), atol=1e-4, rtol=1e-4
        )


def test_param_savings(target: Target, stats: dict[str, Stats]) -> None:
    full, _ = load(target.name, tokenizer=False, device="cpu")
    layer_params = param_count(get_layers(full)[0])
    dropped = (num_layers(full) - target.upto) * layer_params
    assert stats["upto+head"].params == stats["full"].params - dropped
    assert stats["upto"].params <= stats["upto+head"].params < stats["full"].params


def test_memory_savings(target: Target, stats: dict[str, Stats]) -> None:
    assert stats["upto"].alloc <= stats["upto+head"].alloc < stats["full"].alloc
    assert stats["upto"].peak <= stats["upto+head"].peak < stats["full"].peak
    assert stats["upto+head"].alloc >= stats["upto+head"].bytes
    assert stats["upto+head"].alloc <= 1.05 * stats["upto+head"].bytes


def test_layer_types_trimmed(tiny_dir: str) -> None:
    model, _ = load(tiny_dir, tokenizer=False, device="cpu", upto=2)
    assert len(model.config.layer_types) == 2
    assert model.config.num_hidden_layers == 2


@pytest.mark.parametrize("upto", [0, 7])
def test_invalid_upto(tiny_dir: str, upto: int) -> None:
    with pytest.raises(ValueError, match="upto must be"):
        load(tiny_dir, tokenizer=False, device="cpu", upto=upto)


def test_device_and_device_map_exclusive(tiny_dir: str) -> None:
    with pytest.raises(ValueError, match="either device or device_map"):
        load(tiny_dir, tokenizer=False, device="cpu", device_map="auto")


def test_tokenizer(target: Target) -> None:
    _, tok = load(target.name, tokenizer=False, device=target.device, upto=1)
    assert tok is None
    if target.has_tok:
        _, tok = load(target.name, device=target.device, upto=1)
        assert tok is not None
        assert tok("hello world")["input_ids"]
