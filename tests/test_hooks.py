from typing import Any

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from lup.hooks import hook
from lup.load import load


@pytest.fixture(scope="module")
def model(tiny_dir: str) -> PreTrainedModel:
    model, _ = load(tiny_dir, tokenizer=False, device="cpu", dtype=torch.float32, freeze=False)
    return model


@pytest.fixture
def ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 100, (2, 8))


def logits(model: PreTrainedModel, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(ids).logits


def unwrap(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def n_hooks(model: nn.Module) -> int:
    return sum(
        len(m._forward_hooks)
        + len(m._forward_pre_hooks)
        + len(m._backward_hooks)
        + len(m._backward_pre_hooks)
        for m in model.modules()
    )


def observe(module: nn.Module, args: tuple, output: Any) -> None:
    return None


def keep(module: nn.Module, args: tuple, output: Any) -> Any:
    return output


def zero(module: nn.Module, args: tuple, output: Any) -> Any:
    match output:
        case tuple():
            return (torch.zeros_like(output[0]), *output[1:])
        case _:
            return torch.zeros_like(output)


def test_records_returned_value_per_call(model: PreTrainedModel, ids: torch.Tensor) -> None:
    def grab(module: nn.Module, args: tuple, output: Any) -> torch.Tensor:
        return unwrap(output).detach().clone()

    with torch.no_grad(), hook(model, "model.layers.1", grab) as outs:
        hs = model(ids, output_hidden_states=True).hidden_states
        model(ids)
    assert len(outs) == 2
    torch.testing.assert_close(outs[0], hs[2])


def test_observer_does_not_change_output(model: PreTrainedModel, ids: torch.Tensor) -> None:
    ref = logits(model, ids)
    with hook(model, "model.layers.2", observe) as outs:
        out = logits(model, ids)
    assert outs == [None]
    torch.testing.assert_close(out, ref)


def test_returning_own_activation_is_a_no_op(model: PreTrainedModel, ids: torch.Tensor) -> None:
    ref = logits(model, ids)
    with hook(model, "model.layers.2.mlp", keep):
        out = logits(model, ids)
    torch.testing.assert_close(out, ref)


def test_non_none_return_replaces_output(model: PreTrainedModel, ids: torch.Tensor) -> None:
    ref = logits(model, ids)
    with hook(model, "model.layers.0", zero) as outs:
        out = logits(model, ids)
    assert not torch.allclose(out, ref)
    assert torch.count_nonzero(unwrap(outs[0])) == 0


def test_forward_pre_hook_can_replace_input(model: PreTrainedModel, ids: torch.Tensor) -> None:
    def zero_in(module: nn.Module, args: tuple) -> tuple:
        return (torch.zeros_like(args[0]), *args[1:])

    with torch.no_grad(), hook(model, "model.norm", zero_in, kind="forward_pre") as outs:
        hidden = model.model(ids).last_hidden_state
    assert len(outs) == 1
    assert torch.count_nonzero(hidden) == 0


def test_backward_hook_sees_gradients(model: PreTrainedModel, ids: torch.Tensor) -> None:
    def grab(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        return None

    with hook(model, "model.layers.3", grab, kind="full_backward") as outs:
        model(ids).logits.sum().backward()
    model.zero_grad()
    assert len(outs) == 1


def test_backward_pre_hook_can_replace_grad(model: PreTrainedModel, ids: torch.Tensor) -> None:
    def zero_grad_out(module: nn.Module, grad_output: tuple) -> tuple:
        return tuple(torch.zeros_like(g) for g in grad_output)

    with hook(model, "model.layers.3", zero_grad_out, kind="full_backward_pre"):
        model(ids).logits.sum().backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    cut = [
        g
        for name, g in grads.items()
        if name.startswith(tuple(f"model.layers.{i}." for i in range(4)))
    ]
    kept = [g for name, g in grads.items() if name.startswith("model.layers.5.")]
    assert cut and kept
    assert all(torch.count_nonzero(g) == 0 for g in cut)
    assert any(torch.count_nonzero(g) > 0 for g in kept)
    model.zero_grad()


def test_root_path_hooks_the_model_itself(model: PreTrainedModel, ids: torch.Tensor) -> None:
    with hook(model, "", observe) as outs:
        logits(model, ids)
    assert len(outs) == 1


def test_prepend_runs_first(model: PreTrainedModel, ids: torch.Tensor) -> None:
    order: list[str] = []

    def tag(name: str) -> Any:
        def fn(module: nn.Module, args: tuple, output: Any) -> None:
            order.append(name)

        return fn

    with (
        hook(model, "model.layers.0", tag("a")),
        hook(model, "model.layers.0", tag("b"), prepend=True),
    ):
        logits(model, ids)
    assert order == ["b", "a"]


def test_with_kwargs_passthrough(model: PreTrainedModel, ids: torch.Tensor) -> None:
    def fn(module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        return None

    with hook(model, "model.layers.0", fn, with_kwargs=True) as outs:
        logits(model, ids)
    assert len(outs) == 1


@pytest.mark.parametrize("kind", ["forward", "forward_pre", "full_backward", "full_backward_pre"])
def test_hooks_are_removed(model: PreTrainedModel, kind: str) -> None:
    before = n_hooks(model)
    with hook(model, "model.layers.0", observe, kind=kind):  # type: ignore[arg-type]
        assert n_hooks(model) == before + 1
    assert n_hooks(model) == before


def test_hook_is_removed_when_forward_raises(model: PreTrainedModel, ids: torch.Tensor) -> None:
    def boom(module: nn.Module, args: tuple, output: Any) -> None:
        raise RuntimeError("boom")

    before = n_hooks(model)
    with pytest.raises(RuntimeError, match="boom"), hook(model, "model.layers.0", boom):
        model(ids)
    assert n_hooks(model) == before


def test_hook_is_removed_when_body_raises(model: PreTrainedModel) -> None:
    before = n_hooks(model)
    with pytest.raises(KeyError), hook(model, "model.layers.0", observe):
        raise KeyError("x")
    assert n_hooks(model) == before


def test_bad_path_and_kind_register_nothing(model: PreTrainedModel) -> None:
    before = n_hooks(model)
    with pytest.raises(AttributeError), hook(model, "model.nope", observe):
        pass
    with (
        pytest.raises(ValueError, match="unknown hook kind"),
        hook(
            model,
            "model.layers.0",
            observe,
            kind="sideways",  # type: ignore[arg-type]
        ),
    ):
        pass
    assert n_hooks(model) == before
