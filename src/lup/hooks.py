from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from torch import nn

type HookKind = Literal["forward", "forward_pre", "full_backward", "full_backward_pre"]


@contextmanager
def hook(
    model: nn.Module,
    path: str,
    fn: Callable[..., Any],
    kind: HookKind = "forward",
    **register_kwargs: Any,
) -> Iterator[list[Any]]:
    """Register `fn` as a hook on the submodule at `path` and always remove it on exit.

    Args:
        model: root module; `path` is resolved with `model.get_submodule` ("" is the model itself).
        path: dotted module path, e.g. "model.layers.5" or "model.layers.5.mlp".
        fn: a standard PyTorch hook for the chosen `kind`. As in PyTorch, a non-None return
            replaces the module's output (forward), input (forward_pre) or gradients (backward).
        kind: "forward" fn(module, args, output), "forward_pre" fn(module, args),
            "full_backward" fn(module, grad_input, grad_output) or
            "full_backward_pre" fn(module, grad_output).
        **register_kwargs: forwarded to the register call, e.g. `prepend=True`,
            `with_kwargs=True` (forward kinds) or `always_call=True` (forward).

    Yields:
        A list that receives the value `fn` returned on every call (None for pure observers).
        Keep references small, e.g. detach and move to CPU inside `fn`.
    """
    module = model.get_submodule(path)
    outs: list[Any] = []

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        outs.append(out)
        return out

    match kind:
        case "forward":
            handle = module.register_forward_hook(wrapper, **register_kwargs)
        case "forward_pre":
            handle = module.register_forward_pre_hook(wrapper, **register_kwargs)
        case "full_backward":
            handle = module.register_full_backward_hook(wrapper, **register_kwargs)
        case "full_backward_pre":
            handle = module.register_full_backward_pre_hook(wrapper, **register_kwargs)
        case _:
            raise ValueError(f"unknown hook kind {kind!r}")
    try:
        yield outs
    finally:
        handle.remove()
