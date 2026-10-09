from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils import logging as hf_logging

PER_LAYER_FIELDS = ("layer_types", "activation_sparsity_pattern")


def pick_device() -> str:
    """Return "cuda", "mps" or "cpu", whichever is available first."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def quiet_hf() -> Iterator[None]:
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(level)


def truncate_config(cfg: Any, upto: int) -> None:
    text_cfg = cfg.get_text_config()
    n = text_cfg.num_hidden_layers
    if not 1 <= upto <= n:
        raise ValueError(f"upto must be in [1, {n}], got {upto}")
    text_cfg.num_hidden_layers = upto
    for field in PER_LAYER_FIELDS:
        if isinstance(getattr(text_cfg, field, None), list) and is_settable(text_cfg, field):
            setattr(text_cfg, field, getattr(text_cfg, field)[:upto])


def is_settable(cfg: Any, field: str) -> bool:
    prop = getattr(type(cfg), type(cfg).attribute_map.get(field, field), None)
    return not isinstance(prop, property) or prop.fset is not None


def load(
    name: str,
    *,
    upto: int | None = None,
    lm_head: bool = False,
    tokenizer: bool = True,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    use_cache: bool = False,
    freeze: bool = True,
    device_map: str | dict[str, Any] | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
    token: str | None = None,
    config_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    layers_path: str | None = None,
    **hf_kwargs: Any,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase | None]:
    """Load a HF model and tokenizer, optionally truncated to its first `upto` layers.

    Variants:
        upto=None: full AutoModelForCausalLM.
        upto=L: AutoModel with layers 0..L-1 only (no final layers, no LM head).
        upto=L, lm_head=True: AutoModelForCausalLM with layers 0..L-1 plus final norm and LM head.

    Args:
        name: HF hub id or local path.
        upto: number of layers to keep, in [1, num_hidden_layers].
        lm_head: keep the LM head when truncating (ignored if upto is None).
        tokenizer: also load the tokenizer; if False the second return value is None.
        device: "cuda", "mps" or "cpu"; defaults to the best available. Exclusive with device_map.
        dtype: weight dtype.
        use_cache: enable the KV cache (off by default; set True if you want to call generate()).
        freeze: set requires_grad=False on all parameters; use False for fine-tuning.
        device_map: e.g. "auto" to shard across GPUs (pair with max_memory via hf_kwargs).
        trust_remote_code, revision, token: forwarded to config, model and tokenizer.
        config_kwargs: config overrides, forwarded to AutoConfig.from_pretrained.
        tokenizer_kwargs: forwarded to AutoTokenizer.from_pretrained.
        layers_path: dotted path of the layer ModuleList, e.g. "model.layers"; only needed when
            get_layers cannot find it on its own (used to verify the truncation).
        **hf_kwargs: forwarded to the model's from_pretrained.

    Returns:
        (model, tokenizer) with the model in eval mode.

    Note:
        In a truncated model the last entry of `hidden_states` is post final norm, so it does
        not equal the full model's layer-L output; hook the layer output for exact activations.
    """
    if device is not None and device_map is not None:
        raise ValueError("pass either device or device_map, not both")
    if device_map is None:
        device_map = {"": device or pick_device()}

    shared = {"trust_remote_code": trust_remote_code, "revision": revision, "token": token}
    cfg = AutoConfig.from_pretrained(name, **shared, **(config_kwargs or {}))
    cfg.use_cache = use_cache
    cfg.get_text_config().use_cache = use_cache
    if upto is not None:
        truncate_config(cfg, upto)

    auto_cls = AutoModel if upto is not None and not lm_head else AutoModelForCausalLM
    with quiet_hf() if upto is not None else nullcontext():
        model = auto_cls.from_pretrained(
            name, config=cfg, dtype=dtype, device_map=device_map, **shared, **hf_kwargs
        )
    if upto is not None and (kept := num_layers(model, layers_path)) != upto:
        raise ValueError(
            f"truncating to {upto} layers left {kept}; this architecture is unsupported"
        )
    model.eval().requires_grad_(not freeze)

    tok = None
    if tokenizer:
        tok = AutoTokenizer.from_pretrained(name, **shared, **(tokenizer_kwargs or {}))
    return model, tok


def find_lists(root: nn.Module, n: int) -> dict[str, nn.ModuleList]:
    found = {
        name: m
        for name, m in root.named_modules()
        if isinstance(m, nn.ModuleList) and len(m) == n and next(m.parameters(), None) is not None
    }
    return {
        name: m
        for name, m in found.items()
        if not any(name.startswith(f"{other}.") for other in found)
    }


def get_layers(model: PreTrainedModel, path: str | None = None) -> nn.ModuleList:
    """Return the ModuleList of decoder layers of a base or causal-LM model.

    The list is found structurally: the outermost nn.ModuleList with parameters whose length equals
    `config.get_text_config().num_hidden_layers`, searched in `model.get_decoder()` first and
    then in the whole model. Raises if there is not exactly one match (e.g. models with several
    lists of that length or a layer count that differs from the config); pass `path`, a dotted
    module path such as "model.layers", to select the list explicitly.
    """
    if path is not None:
        return model.get_submodule(path)
    n = model.config.get_text_config().num_hidden_layers
    found = find_lists(model.get_decoder(), n) or find_lists(model, n)
    if len(found) != 1:
        raise ValueError(
            f"expected one ModuleList of {n} layers in {type(model).__name__}, found "
            f"{list(found) or 'none'}; pass path= to choose"
        )
    return next(iter(found.values()))


def num_layers(model: PreTrainedModel, path: str | None = None) -> int:
    """Return how many decoder layers the (possibly truncated) model has."""
    return len(get_layers(model, path))
