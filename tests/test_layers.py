import pytest
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    BloomConfig,
    FalconConfig,
    Gemma2Config,
    Gemma3TextConfig,
    GPT2Config,
    GPTNeoXConfig,
    JambaConfig,
    LlamaConfig,
    MambaConfig,
    MistralConfig,
    OPTConfig,
    Phi3Config,
    PreTrainedConfig,
    PreTrainedModel,
    Qwen2Config,
    Qwen3MoeConfig,
    XLMConfig,
)

from lup.load import get_layers, num_layers, truncate_config

CONFIGS = [
    LlamaConfig,
    GPT2Config,
    GPTNeoXConfig,
    OPTConfig,
    FalconConfig,
    BloomConfig,
    MistralConfig,
    Qwen2Config,
    Phi3Config,
    Gemma2Config,
    Gemma3TextConfig,
    Qwen3MoeConfig,
    MambaConfig,
    JambaConfig,
]


def cfg_id(cfg_cls: type) -> str:
    return cfg_cls.__name__


def build(auto: type, cfg: PreTrainedConfig) -> PreTrainedModel:
    with torch.device("meta"):
        return auto.from_config(cfg)


@pytest.mark.parametrize("cfg_cls", CONFIGS, ids=cfg_id)
@pytest.mark.parametrize("auto", [AutoModel, AutoModelForCausalLM], ids=["base", "causal"])
@pytest.mark.parametrize("upto", [1, 2])
def test_get_layers_after_truncation(cfg_cls: type, auto: type, upto: int) -> None:
    cfg = cfg_cls()
    n = cfg.get_text_config().num_hidden_layers
    assert num_layers(build(auto, cfg)) == n
    truncate_config(cfg, upto)
    assert num_layers(build(auto, cfg)) == upto


def test_ambiguous_architecture_needs_path() -> None:
    model = build(AutoModelForCausalLM, XLMConfig())
    with pytest.raises(ValueError, match="pass path="):
        get_layers(model)
    assert len(get_layers(model, "transformer.ffns")) == model.config.num_hidden_layers
