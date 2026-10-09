import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
        use_sliding_window=True,
        sliding_window=8,
        max_window_layers=3,
    )
    path = tmp_path_factory.mktemp("tiny")
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    return str(path)
