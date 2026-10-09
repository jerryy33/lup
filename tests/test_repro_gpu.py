import os
import subprocess
import sys

import pytest
import torch

NEEDS_CUDA = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")]
pytestmark = NEEDS_CUDA

GPT2 = "openai-community/gpt2"

FORWARD = f"""
import hashlib, sys, torch
from lup.load import load
from lup.repro import set_repro

rngs = set_repro(int(sys.argv[1]), "extreme")
model, _ = load("{GPT2}", tokenizer=False, device="cuda")
ids = torch.randint(0, 50000, (4, 64), generator=rngs.generator("cuda"), device="cuda")
with torch.no_grad():
    a = model(ids).logits
    b = model(ids).logits
assert torch.equal(a, b)
print("DIGEST", hashlib.sha256(a.float().cpu().numpy().tobytes()).hexdigest())
"""

CUDA_FIRST = """
import torch
from lup.repro import set_repro
torch.zeros(1, device="cuda")
try:
    set_repro(0, "extreme")
except RuntimeError as err:
    print("RAISED", "CUBLAS_WORKSPACE_CONFIG" in str(err))
else:
    print("NO ERROR")
"""


def run_python(code: str, *args: str, hash_seed: str | None = "0") -> str:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
    }
    if hash_seed is not None:
        env["PYTHONHASHSEED"] = hash_seed
    out = subprocess.run(
        [sys.executable, "-c", code, *args], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_extreme_refuses_when_cuda_already_initialized() -> None:
    assert run_python(CUDA_FIRST) == "RAISED True"


def test_extreme_forward_is_bitwise_reproducible_across_processes() -> None:
    first = run_python(FORWARD, "0")
    second = run_python(FORWARD, "0")
    other = run_python(FORWARD, "1")
    assert first.startswith("DIGEST ")
    assert first == second
    assert first != other
