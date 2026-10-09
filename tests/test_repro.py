import json
import os
import random
import subprocess
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from lup.repro import (
    DEFAULT_THREADS,
    LEVELS,
    check_repro,
    describe,
    loader_kwargs,
    make_run_dir,
    set_repro,
    setup_run,
)


def cuda_not_initialized() -> bool:
    return False


@pytest.fixture(autouse=True)
def reset_flags(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(torch.cuda, "is_initialized", cuda_not_initialized)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    yield
    set_repro(0, "standard")


class RandomData(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> torch.Tensor:
        values = [random.random(), float(np.random.rand()), torch.rand(1).item()]
        return torch.tensor(values)


def draws() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), torch.rand(1).item()


def test_reseeding_reproduces_all_global_streams() -> None:
    set_repro(7)
    first = draws()
    set_repro(7)
    assert draws() == first
    set_repro(8)
    assert draws() != first


def test_rngs_numpy_matches_default_rng() -> None:
    rngs = set_repro(3)
    assert rngs.np.random(4).tolist() == np.random.default_rng(3).random(4).tolist()


def test_rngs_are_reproducible_and_children_are_independent() -> None:
    a, b = set_repro(5), set_repro(5)
    first = [c.np.random(3).tolist() for c in a.spawn(3)]
    again = [c.np.random(3).tolist() for c in b.spawn(3)]
    assert first == again
    assert len({tuple(x) for x in first}) == 3
    torch_first = [c.torch.initial_seed() for c in set_repro(5).spawn(3)]
    assert torch_first == [c.torch.initial_seed() for c in set_repro(5).spawn(3)]
    assert len(set(torch_first)) == 3


def test_generator_on_device_is_reproducible() -> None:
    rngs = set_repro(2)
    assert (
        torch.rand(3, generator=rngs.generator()).tolist()
        == torch.rand(3, generator=rngs.generator()).tolist()
    )


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be"):
        set_repro(seed)


def test_invalid_level() -> None:
    with pytest.raises(ValueError, match="unknown level"):
        set_repro(0, "ludicrous")  # type: ignore[arg-type]


def test_default_level_is_fast() -> None:
    set_repro(0)
    assert describe() == LEVELS["fast"]


@pytest.mark.parametrize("level", ["fast", "standard", "extreme"])
def test_flags_match_level(level: str) -> None:
    set_repro(0, level)  # type: ignore[arg-type]
    assert describe() == LEVELS[level]
    check_repro(level)  # type: ignore[arg-type]


def test_extreme_sets_torch_state() -> None:
    set_repro(0, "extreme")
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_num_threads() == 1


def test_switching_levels_resets_everything() -> None:
    set_repro(0, "extreme")
    set_repro(0, "fast")
    assert describe() == LEVELS["fast"]
    assert torch.get_num_threads() == DEFAULT_THREADS
    assert not torch.are_deterministic_algorithms_enabled()


def test_standard_equals_fresh_process_torch_defaults() -> None:
    code = "import json; from lup.repro import describe; print(json.dumps(describe()))"
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert json.loads(out.splitlines()[-1]) == LEVELS["standard"]


def test_extreme_sets_cublas_env() -> None:
    set_repro(0, "extreme")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("value", [None, "", "random"])
def test_extreme_raises_without_fixed_hash_seed(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("PYTHONHASHSEED")
    else:
        monkeypatch.setenv("PYTHONHASHSEED", value)
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        set_repro(0, "extreme")


def cuda_initialized() -> bool:
    return True


def test_extreme_raises_if_cuda_initialized_without_cublas_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", cuda_initialized)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        set_repro(0, "extreme")


def test_extreme_accepts_initialized_cuda_when_env_already_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", cuda_initialized)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    set_repro(0, "extreme")


@pytest.mark.parametrize("flag", ["cudnn_benchmark", "cudnn_tf32", "deterministic_algorithms"])
def test_check_repro_raises_when_a_flag_drifts(flag: str) -> None:
    set_repro(0, "extreme")
    match flag:
        case "cudnn_benchmark":
            torch.backends.cudnn.benchmark = True
        case "cudnn_tf32":
            torch.backends.cudnn.allow_tf32 = True
        case _:
            torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(RuntimeError, match=flag):
        check_repro("extreme")


def test_check_repro_raises_when_env_is_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    set_repro(0, "extreme")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        check_repro("extreme")


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_loader_is_reproducible_with_workers() -> None:
    def run(seed: int) -> list[list[float]]:
        loader = DataLoader(
            RandomData(),
            batch_size=2,
            num_workers=2,
            multiprocessing_context="fork",
            **loader_kwargs(seed),
        )
        return [batch.tolist() for batch in loader]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_make_run_dir_name_and_collisions(tmp_path: Path) -> None:
    first = make_run_dir("probe", tmp_path)
    second = make_run_dir("probe", tmp_path)
    third = make_run_dir("probe", tmp_path)
    base = f"{date.today().isoformat()}_probe"
    assert [p.name for p in (first, second, third)] == [base, f"{base}_2", f"{base}_3"]
    assert all(p.is_dir() for p in (first, second, third))


@pytest.mark.parametrize("name", ["", "a/b", ".hidden"])
def test_make_run_dir_rejects_bad_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="invalid run name"):
        make_run_dir(name, tmp_path)


def rev_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rev"
    path.write_text(text)
    return path


def test_setup_run_writes_everything(tmp_path: Path) -> None:
    args = {"lr": 0.1, "out": Path("x"), "layers": [1, 2]}
    run = setup_run(
        "demo", args, 4, "standard", root=tmp_path / "runs", rev_file=rev_file(tmp_path, "abc123")
    )
    assert (run.dir / "args.json").exists()
    assert json.loads((run.dir / "args.json").read_text()) == {
        "lr": 0.1,
        "out": "x",
        "layers": [1, 2],
    }
    assert (run.dir / ".git_rev").read_text().strip() == "abc123"
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta == json.loads(json.dumps(run.meta))
    assert meta["seed"] == 4
    assert meta["level"] == "standard"
    assert meta["flags"] == LEVELS["standard"]
    assert meta["git_rev"] == "abc123"
    assert meta["versions"]["torch"] == torch.__version__
    assert meta["versions"]["lup"] is not None
    for key in ("started", "host", "python", "platform", "argv", "gpus", "env"):
        assert key in meta
    assert draws() != draws()


def test_setup_run_default_level_is_fast(tmp_path: Path) -> None:
    run = setup_run("demo", {}, 0, root=tmp_path, rev_file=rev_file(tmp_path, "abc"))
    assert run.meta["level"] == "fast"
    assert run.meta["flags"] == LEVELS["fast"]


def test_setup_run_extreme_with_clean_rev(tmp_path: Path) -> None:
    run = setup_run("demo", {}, 0, "extreme", root=tmp_path, rev_file=rev_file(tmp_path, "abc"))
    assert run.meta["env"]["PYTHONHASHSEED"] == "0"
    assert run.meta["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_setup_run_extreme_refuses_dirty_rev_and_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="clean git revision"):
        setup_run("demo", {}, 0, "extreme", root=root, rev_file=rev_file(tmp_path, "abc\ndirty"))
    assert not root.exists()


def test_setup_run_extreme_refuses_unknown_rev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="clean git revision"):
        setup_run("demo", {}, 0, "extreme", root=tmp_path / "runs", rev_file=tmp_path / "nope")


def test_setup_run_extreme_without_hash_seed_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED")
    root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        setup_run("demo", {}, 0, "extreme", root=root, rev_file=rev_file(tmp_path, "abc"))
    assert not root.exists()


def test_setup_run_falls_back_to_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = setup_run("demo", {}, 0, "standard", root=tmp_path, rev_file=tmp_path / "missing")
    rev = run.meta["git_rev"].split()[0]
    assert rev == "unknown" or len(rev) == 40
