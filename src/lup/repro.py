import json
import os
import platform
import random
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

type Level = Literal["fast", "standard", "extreme"]

DEFAULT_THREADS = torch.get_num_threads()
CUBLAS_VALUES = (":4096:8", ":16:8")
MAX_SEED = 2**32
PACKAGES = ("lup", "torch", "transformers", "accelerate", "numpy")

LEVELS: dict[str, dict[str, Any]] = {
    "fast": {
        "deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
        "matmul_precision": "high",
        "cudnn_tf32": True,
        "bf16_reduced_reduction": True,
        "fp16_reduced_reduction": True,
        "cudnn_sdp": True,
        "num_threads": DEFAULT_THREADS,
        "fill_uninitialized": True,
    },
    "standard": {
        "deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "matmul_precision": "highest",
        "cudnn_tf32": True,
        "bf16_reduced_reduction": True,
        "fp16_reduced_reduction": True,
        "cudnn_sdp": True,
        "num_threads": DEFAULT_THREADS,
        "fill_uninitialized": True,
    },
    "extreme": {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_precision": "highest",
        "cudnn_tf32": False,
        "bf16_reduced_reduction": False,
        "fp16_reduced_reduction": False,
        "cudnn_sdp": False,
        "num_threads": 1,
        "fill_uninitialized": True,
    },
}


@dataclass(slots=True)
class Rngs:
    """Explicit random generators derived from one seed, with independent child streams."""

    seq: np.random.SeedSequence
    np: np.random.Generator
    torch: torch.Generator

    @classmethod
    def from_seq(cls, seq: np.random.SeedSequence) -> Rngs:
        gen = torch.Generator().manual_seed(torch_seed(seq))
        return cls(seq, np.random.Generator(np.random.PCG64(seq)), gen)

    def spawn(self, n: int) -> list[Rngs]:
        """Return `n` statistically independent child Rngs (reproducible given the parent)."""
        return [Rngs.from_seq(child) for child in self.seq.spawn(n)]

    def generator(self, device: str = "cpu") -> torch.Generator:
        """Return a torch.Generator on `device` seeded from this Rngs (e.g. "cuda" sampling)."""
        return torch.Generator(device).manual_seed(torch_seed(self.seq))


@dataclass(slots=True)
class Run:
    dir: Path
    rngs: Rngs
    meta: dict[str, Any]


def torch_seed(seq: np.random.SeedSequence) -> int:
    return int(seq.generate_state(1, np.uint64)[0])


def describe() -> dict[str, Any]:
    """Return the current values of every reproducibility and efficiency flag we manage."""
    cuda = torch.backends.cuda
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "bf16_reduced_reduction": cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduced_reduction": cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cudnn_sdp": cuda.cudnn_sdp_enabled(),
        "num_threads": torch.get_num_threads(),
        "fill_uninitialized": torch.utils.deterministic.fill_uninitialized_memory,
    }


def apply_flags(level: Level) -> None:
    flags = LEVELS[level]
    torch.use_deterministic_algorithms(flags["deterministic_algorithms"])
    torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
    torch.set_float32_matmul_precision(flags["matmul_precision"])
    torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flags[
        "bf16_reduced_reduction"
    ]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags[
        "fp16_reduced_reduction"
    ]
    torch.backends.cuda.enable_cudnn_sdp(flags["cudnn_sdp"])
    torch.set_num_threads(flags["num_threads"])
    torch.utils.deterministic.fill_uninitialized_memory = flags["fill_uninitialized"]


def prepare_extreme_env() -> None:
    if os.environ.get("PYTHONHASHSEED") in (None, "", "random"):
        raise RuntimeError(
            "extreme needs a fixed hash seed: start Python with PYTHONHASHSEED=0 (it cannot be "
            "set from inside a running interpreter)"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") in CUBLAS_VALUES:
        return
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "extreme needs CUBLAS_WORKSPACE_CONFIG set before CUDA is initialized: call "
            "set_repro first thing, or export CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_VALUES[0]


def check_repro(level: Level) -> None:
    """Raise RuntimeError unless every managed flag currently has the value `level` requires.

    `extreme` additionally requires PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG to be set. Call it
    again before a long run to catch libraries that changed a flag after `set_repro`.
    """
    now = describe()
    wrong = {k: (now[k], want) for k, want in LEVELS[level].items() if now[k] != want}
    problems = [f"{k}: is {got!r}, {level} needs {want!r}" for k, (got, want) in wrong.items()]
    if level == "extreme":
        if os.environ.get("PYTHONHASHSEED") in (None, "", "random"):
            problems.append("PYTHONHASHSEED is not set to a fixed value")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in CUBLAS_VALUES:
            problems.append(f"CUBLAS_WORKSPACE_CONFIG is not one of {CUBLAS_VALUES}")
    if problems:
        raise RuntimeError(f"not {level}: " + "; ".join(problems))


def set_repro(seed: int, level: Level = "fast") -> Rngs:
    """Seed all libraries and set the reproducibility / efficiency flags of `level`.

    Levels:
        fast: seeds, plus speed flags (cudnn.benchmark, TF32 matmuls). Not reproducible.
        standard: seeds only; every flag at the torch default.
        extreme: seeds, deterministic algorithms (raising on any op without a deterministic
            implementation), cuDNN deterministic, no TF32 or reduced-precision reductions, no
            cuDNN attention, a single CPU thread, CUBLAS_WORKSPACE_CONFIG and a fixed
            PYTHONHASHSEED. Raises RuntimeError if anything cannot be set (see check_repro).

    Every call sets all managed flags, so switching levels fully resets the previous one.
    Results still differ across hardware, library versions and batch sizes.

    Args:
        seed: integer in [0, 2**32).
        level: "fast" (default), "standard" or "extreme".

    Returns:
        Rngs with explicit numpy and torch generators and `spawn` for independent streams.
        The numpy generator equals `np.random.default_rng(seed)`.
    """
    if not 0 <= seed < MAX_SEED:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}, choose from {list(LEVELS)}")
    if level == "extreme":
        prepare_extreme_env()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    apply_flags(level)
    check_repro(level)
    return Rngs.from_seq(np.random.SeedSequence(seed))


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % MAX_SEED
    random.seed(seed)
    np.random.seed(seed)


def loader_kwargs(seed: int) -> dict[str, Any]:
    """Return `worker_init_fn` and `generator` kwargs that make a DataLoader reproducible."""
    return {
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }


def make_run_dir(name: str, root: str | Path = "runs") -> Path:
    """Create and return `<root>/<YYYY-MM-DD>_<name>`, adding `_2`, `_3`, ... if it exists."""
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"invalid run name {name!r}")
    base = f"{date.today().isoformat()}_{name}"
    path = Path(root) / base
    count = 1
    while True:
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            count += 1
            path = Path(root) / f"{base}_{count}"
        else:
            return path


def read_git_rev(rev_file: str | Path = ".git_rev") -> str:
    rev_file = Path(rev_file)
    if rev_file.exists():
        return rev_file.read_text().strip()
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet"]).returncode != 0
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return f"{rev}\ndirty" if dirty else rev


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    out["torch"] = torch.__version__
    return out


def collect_meta(name: str, seed: int, level: Level, git_rev: str) -> dict[str, Any]:
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "name": name,
        "started": datetime.now(UTC).isoformat(),
        "seed": seed,
        "level": level,
        "flags": describe(),
        "env": {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")},
        "git_rev": git_rev,
        "versions": package_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": gpus,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
    }


def setup_run(
    name: str,
    args: Mapping[str, Any],
    seed: int,
    level: Level = "fast",
    *,
    root: str | Path = "runs",
    rev_file: str | Path = ".git_rev",
) -> Run:
    """Create a run directory, apply `set_repro`, and record everything needed to trace the run.

    The directory `<root>/<YYYY-MM-DD>_<name>` receives `args.json` (the resolved arguments),
    `.git_rev` (copied from `rev_file`, or taken from git if that file is missing) and
    `meta.json` (seed, level, all flags, environment, package versions, hardware, host, SLURM
    job id, command line). With level "extreme" the run is refused if the code version is
    unknown or dirty.

    Args:
        name: short run name, used in the directory name.
        args: resolved arguments, e.g. `vars(parsed_args)`; must be JSON-serializable apart from
            paths and other values that `str` can represent.
        seed: seed for `set_repro`.
        level: "fast" (default), "standard" or "extreme".
        root: parent directory of the run directories.
        rev_file: file holding the git revision, for code copied without its .git directory.

    Returns:
        Run with the directory, the Rngs from `set_repro`, and the metadata that was written.
    """
    git_rev = read_git_rev(rev_file)
    if level == "extreme" and (git_rev == "unknown" or "dirty" in git_rev.split()):
        raise RuntimeError(f"extreme needs a known, clean git revision, got {git_rev!r}")
    rngs = set_repro(seed, level)
    run_dir = make_run_dir(name, root)
    meta = collect_meta(name, seed, level, git_rev)
    (run_dir / "args.json").write_text(json.dumps(dict(args), indent=2, default=str))
    (run_dir / ".git_rev").write_text(git_rev + "\n")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return Run(run_dir, rngs, meta)
