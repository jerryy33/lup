# Project: shared library for interpretability research code

Project name: lup        (used for remote directories)

## Goal
Code shared by all new interpretability projects: model loading, hooks, reproducibility and
plotting. Experiment-specific code does not belong here. vLLM and the torch CUDA index also stay
out, since they have their own environments.

## Layout
- `src/lup/` - the package: `load.py`, `hooks.py`, `repro.py`, `plot.py`, `paper.mplstyle`
- `tests/` - pytest; tests marked `gpu` need a CUDA GPU
- `demo/` - `plot_demo.py` (output `plot_demo.pdf`) and `palette_demo.py` (output `palettes/`), not part
  of the package

## Commands
- Local tests: `uv run pytest -m "not gpu"`
- Lint and format: `uv run ruff check --fix && uv run ruff format`
- Full suite including GPU tests: `uv run pytest` on a machine with a CUDA GPU

## Models and data
Tests use a tiny random Qwen3 model built on the fly (CPU) and `openai-community/gpt2` (GPU).

## Notes
- `lup.plot` needs matplotlib (`lup[plot]` extra); it is also in the dev group for the tests.
- Plot styling: `src/lup/paper.mplstyle` is the base style; venues and the 24 palettes are data in
  `plot.py`, registered as named matplotlib styles by `use_style`. Importing `lup.plot` must not
  change matplotlib state (tested).
- Manual plotting steps (grid on images, white bar edges, legend placement, markers, ...) are in
  the README "Plotting best practices" and are what `demo/plot_demo.py` follows.
- Each venue layer carries the venue's body font (family and LaTeX preamble); the tests check the
  fonts embedded in a rendered PDF.
