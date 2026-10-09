# lup

Shared code library for interpretability projects. Python 3.14, managed with uv.

## Install

Add it to a project as a git dependency, pinned to a release tag:

```bash
uv add "lup @ git+https://github.com/jerryy33/lup.git@v0.1.0"
uv add "lup[plot] @ git+https://github.com/jerryy33/lup.git@v0.1.0"   # with matplotlib
```

Tested with Python 3.14, torch 2.14.1, transformers 5.19.0 and matplotlib 3.11.2 (see `uv.lock`).
Torch comes from the default PyPI index; the CUDA index is configured in the downstream project.

```python
import torch
from lup.hooks import hook
from lup.load import load
from lup.repro import setup_run


def observe(module, args, output):
    return None


model, tok = load("openai-community/gpt2", upto=6)  # first 6 layers, no LM head
ids = tok("Hello world", return_tensors="pt")["input_ids"].to(model.device)
with torch.no_grad(), hook(model, "h.3", observe) as outs:  # outs: what the hook returned
    model(ids)
run = setup_run("probe", {"layers": 6}, seed=0, level="standard")  # runs/<date>_probe/
```

```python
from matplotlib.figure import Figure
from lup.plot import figsize, save, use_style

use_style("icml")  # base style + venue font and size + default palette
fig = Figure(figsize=figsize("icml"))
save(fig, "demo", final=True)  # figs/demo.pdf
```

## Modules

- `lup.load`: load any HF model and tokenizer (bf16 default, cuda/mps/cpu or `device_map`),
  optionally truncated to its first `upto` layers, with or without the LM head. `get_layers` and
  `num_layers` find the decoder layers of any architecture.
- `lup.hooks`: `hook`, one context manager that registers a forward, forward-pre or backward hook
  on a module path, always removes it, and yields what the hook function returned.
- `lup.repro`: `set_repro(seed, level)` with levels `fast` (default), `standard` and `extreme`
  (raises unless everything is deterministic), explicit generators with independent streams,
  `check_repro`, DataLoader seeding, and run directories with `args.json`, `.git_rev` and
  `meta.json` (`setup_run`).
- `lup.plot`: `use_style(venue, palette="kiki-medium")` applies the base style (`paper.mplstyle`,
  LaTeX on), a venue layer (body font, figure font size, default figure size and constrained
  layout for ICML, ICLR, NeurIPS, ACL, EMNLP, COLM, TMLR, Nature) and one of 24 color palettes (7 colors
  each, ordered so neighboring series are far apart; color and linestyle cycle plus the matching
  sequential and diverging colormaps). Everything is also
  available as named matplotlib styles, e.g. `plt.style.use(["paper", "icml", "kiki-medium"])`;
  importing the module changes nothing. Venue fonts: Times for ICML, ICLR, NeurIPS, ACL and EMNLP,
  Palatino for COLM, Latin Modern for TMLR, Helvetica for Nature. Also `figsize`, `save` and the
  helpers `div_cmap`, `cycle`, `headroom`. Needs the `lup[plot]` extra. `demo/plot_demo.py` renders
  `demo/plot_demo.pdf`, a 12-page showcase of the default style with the ACL venue (plain
  matplotlib calls plus the manual steps below; not part of the package). `demo/palette_demo.py`
  renders one figure per palette into `demo/palettes/`.

## Plotting best practices

The style does most of the work, but some things are manual:

- Call `use_style(venue)` once before creating any figure (figure settings are read when a figure
  is created). Pass `usetex=False` on machines without LaTeX; the venue fonts then fall back to
  matplotlib's serif/sans-serif.
- Size every figure with `figsize(venue, span, frac, ratio)` (`span="col"` or `"text"`, `frac=0.5`
  for side-by-side figures) and include it at 100% width, never rescaled. The venue layer only sets
  the default one-column size.
- All text is rendered by LaTeX: use raw strings, escape `_`, `%`, `&`, `#`, and put math in `$...$`.
  The LaTeX packages `times`, `palatino`, `lmodern`, `helvet` and `sfmath` must be installed.
- Turn the grid off on heatmaps and images (`ax.grid(False)`). The base style draws a grid and
  `imshow` images sit below it. Do not turn it off anywhere else.
- Patches have no edge by default (`patch.force_edgecolor` is off). Add the white edge by hand to
  bars, histograms, pies and areas: `edgecolor="white", linewidth=0.5`. Hatches use the edge color.
- Legends have no frame. When one overlaps the data, place it explicitly: `loc=` and `ncols=`, outside
  with `loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0`, or make room with
  `headroom(ax, 0.25)`. Avoid wide multi-column legends in one-column figures.
- Markers are not in the color/linestyle cycle (a marker on every point is overwhelming on dense
  lines). Pass `marker=` (see `MARKERS` for the order) and `markevery=` yourself when you want them.
  Do not put `markevery` into a property cycle: it breaks `Axes.fill`.
- A twin axis restarts the color cycle: give it an explicit color, e.g. `color=cycle("color")[2]`.
  Take colors for fit lines and reference series from `cycle("color")` too.
- Every palette has 7 colors; the cycle (colors and linestyles) repeats from the 8th series.
  The dark and light palettes are low-contrast by design: neighbors are kept apart by their order,
  so avoid skipping around the cycle.
- Signed data (activation patching effects, correlations): `cmap=div_cmap()` with a symmetric
  `vmin`/`vmax`. Sequential data needs no `cmap`; it follows the palette.
- Do not set `layout=` or call `tight_layout`: constrained layout is on once a venue is set. Do not
  pass `bbox_inches` to `savefig`; use `save(fig, name, final=False)` (PNG for exploratory, vector
  PDF for final).

## Limitations

- `upto` and `get_layers` need a layer list whose length equals the text config's
  `num_hidden_layers`. Checked on default configs of all causal-LM architectures in transformers
  5.19; these do not work: XLM (several lists, pass `path=`), Blenderbot (config count differs),
  LongcatFlash, HRM, xLSTM, BLT (no `num_hidden_layers`), Gemma3n (KV sharing) and ProphetNet. The
  decoder-only Bart family (Bart, BigBirdPegasus, BlenderbotSmall, Marian, MBart, Mvp, Pegasus,
  PLBart, Whisper) is detected and `load(upto=...)` raises instead of silently not truncating.
- In a truncated model the last `hidden_states` entry is post final norm; hook the layer output
  for exact activations.
- `extreme` makes results reproducible on the same hardware and library versions only. Hardware,
  library versions and batch size still change low bits. It sets one CPU thread (slow on CPU).
- `hook` keeps every value the hook function returns; detach and move activations to CPU inside it.
- Venue fonts need the LaTeX packages listed above. COLM's math font is unverified (left as
  Computer Modern); the venue widths for NeurIPS and COLM were read from template copies, not the
  official style zips.

## TODO

- [ ] Add HF monkey patching once it is no longer experimental. It lets us overwrite the forward
  pass of a module, which can be more efficient than hooks in some cases:
  https://huggingface.co/docs/transformers/en/monkey_patching
- [ ] Optional: `lup.patch`, batched "run from layer L" utilities for activation patching, if
  needed.
- [ ] Needs testing before relying on it: multi-GPU `device_map="auto"`, sharded checkpoints (does
  HF skip shard files that only hold dropped layers?), `mps`, multimodal models, and `extreme` on
  different GPU types. The GPU tests only cover gpt2 on a single GPU.
- [ ] Optional: upper bound on `transformers` (`get_layers` and `truncate_config` depend on 5.x
  internals).
