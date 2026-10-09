import re
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).parent.parent / "demo" / "plot_demo.py"


def test_demo_renders_a_multi_page_pdf_without_latex(tmp_path: Path) -> None:
    out = tmp_path / "demo.pdf"
    result = subprocess.run(
        [sys.executable, str(DEMO), "--no-latex", "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    pages = re.findall(rb"/Type\s*/Page[^s]", out.read_bytes())
    assert len(pages) >= 10


PALETTE_DEMO = Path(__file__).parent.parent / "demo" / "palette_demo.py"


def test_palette_demo_writes_one_pdf_per_palette(tmp_path: Path) -> None:
    names = ["kiki-medium", "marnie-dark1"]
    result = subprocess.run(
        [
            sys.executable,
            str(PALETTE_DEMO),
            "--no-latex",
            "--out",
            str(tmp_path),
            "--palettes",
            *names,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{n}.pdf" for n in names]
    assert all(p.stat().st_size > 0 for p in tmp_path.iterdir())
