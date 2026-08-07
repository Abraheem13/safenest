"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

#: Master seed. Every experiment derives its own stream from this, so the whole
#: results set is reproducible from this one number.
MASTER_SEED = 20260806


def rng_for(experiment: str) -> np.random.Generator:
    """A named, independent random stream."""
    return np.random.default_rng(
        [MASTER_SEED, int.from_bytes(experiment.encode()[:8].ljust(8, b"\0"), "little")]
    )


def provenance() -> dict:
    """Environment metadata recorded with every result file."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = None
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": commit,
        "master_seed": MASTER_SEED,
    }


def save(name: str, payload: dict) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"provenance": provenance(), **payload}, indent=2, default=str))
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def table(rows: list[dict], columns: list[str], title: str = "") -> str:
    """Render rows as a fixed-width table for the console and the log."""
    if title:
        print(f"\n{title}")
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append("  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in columns))
    out = "\n".join(lines)
    print(out)
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}"
