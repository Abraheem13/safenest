"""Reproduce every result, table, figure and quoted number with one command.

    python3 scripts/reproduce.py              # everything (needs the [real] extra and the corpora)
    python3 scripts/reproduce.py --synthetic  # Experiments 1-12 only (NumPy only, about 2 min)
    python3 scripts/reproduce.py --check      # only compare the outputs on disk with the commit

The script runs the unit tests, verifies the corpora against their recorded
checksums, extracts features and sentence embeddings where they are missing,
runs the experiments, regenerates the tables, figures and quoted numbers, and
finally compares every regenerated output with the version committed at HEAD.

Everything is deterministic given the master seed except the latency
measurements of Experiment 7, which depend on the machine; those, the
provenance block of each result file and the commit identifier are the only
values allowed to differ. Any other difference is reported and makes the
script exit with status 1.

Run it from a clean working tree: the number generator refuses results that
were produced from uncommitted code.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

#: Result fields that measure the machine rather than the method.
MACHINE_DEPENDENT = {
    "exp07_overhead": {"platform", "processor", "measurements_seconds",
                       "synchronous_per_response_seconds", "share_of_500ms_llm",
                       "share_of_2000ms_llm"},
}
#: Quoted numbers that are latencies, or the commit identifier.
MACHINE_DEPENDENT_VALUES = {"commit", "engineUs", "updateUs", "featMs", "embMs", "inputsShare"}
#: Generated tables that print latencies.
MACHINE_DEPENDENT_TABLES = {"tab_overhead.tex"}
SYNTHETIC = [f"exp{n:02d}" for n in range(1, 13)]


def step(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def run(*args: str) -> None:
    code = subprocess.run([PY, *args], cwd=ROOT).returncode
    if code != 0:
        sys.exit(f"failed: {' '.join(args)} (exit status {code})")


def committed(path: str) -> str | None:
    out = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def differences(a, b, path: str = "") -> list[str]:
    """Paths at which two JSON values differ (floats compared to 1e-12)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k} (present in one version only)")
            else:
                out += differences(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def check(prefixes: list[str] | None) -> int:
    """Compare regenerated outputs with HEAD; return the number of unexpected differences."""
    step("Comparing regenerated outputs with the committed versions")
    bad = 0
    results = sorted((ROOT / "results").glob("exp*.json"))
    if prefixes:
        results = [p for p in results if p.name[:5] in prefixes]
    for p in results:
        old = committed(f"results/{p.name}")
        if old is None:
            print(f"  new      results/{p.name} (not committed yet)")
            continue
        a, b = json.loads(old), json.loads(p.read_text())
        skip = {"provenance"} | MACHINE_DEPENDENT.get(p.stem, set())
        diff = differences({k: v for k, v in a.items() if k not in skip},
                           {k: v for k, v in b.items() if k not in skip})
        print(f"  {'same    ' if not diff else 'DIFFERS '} results/{p.name}")
        for d in diff[:5]:
            print(f"      {d}")
        bad += bool(diff)
    if prefixes:  # tables and numbers need every experiment
        return bad
    for p in sorted((ROOT / "tex").glob("tab_*.tex")):
        old = committed(f"tex/{p.name}")
        if p.name in MACHINE_DEPENDENT_TABLES or old is None:
            continue
        same = old == p.read_text()
        print(f"  {'same    ' if same else 'DIFFERS '} tex/{p.name}")
        bad += not same
    for p in sorted((ROOT / "figures").glob("fig_*.pdf")):
        out = subprocess.run(["git", "show", f"HEAD:figures/{p.name}"], cwd=ROOT,
                             capture_output=True)
        if out.returncode != 0:
            continue
        same = out.stdout == p.read_bytes()
        print(f"  {'same    ' if same else 'DIFFERS '} figures/{p.name}")
        bad += not same
    old = committed("tex/numbers.json")
    if old is not None:
        a, b = json.loads(old), json.loads((ROOT / "tex" / "numbers.json").read_text())
        diff = [k for k in sorted(set(a) | set(b))
                if k not in MACHINE_DEPENDENT_VALUES and a.get(k) != b.get(k)]
        timing = [k for k in sorted(MACHINE_DEPENDENT_VALUES) if a.get(k) != b.get(k)]
        print(f"  {'same    ' if not diff else 'DIFFERS '} tex/numbers.json "
              f"({len(b) - len(MACHINE_DEPENDENT_VALUES)} values checked; "
              f"machine-dependent values that changed: {', '.join(timing) or 'none'})")
        for k in diff:
            print(f"      {k}: {a.get(k)} vs {b.get(k)}")
        bad += bool(diff)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", action="store_true", help="Experiments 1-12 only")
    ap.add_argument("--check", action="store_true", help="only compare outputs with HEAD")
    args = ap.parse_args()

    if args.check:
        bad = check(SYNTHETIC if args.synthetic else None)
        print("\nAll outputs match." if not bad else f"\n{bad} output(s) differ.")
        return 1 if bad else 0

    step("Unit tests and invariant verification")
    run("-m", "pytest", "-q")

    if args.synthetic:
        step("Experiments 1-12")
        run("experiments/run_all.py", "--synthetic")
        bad = check(SYNTHETIC)
        print("\nAll outputs match." if not bad else f"\n{bad} output(s) differ.")
        return 1 if bad else 0

    step("Corpora: download where possible and verify checksums")
    run("scripts/download_data.py")
    from safenest.data import available  # noqa: PLC0415  (needs the [real] extra)
    from safenest.data.registry import PROCESSED  # noqa: PLC0415

    present = available()
    missing = [c for c in ("persuade", "asap", "ellipse", "gillam", "enni") if c not in present]
    if missing:
        print(f"  not present: {', '.join(missing)}; their experiments will record a skip")
    todo = [c for c in present if not (PROCESSED / f"{c}_w50.parquet").exists()]
    if todo:
        step(f"Features for {', '.join(todo)}")
        run("scripts/prepare_features.py", *todo)
    if {"persuade", "asap", "ellipse"} <= set(present) and not (
            PROCESSED / "written_w50_minilm.npy").exists():
        step("Sentence embeddings of the essay windows")
        run("scripts/embed_windows.py", "written", "50")

    step("Experiments 1-16")
    run("experiments/run_all.py")

    step("Tables, figures and quoted numbers")
    run("experiments/make_tables.py")
    if shutil.which("pdflatex") and shutil.which("pdftoppm"):
        run("experiments/make_figures.py")
    else:
        print("  pdflatex or pdftoppm not found: figures not rebuilt")
    run("experiments/make_numbers.py")

    bad = check(None)
    print("\nAll outputs match the committed versions." if not bad
          else f"\n{bad} output(s) differ from the committed versions.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
