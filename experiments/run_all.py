"""Run the full validation suite and write every result to results/.

    python3 experiments/run_all.py

Each experiment is independent and seeded from `common.MASTER_SEED`, so the
whole results set is reproducible from a single number.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import save  # noqa: E402

EXPERIMENTS = [
    ("exp01_convergence", "Bayesian estimator convergence (Table 8, Figure 6)"),
    ("exp02_separability", "Signal separability and D_min reconciliation"),
    ("exp03_socratic", "Optimal Socratic policy and reward sensitivity"),
    ("exp04_comparative", "Comparative evaluation"),
    ("exp05_privacy", "Privacy/utility and the formal DP statement"),
    ("exp06_populations", "Neurodivergent and non-WEIRD populations"),
    ("exp07_overhead", "Computational overhead"),
    ("exp08_ceiling", "Achievable ceiling on the Developmental Safety Rate"),
    ("exp09_operating", "Operating characteristics of gamma and the severity gate"),
    ("exp10_bypass", "Bypass detection against a graded impersonation adversary"),
    ("exp11_reliability", "Seed variance, calibration, multiplicity, symmetry"),
    ("exp12_ablation", "Component ablation"),
]


def main() -> int:
    failures = []
    started = time.time()
    for name, description in EXPERIMENTS:
        print(f"\n{'=' * 78}\n{name}: {description}\n{'=' * 78}")
        t0 = time.time()
        try:
            module = importlib.import_module(f"experiments.{name}")
            save(name, module.run())
            print(f"  [ok] {time.time() - t0:.1f}s")
        except Exception as exc:  # keep going; report at the end
            failures.append((name, repr(exc)))
            print(f"  [FAILED] {exc!r}")
    print(f"\n{'=' * 78}")
    print(f"Completed {len(EXPERIMENTS) - len(failures)}/{len(EXPERIMENTS)} "
          f"experiments in {time.time() - started:.1f}s")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
