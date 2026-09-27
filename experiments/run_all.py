"""Run every experiment and write each result to results/.

    python3 experiments/run_all.py              # all experiments
    python3 experiments/run_all.py --synthetic  # Experiments 1-12 only (NumPy only)
    python3 experiments/run_all.py --real       # Experiments 13-16 only

Experiments 1-12 exercise the specification in simulation and need NumPy only.
Experiments 13-16 use real corpora (see data/README.md); each records a
`skipped` result when a corpus it needs has not been prepared. Every experiment
is seeded from `common.MASTER_SEED`.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import save  # noqa: E402

EXPERIMENTS = [
    ("exp01_convergence", "Bayesian estimator convergence"),
    ("exp02_separability", "Signal separability and D_min reconciliation"),
    ("exp03_socratic", "Optimal Socratic policy and reward sensitivity"),
    ("exp04_comparative", "Comparative evaluation"),
    ("exp05_privacy", "Privacy/utility and the formal DP statement"),
    ("exp06_populations", "Simulated atypical signal profiles"),
    ("exp07_overhead", "Computational overhead"),
    ("exp08_ceiling", "Achievable ceiling on the Developmental Safety Rate"),
    ("exp09_operating", "Operating characteristics of gamma and the severity gate"),
    ("exp10_bypass", "Bypass detection against a graded impersonation adversary"),
    ("exp11_reliability", "Seed variance, calibration, multiplicity, symmetry"),
    ("exp12_ablation", "Component ablation"),
    ("exp13_real_estimation", "Learned tier estimation on three essay corpora"),
    ("exp14_real_privacy", "Private release of parameters learned from real text"),
    ("exp15_real_subgroups", "Real subgroups: English learners, disability, economic status"),
    ("exp16_real_spoken", "The youngest tiers from children's speech (CHILDES)"),
]
SYNTHETIC = {name for name, _ in EXPERIMENTS if int(name[3:5]) <= 12}


def main(argv: list[str]) -> int:
    selected = [(n, d) for n, d in EXPERIMENTS
                if not argv
                or ("--synthetic" in argv and n in SYNTHETIC)
                or ("--real" in argv and n not in SYNTHETIC)]
    failures = []
    started = time.time()
    for name, description in selected:
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
    print(f"Completed {len(selected) - len(failures)}/{len(selected)} "
          f"experiments in {time.time() - started:.1f}s")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
