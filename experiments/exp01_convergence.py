"""Experiment 01 -- Bayesian estimator convergence.

Also produces the confusion matrix at n = 10 and reports, for each
tier, the first milestone at which accuracy exceeds 90%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS, pct, rng_for, save, table  # noqa: E402
from safenest.estimator import BayesianAgeEstimator, EstimatorState  # noqa: E402
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.tiers import ALL_TIERS, K_TIERS  # noqa: E402

N_TRIALS = 500
MILESTONES = (1, 3, 5, 7, 10, 12, 15)


def run(n_trials: int = N_TRIALS) -> dict:
    rng = rng_for("convergence")
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))

    accuracy: dict[str, dict[int, float]] = {}
    confusion = np.zeros((K_TIERS, K_TIERS))
    confusion_n1 = np.zeros((K_TIERS, K_TIERS))
    confusion_n3 = np.zeros((K_TIERS, K_TIERS))

    for tier in ALL_TIERS:
        hits = {n: 0 for n in MILESTONES}
        for _ in range(n_trials):
            # One trajectory per trial, sampled once and read at each milestone,
            # so the milestones are paired rather than independent runs.
            s = EstimatorState()
            last = None
            for n in range(1, max(MILESTONES) + 1):
                signals = est.model.sample(tier, rng)
                last = signals["linguistic"]
                est.observe(s, signals, rng)
                if n in MILESTONES:
                    got = est.resolve(s, linguistic=last)
                    hits[n] += got is tier
                    if n == 10:
                        confusion[int(tier) - 1, int(got) - 1] += 1
                    if n == 1:
                        confusion_n1[int(tier) - 1, int(got) - 1] += 1
                    if n == 3:
                        confusion_n3[int(tier) - 1, int(got) - 1] += 1
        accuracy[tier.label] = {n: hits[n] / n_trials for n in MILESTONES}

    rows = []
    first90 = {}
    for tier in ALL_TIERS:
        acc = accuracy[tier.label]
        row = {"Tier": f"{tier.label} ({_ages(tier)})"}
        for n in MILESTONES:
            row[f"n={n}"] = pct(acc[n])
        reached = [n for n in MILESTONES if acc[n] >= 0.90]
        first90[tier.label] = reached[0] if reached else None
        row["first>=90%"] = str(first90[tier.label])
        rows.append(row)

    table(rows, ["Tier", *[f"n={n}" for n in MILESTONES], "first>=90%"],
          "Classification accuracy (%), CORPUS-DP mode")

    conf_norm = confusion / confusion.sum(axis=1, keepdims=True)
    conf_n1 = confusion_n1 / confusion_n1.sum(axis=1, keepdims=True)
    conf_n3 = confusion_n3 / confusion_n3.sum(axis=1, keepdims=True)
    # Direction of error at n = 1: assignment above the true tier is
    # under-protection; below it (including the t1 floor) is over-restriction.
    upward = float(np.triu(confusion_n1, 1).sum() / confusion_n1.sum())
    downward = float(np.tril(confusion_n1, -1).sum() / confusion_n1.sum())
    print("\nFigure 6 -- confusion matrix at n=10 (row = true tier)")
    print("        " + "  ".join(t.label.rjust(6) for t in ALL_TIERS))
    for i, t in enumerate(ALL_TIERS):
        print(f"  {t.label}  " + "  ".join(f"{conf_norm[i, j]:6.3f}" for j in range(K_TIERS)))

    return {
        "n_trials": n_trials,
        "milestones": list(MILESTONES),
        "accuracy": accuracy,
        "first_milestone_above_90": first90,
        "confusion_at_n10": conf_norm.tolist(),
        "confusion_at_n1": conf_n1.tolist(),
        "confusion_at_n3": conf_n3.tolist(),
        "n1_error_upward_share_of_sessions": upward,
        "n1_error_downward_share_of_sessions": downward,
    }


def _ages(tier) -> str:
    from safenest.tiers import TIER_SPECS

    s = TIER_SPECS[tier]
    return f"{s.age_low}-{s.age_high}"


if __name__ == "__main__":
    save("exp01_convergence", run())
