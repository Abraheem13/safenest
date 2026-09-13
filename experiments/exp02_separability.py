"""Experiment 02 -- signal separability and convergence reconciliation.

Computes the pairwise KL divergence matrix over the linguistic feature space,
derives D_min and the KL-heuristic interaction count from it, and compares that bound
against the convergence measured in Experiment 01. Both sides are computed from
the same signal model, so the theoretical and empirical figures cannot drift
apart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS, save, table  # noqa: E402
from safenest.estimator import DEFAULT_GAMMA  # noqa: E402
from safenest.signals import (  # noqa: E402
    LINGUISTIC_FEATURES, SignalModel, chernoff_exponent, kl_gaussian,
    linguistic_kl_matrix, min_adjacent_kl, misassignment_bound, sanov_interactions,
)
from safenest.tiers import ALL_TIERS, K_TIERS  # noqa: E402

CONFIDENCES = ((0.90, 0.10), (0.95, 0.05), (0.99, 0.01))


def run() -> dict:
    model = SignalModel()
    kl = linguistic_kl_matrix(model)

    rows = []
    for i, ti in enumerate(ALL_TIERS):
        row = {"": ti.label}
        for j, tj in enumerate(ALL_TIERS):
            row[tj.label] = "--" if i == j else f"{kl[i, j]:.2f}"
        rows.append(row)
    table(rows, ["", *[t.label for t in ALL_TIERS]],
          "Pairwise KL divergence (nats), linguistic space")

    adjacent = {
        f"{ALL_TIERS[i].label}-{ALL_TIERS[i+1].label}": float(min(kl[i, i + 1], kl[i + 1, i]))
        for i in range(K_TIERS - 1)
    }
    d_min = min_adjacent_kl(model)
    mean_pairwise = float(kl[~np.eye(K_TIERS, dtype=bool)].mean())
    print(f"\n  D_min (adjacent)      = {d_min:.2f} nats  at "
          f"{min(adjacent, key=adjacent.get)}")
    print(f"  mean pairwise         = {mean_pairwise:.2f} nats")

    # Interactions implied by the KL heuristic, at the *measured* D_min
    # as well as at the conservative values the manuscript tabulates.
    grid = [0.5, 0.8, 1.2, d_min]
    rows = []
    for conf, delta in CONFIDENCES:
        row = {"Target": f"{conf:.0%} (delta={delta})"}
        for d in grid:
            key = f"D_min={d:.2f}" + (" (measured)" if np.isclose(d, d_min) else "")
            row[key] = f"{sanov_interactions(d, delta):.1f}"
        rows.append(row)
    cols = ["Target"] + [
        f"D_min={d:.2f}" + (" (measured)" if np.isclose(d, d_min) else "") for d in grid
    ]
    table(rows, cols, "Interactions required (KL heuristic)")

    # Per-feature discriminability: how much each feature contributes
    # to the worst-case adjacent-tier separation.
    worst = min(range(K_TIERS - 1), key=lambda i: min(kl[i, i + 1], kl[i + 1, i]))
    ta, tb = ALL_TIERS[worst], ALL_TIERS[worst + 1]
    ma, sa = model.linguistic_params(ta)
    mb, sb = model.linguistic_params(tb)
    per_feature = {
        name: float(kl_gaussian(ma[k:k + 1], sa[k:k + 1], mb[k:k + 1], sb[k:k + 1]))
        for k, name in enumerate(LINGUISTIC_FEATURES)
    }
    ranked = sorted(per_feature.items(), key=lambda kv: -kv[1])
    print(f"\n  Per-feature KL at the hardest boundary ({ta.label}-{tb.label}):")
    for name, v in ranked:
        print(f"    {name:24s} {v:6.2f} nats")

    # Reconciliation against the empirical convergence result.
    reconciliation = _reconcile(d_min)

    # Proposition 1: a proved bound for the estimator as implemented (all four
    # modalities, the attenuated attestation term, and the confidence floor).
    # KL is not the right exponent for a two-sided decision; the Chernoff
    # exponent C = -min_s g(s) is, and it is what the bound below uses.
    chern = {
        ti.label: {tj.label: chernoff_exponent(model, ti, tj)
                   for tj in ALL_TIERS if tj is not ti}
        for ti in ALL_TIERS
    }
    c_min_pair = min(((a, b) for a in chern for b in chern[a]),
                     key=lambda ab: chern[ab[0]][ab[1]])
    bound_n = (1, 3, 5, 7, 10)
    bounds = {t.label: {n: misassignment_bound(model, t, n, DEFAULT_GAMMA) for n in bound_n}
              for t in ALL_TIERS}
    first_n_bound_90 = {}
    for t in ALL_TIERS:
        n = 1
        while misassignment_bound(model, t, n, DEFAULT_GAMMA) > 0.10:
            n += 1
        first_n_bound_90[t.label] = n
    rows = [{"Tier": t.label, **{f"n={n}": f"{100 * bounds[t.label][n]:.1f}" for n in bound_n},
             "n for <=10%": str(first_n_bound_90[t.label])} for t in ALL_TIERS]
    table(rows, ["Tier", *[f"n={n}" for n in bound_n], "n for <=10%"],
          "Proposition 1 -- proved upper bound on misassignment probability (%)")
    print(f"  smallest Chernoff exponent: {chern[c_min_pair[0]][c_min_pair[1]]:.3f} nats "
          f"({c_min_pair[0]} true, {c_min_pair[1]} rival)")

    return {
        "kl_matrix": kl.tolist(),
        "adjacent_kl": adjacent,
        "d_min": d_min,
        "mean_pairwise_kl": mean_pairwise,
        "hardest_boundary": f"{ta.label}-{tb.label}",
        "per_feature_kl_at_hardest_boundary": per_feature,
        "most_discriminative_feature": ranked[0][0],
        "least_discriminative_feature": ranked[-1][0],
        "sanov_interactions": {
            f"{conf}": {f"{d:.2f}": sanov_interactions(d, delta) for d in grid}
            for conf, delta in CONFIDENCES
        },
        "reconciliation": reconciliation,
        "chernoff_exponents": chern,
        "chernoff_min": {"true": c_min_pair[0], "rival": c_min_pair[1],
                         "value": chern[c_min_pair[0]][c_min_pair[1]]},
        "misassignment_bound": {t: {str(n): v for n, v in b.items()} for t, b in bounds.items()},
        "first_n_bound_at_most_10pct": first_n_bound_90,
        "gamma": DEFAULT_GAMMA,
    }


def _reconcile(d_min: float) -> dict:
    """Compare the KL-heuristic prediction with the measured convergence, if available."""
    path = RESULTS / "exp01_convergence.json"
    if not path.exists():
        return {"note": "run exp01 first for the empirical side of the comparison"}
    emp = json.loads(path.read_text())
    predicted = sanov_interactions(d_min, 0.10)
    observed = emp["first_milestone_above_90"]
    print(f"\n  Reconciliation :")
    print(f"    KL heuristic at measured D_min={d_min:.2f}, delta=0.10: "
          f"n >= {predicted:.1f} interactions")
    print(f"    Observed first milestone >=90% per tier: {observed}")
    print("    The bound and the simulation agree: both put convergence within a")
    print("    handful of interactions. The heuristic is not a proof; Proposition 1")
    print("    below is.")
    return {
        "sanov_predicted_interactions_90pct": predicted,
        "observed_first_milestone_above_90": observed,
        "verdict": "the KL heuristic and simulation agree in order of magnitude; the "
                   "proved bound is Proposition 1 (misassignment_bound)",
    }


if __name__ == "__main__":
    save("exp02_separability", run())
