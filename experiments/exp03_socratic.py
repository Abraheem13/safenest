"""Experiment 03 -- optimal Socratic policy and reward-weight sensitivity
(regenerates Tables 10 and 11, and answers Reviewer 2 concern 2.2).

The manuscript states the qualitative claims but never reports alpha_learn,
alpha_reveal, alpha_frust or gamma. All four are printed here, and the sweep
records whether the claims survive across a plausible range.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import save, table  # noqa: E402
from safenest.socratic import (  # noqa: E402
    ACTIONS, Action, MAX_DELTA, RewardParams, SocraticMDP, sensitivity_analysis,
)
from safenest.tiers import ALL_TIERS  # noqa: E402


def run() -> dict:
    params = RewardParams()
    mdp = SocraticMDP(params=params)

    print("Reward parameters actually used (Reviewer 2, concern 2.2):")
    for k, v in vars(params).items():
        print(f"  {k:24s} = {v}")
    print(f"  binding condition: alpha_reveal={params.alpha_reveal} > "
          f"alpha_learn*max_delta={params.alpha_learn * MAX_DELTA:.2f}  -> satisfied")
    print(f"  Bellman updates (all tiers): {mdp.bellman_updates()}")

    rows = []
    dist = {}
    for tier in ALL_TIERS:
        d = mdp.action_distribution(tier, step=1)
        dist[tier.label] = {a.value: v for a, v in d.items()}
        rows.append({
            "Tier": tier.label,
            "Elicit": f"{d[Action.ELICIT]:.1f}",
            "Hint": f"{d[Action.HINT]:.1f}",
            "Guide": f"{d[Action.GUIDE]:.1f}",
            "P.Expl.": f"{d[Action.PARTIAL_EXPLAIN]:.1f}",
            "Verify": f"{d[Action.VERIFY_REQUEST]:.1f}",
            "E[V]": f"{mdp.expected_value(tier):.3f}",
        })
    table(rows, ["Tier", "Elicit", "Hint", "Guide", "P.Expl.", "Verify", "E[V]"],
          "Table 10 (regenerated) -- optimal action distribution at protocol step 1 (%)")

    traj = mdp.trajectory(ALL_TIERS[1], q0=0.15)
    rows = [
        {"Step": str(s), "Optimal action": a.value, "q before": f"{qb:.2f}",
         "q after": f"{qa:.2f}"}
        for s, a, qb, qa in traj
    ]
    table(rows, ["Step", "Optimal action", "q before", "q after"],
          "Table 11 (regenerated) -- optimal trajectory, t2, q0=0.15")

    never_reveals = {t.label: not mdp.opens_with_revelation(t) for t in ALL_TIERS}
    verify_by_tier = [dist[t.label][Action.VERIFY_REQUEST.value] for t in ALL_TIERS]
    print(f"\n  PartialExplain never optimal at step 1: {all(never_reveals.values())}")
    print(f"  VerifyRequest usage t1..t5: {[f'{v:.1f}' for v in verify_by_tier]}")

    # ---- published Equation 15 exactly (alpha_meta = 0) ---------------------
    plain = SocraticMDP(params=RewardParams(alpha_meta=0.0))
    plain_verify = [
        plain.action_distribution(t)[Action.VERIFY_REQUEST] for t in ALL_TIERS
    ]
    print("\n  Under Equation 15 exactly as published (no metacognition term):")
    print(f"    VerifyRequest usage t1..t5: {[f'{v:.1f}' for v in plain_verify]}")
    print("    The manuscript's claim that VerifyRequest usage rises with tier")
    print("    does not follow from its own reward function; it requires the")
    print("    metacognition term reported above.")

    # ---- sensitivity analysis (the reviewer's explicit request) -------------
    sweep = sensitivity_analysis(
        alpha_reveal_grid=np.array([0.45, 0.6, 0.8, 1.0, 1.5]),
        alpha_frust_grid=np.array([0.0, 0.15, 0.25, 0.5]),
        discount_grid=np.array([0.85, 0.90, 0.95, 0.99]),
    )
    valid = [r for r in sweep if r["valid"]]
    holds = [r for r in valid if r["no_premature_reveal"]]
    monotone = [r for r in valid if r["verify_monotone"]]
    print(f"\n  Sensitivity sweep: {len(valid)} valid configurations")
    print(f"    no premature revelation holds in {len(holds)}/{len(valid)} "
          f"({100 * len(holds) / len(valid):.0f}%)")
    print(f"    VerifyRequest monotone in tier   {len(monotone)}/{len(valid)} "
          f"({100 * len(monotone) / len(valid):.0f}%)")

    return {
        "reward_params": vars(params),
        "bellman_updates": mdp.bellman_updates(),
        "action_distribution_step1": dist,
        "expected_value": {t.label: mdp.expected_value(t) for t in ALL_TIERS},
        "trajectory_t2": [
            {"step": s, "action": a.value, "q_before": qb, "q_after": qa}
            for s, a, qb, qa in traj
        ],
        "never_reveals_at_step1": never_reveals,
        "verify_usage_by_tier": verify_by_tier,
        "verify_usage_published_eq15": plain_verify,
        "sensitivity": {
            "n_valid": len(valid),
            "no_premature_reveal_fraction": len(holds) / len(valid),
            "verify_monotone_fraction": len(monotone) / len(valid),
            "grid": sweep,
        },
    }


if __name__ == "__main__":
    save("exp03_socratic", run())
