"""Experiment 06 -- atypically developing and non-Western populations.

Quantifies how the tier estimator behaves when linguistic profile and protective
need diverge, and evaluates the profile/attestation discordance rule that holds a
linguistically advanced young child at the protective tier.

Also reports which signal modality is most robust to cross-cultural shift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.estimator import BayesianAgeEstimator  # noqa: E402
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.signals import MODALITIES, PROFILES, SignalModel  # noqa: E402
from safenest.tiers import ALL_TIERS, Tier  # noqa: E402

N_TRIALS = 400
N_INTERACTIONS = 10


def run() -> dict:
    rng = rng_for("populations")
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))

    # ---- accuracy and *direction of error* by population -------------------
    rows = []
    detail = {}
    for name, profile in PROFILES.items():
        gen = SignalModel(profile=profile)
        stats = {"correct": 0, "under": 0, "over": 0, "n": 0}
        protected = {"correct": 0, "under": 0, "over": 0, "n": 0}
        for tier in ALL_TIERS:
            for _ in range(N_TRIALS // len(ALL_TIERS)):
                # Paired: both conditions read the *same* session, so the
                # difference is attributable to the flag rather than to noise.
                state, last = _session(est, gen, tier, rng)
                _tally(stats, est.resolve(state, linguistic=last), tier)
                # With the discordance flag: an external attestation of the
                # child's true tier is available (parent or school device).
                _tally(
                    protected,
                    est.resolve(state, attested_tier=tier, linguistic=last),
                    tier,
                )
        detail[name] = {"unmitigated": stats, "with_discordance_flag": protected}
        rows.append({
            "Population": name,
            "Acc (%)": pct(stats["correct"] / stats["n"]),
            "Under-prot (%)": pct(stats["under"] / stats["n"]),
            "Over-prot (%)": pct(stats["over"] / stats["n"]),
            "Acc w/ flag (%)": pct(protected["correct"] / protected["n"]),
            "Under w/ flag (%)": pct(protected["under"] / protected["n"]),
        })
    table(rows, ["Population", "Acc (%)", "Under-prot (%)", "Over-prot (%)",
                 "Acc w/ flag (%)", "Under w/ flag (%)"],
          "Tier estimation by population (n=10 interactions). Under-protection means "
          "the child was assigned a *less* restrictive tier than their true one.")

    # ---- worked case: linguistic profile two tiers above chronological band --
    print("\n  Worked case: a highly verbal 8-year-old")
    print("  whose MTLD and parse depth read as t4.")
    gen = SignalModel(profile=PROFILES["neurodivergent_verbal"])
    unflagged = [
        est.run_session(Tier.T2, N_INTERACTIONS, rng, generating_model=gen)[0]
        for _ in range(200)
    ]
    flagged = [
        est.run_session(Tier.T2, N_INTERACTIONS, rng, generating_model=gen,
                        attested_tier=Tier.T2)[0]
        for _ in range(200)
    ]
    up_un = np.mean([int(t) > 2 for t in unflagged])
    up_fl = np.mean([int(t) > 2 for t in flagged])
    print(f"    assigned above t2 without the flag: {100 * up_un:.1f}% of sessions")
    print(f"    assigned above t2 with the flag:    {100 * up_fl:.1f}% of sessions")

    # ---- modality robustness to cross-cultural shift -----------------------
    print("\n  Modality robustness under non-Western / L2 shift:")
    robustness = {}
    base = SignalModel()
    shifted = SignalModel(profile=PROFILES["non_weird_l2"])
    for modality in MODALITIES:
        agree = 0
        for tier in ALL_TIERS:
            for _ in range(200):
                sig = shifted.sample(tier, rng)
                llr = base.llr_vector(modality, sig[modality])
                agree += int(np.argmax(llr)) == int(tier) - 1
        robustness[modality] = agree / (200 * len(ALL_TIERS))
    ranked = sorted(robustness.items(), key=lambda kv: -kv[1])
    for m, v in ranked:
        print(f"    {m:14s} single-modality accuracy under shift: {pct(v)}%")
    print(f"    most robust: {ranked[0][0]}; least robust: {ranked[-1][0]}")

    return {
        "n_trials": N_TRIALS,
        "n_interactions": N_INTERACTIONS,
        "by_population": detail,
        "population_table": rows,
        "verbal_case": {
            "assigned_above_t2_unflagged": float(up_un),
            "assigned_above_t2_flagged": float(up_fl),
        },
        "modality_robustness_under_shift": robustness,
        "most_robust_modality": ranked[0][0],
        "least_robust_modality": ranked[-1][0],
    }


def _session(est, gen, tier, rng):
    """Run one session and return its final state plus the last linguistic vector."""
    from safenest.estimator import EstimatorState

    state = EstimatorState()
    last = None
    for _ in range(N_INTERACTIONS):
        signals = gen.sample(tier, rng)
        last = signals["linguistic"]
        est.observe(state, signals, rng)
    return state, last


def _tally(stats: dict, got: Tier, true: Tier) -> None:
    stats["n"] += 1
    if got is true:
        stats["correct"] += 1
    elif int(got) > int(true):
        stats["under"] += 1   # less restrictive than warranted
    else:
        stats["over"] += 1    # more restrictive than warranted


if __name__ == "__main__":
    save("exp06_populations", run())
