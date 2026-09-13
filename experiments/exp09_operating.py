"""Experiment 09 -- operating characteristics of the two free thresholds.

Two constants govern the framework's central trade-off, and neither was
justified in the original specification: the estimator's confidence threshold
gamma, which decides when the system falls back to the most restrictive tier,
and the severity threshold above which content is refused at every tier.

A threshold that is never varied is an unstated modelling choice. This
experiment sweeps both and reports the resulting curve, so a deployment can
choose an operating point deliberately rather than inheriting ours.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.estimator import BayesianAgeEstimator  # noqa: E402
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.signals import PROFILES, SignalModel  # noqa: E402
from safenest.tiers import ALL_TIERS  # noqa: E402

GAMMA_GRID = (0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95)
SEVERITY_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
N_TRIALS = 400
N_INTERACTIONS = 5


def _sweep_gamma(profile_name: str, rng) -> dict:
    """Under- and over-protection as a function of the confidence threshold."""
    gen = SignalModel(profile=PROFILES[profile_name])
    out = {}
    for gamma in GAMMA_GRID:
        est = BayesianAgeEstimator(
            privacy=PrivacyConfig(mode=PrivacyMode.CORPUS), gamma=gamma
        )
        correct = under = over = n = 0
        for tier in ALL_TIERS:
            for _ in range(N_TRIALS // len(ALL_TIERS)):
                got, _ = est.run_session(
                    tier, N_INTERACTIONS, rng, generating_model=gen
                )
                n += 1
                if got is tier:
                    correct += 1
                elif int(got) > int(tier):
                    under += 1   # less restrictive than warranted
                else:
                    over += 1    # more restrictive than warranted
        out[gamma] = {
            "accuracy": correct / n,
            "under_protection": under / n,
            "over_restriction": over / n,
        }
    return out


def run() -> dict:
    rng = rng_for("operating")

    typical = _sweep_gamma("typical", rng)
    verbal = _sweep_gamma("neurodivergent_verbal", rng)

    table(
        [{"gamma": f"{g:.2f}" + (" *" if g == 0.55 else ""),
          "Acc (%)": pct(typical[g]["accuracy"]),
          "Under (%)": pct(typical[g]["under_protection"]),
          "Over (%)": pct(typical[g]["over_restriction"]),
          "Acc atyp (%)": pct(verbal[g]["accuracy"]),
          "Under atyp (%)": pct(verbal[g]["under_protection"])}
         for g in GAMMA_GRID],
        ["gamma", "Acc (%)", "Under (%)", "Over (%)", "Acc atyp (%)", "Under atyp (%)"],
        "Confidence threshold sweep (n=5 interactions). '*' is the adopted value. "
        "'atyp' is the verbally advanced cohort.",
    )

    # The protective reading: gamma buys under-protection at the price of
    # over-restriction, and the exchange rate is what a deployment must choose.
    best_typ = min(GAMMA_GRID, key=lambda g: typical[g]["under_protection"])
    print(f"\n  Under-protection is minimised at gamma = {best_typ:.2f} "
          f"({pct(typical[best_typ]['under_protection'])}%), at "
          f"{pct(typical[best_typ]['over_restriction'])}% over-restriction.")
    print(f"  The adopted gamma = 0.55 gives "
          f"{pct(typical[0.55]['under_protection'])}% under and "
          f"{pct(typical[0.55]['over_restriction'])}% over.")

    # ---- severity threshold ------------------------------------------------
    from safenest import policy as policy_mod
    from safenest.baselines import make_npl
    from safenest.corpus import HARM_CATEGORIES, build_corpus
    from safenest.labeling import rubric_label
    from safenest.metrics import evaluate_framework

    prompts = build_corpus()
    harm = [p for p in prompts if p.category in HARM_CATEGORIES]
    original = policy_mod.SEVERITY_REJECT_THRESHOLD
    severity = {}
    try:
        for thr in SEVERITY_GRID:
            policy_mod.SEVERITY_REJECT_THRESHOLD = thr
            r = evaluate_framework(make_npl(), harm, rubric_label)["overall"]
            severity[thr] = {
                "dsr": r["dsr"],
                "under_protection": r["under_protection"],
                "over_restriction": r["over_restriction"],
            }
    finally:
        policy_mod.SEVERITY_REJECT_THRESHOLD = original

    table(
        [{"threshold": f"{t:.1f}" + (" *" if t == 0.5 else ""),
          "DSR (%)": pct(severity[t]["dsr"]),
          "Under (%)": pct(severity[t]["under_protection"]),
          "Over (%)": pct(severity[t]["over_restriction"])}
         for t in SEVERITY_GRID],
        ["threshold", "DSR (%)", "Under (%)", "Over (%)"],
        "Severity gate sweep, harm categories only (n=3,000 prompts)",
    )
    best_sev = max(SEVERITY_GRID, key=lambda t: severity[t]["dsr"])
    print(f"\n  Best DSR on harm categories at threshold {best_sev:.1f} "
          f"({pct(severity[best_sev]['dsr'])}%); adopted 0.5 gives "
          f"{pct(severity[0.5]['dsr'])}%.")

    return {
        "gamma_grid": list(GAMMA_GRID),
        "gamma_typical": {str(k): v for k, v in typical.items()},
        "gamma_verbally_advanced": {str(k): v for k, v in verbal.items()},
        "adopted_gamma": 0.55,
        "severity_grid": list(SEVERITY_GRID),
        "severity_sweep": {str(k): v for k, v in severity.items()},
        "adopted_severity_threshold": 0.5,
        "n_trials": N_TRIALS,
        "n_interactions": N_INTERACTIONS,
    }


if __name__ == "__main__":
    save("exp09_operating", run())
