"""Experiment 05 -- privacy/utility (regenerates Table 12 and Figure 8), and the
formal privacy statement Reviewer 1 asks for in major comment 3.

The headline finding is negative and load-bearing: under a rigorous *local* DP
accounting at the live user's own signals, eps = 1.0 cannot support a five-way
developmental classification at any accuracy. The manuscript's 98.6% at eps=1.0
is only attainable if the eps guarantee is understood as protecting the
calibration corpus, not the live user. Both readings are measured here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.estimator import BayesianAgeEstimator  # noqa: E402
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.signals import SignalModel, with_corpus_dp_noise  # noqa: E402
from safenest.tiers import ALL_TIERS  # noqa: E402

EPSILONS = (0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 10.0, 30.0, 100.0)
N_TRIALS = 400
N_INTERACTIONS = 10


def run() -> dict:
    rng = rng_for("privacy")

    print("Formal privacy statements (Reviewer 1, major comment 3):")
    statements = {}
    for mode in PrivacyMode:
        d = PrivacyConfig(mode=mode).describe()
        statements[mode.value] = d
        print(f"\n  [{mode.value}]")
        for k in ("privacy_unit", "adjacency", "protected_output", "mechanism"):
            print(f"    {k:18s}: {d[k]}")
        if mode is PrivacyMode.LOCAL_INFERENCE:
            print(f"    {'l1_sensitivity':18s}: {d['l1_sensitivity']}")
            print(f"    {'cum. eps @ n=10':18s}: {d['cumulative_epsilon_10_interactions']:.1f} "
                  f"(basic) / {d['cumulative_epsilon_10_advanced']:.1f} (advanced)")

    # ---- local DP at inference ------------------------------------------
    local_acc = {}
    for eps in EPSILONS:
        est = BayesianAgeEstimator(
            privacy=PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, epsilon_total=eps)
        )
        hits = 0
        for tier in ALL_TIERS:
            for _ in range(N_TRIALS // len(ALL_TIERS)):
                got, _ = est.run_session(tier, N_INTERACTIONS, rng)
                hits += got is tier
        local_acc[eps] = hits / (N_TRIALS // len(ALL_TIERS) * len(ALL_TIERS))

    # ---- corpus DP ------------------------------------------------------
    corpus_acc = {}
    for eps in EPSILONS:
        cfg = PrivacyConfig(mode=PrivacyMode.CORPUS, epsilon_total=eps)
        released = with_corpus_dp_noise(SignalModel(), cfg.corpus_parameter_noise_std(), rng)
        est = BayesianAgeEstimator(model=released, privacy=cfg)
        hits = 0
        for tier in ALL_TIERS:
            for _ in range(N_TRIALS // len(ALL_TIERS)):
                # Users are drawn from the true distribution; the estimator uses
                # the DP-released parameters.
                got, _ = est.run_session(
                    tier, N_INTERACTIONS, rng, generating_model=SignalModel()
                )
                hits += got is tier
        corpus_acc[eps] = hits / (N_TRIALS // len(ALL_TIERS) * len(ALL_TIERS))

    rows = [
        {
            "epsilon": f"{eps:g}" + (" *" if eps == 1.0 else ""),
            "CORPUS DP (%)": pct(corpus_acc[eps]),
            "LOCAL DP at inference (%)": pct(local_acc[eps]),
            "cum. eps @ n=10 (local)": f"{eps * N_INTERACTIONS:g}",
        }
        for eps in EPSILONS
    ]
    table(rows, ["epsilon", "CORPUS DP (%)", "LOCAL DP at inference (%)",
                 "cum. eps @ n=10 (local)"],
          "Table 12 (regenerated) -- accuracy vs privacy budget, n=10 interactions")

    cfg = PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, epsilon_total=1.0)
    snr = cfg.per_release_snr("linguistic")
    print(f"\n  Why LOCAL DP fails here: per-release SNR = eps_m / (2(K-1)) = {snr:.3f},")
    print("  independent of the clip bound C, so tightening the clip cannot help.")
    print(f"  Chance accuracy is {100 / len(ALL_TIERS):.0f}%; the fail-safe floor sends")
    print("  unconfident sessions to t1, so residual error is over-protection.")

    return {
        "privacy_statements": statements,
        "epsilons": list(EPSILONS),
        "accuracy_corpus_dp": corpus_acc,
        "accuracy_local_dp": local_acc,
        "local_snr_at_eps1": snr,
        "n_trials": N_TRIALS,
        "n_interactions": N_INTERACTIONS,
        "verdict": (
            "The 98.6% at eps=1.0 reported in the manuscript is reproducible only "
            "under CORPUS-mode DP, where eps protects the children in the "
            "calibration corpora. Under local DP applied to the live user's own "
            "signals, eps=1.0 gives near-chance accuracy, and the cumulative loss "
            "over a 10-interaction session is eps=10 under basic composition. The "
            "paper must state which reading it intends."
        ),
    }


if __name__ == "__main__":
    save("exp05_privacy", run())
