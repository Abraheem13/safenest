"""Experiment 05 -- privacy/utility trade-off and the formal privacy statement.

The headline finding is negative and load-bearing: under a rigorous *local* DP
accounting over the live user's own signals, eps = 1.0 cannot support a five-way
developmental classification at any usable accuracy. High accuracy at eps = 1.0
is attainable only when the guarantee is understood as protecting the
calibration corpus rather than the live user. Both readings are measured here.
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
from safenest.signals import SignalModel, release_corpus_parameters  # noqa: E402
from safenest.tiers import ALL_TIERS  # noqa: E402

EPSILONS = (0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 10.0, 30.0, 100.0)
N_TRIALS = 400
N_INTERACTIONS = 10
#: Independent corpus releases per epsilon. Accuracy under corpus DP depends on
#: the particular noise draw, so it is reported as a mean over releases.
N_RELEASES = 20


def run() -> dict:
    rng = rng_for("privacy")

    print("Formal privacy statements:")
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
    # Each release simulates the whole mechanism: draw a finite calibration
    # corpus, clip and average it, add analytically calibrated Gaussian noise,
    # then classify fresh users drawn from the true model with the released
    # parameters. `None` is the non-private release (sampling error only).
    def corpus_accuracy(eps: float | None) -> tuple[float, float]:
        accs = []
        for _ in range(N_RELEASES):
            cfg = PrivacyConfig(mode=PrivacyMode.CORPUS, epsilon_total=eps or 1.0)
            released = release_corpus_parameters(
                SignalModel(), cfg, rng, private=eps is not None)
            est = BayesianAgeEstimator(model=released, privacy=cfg)
            hits = total = 0
            for tier in ALL_TIERS:
                for _ in range(N_TRIALS // len(ALL_TIERS)):
                    got, _ = est.run_session(
                        tier, N_INTERACTIONS, rng, generating_model=SignalModel()
                    )
                    hits += got is tier
                    total += 1
            accs.append(hits / total)
        return float(np.mean(accs)), float(np.std(accs))

    corpus_acc, corpus_sd, corpus_sigma = {}, {}, {}
    for eps in EPSILONS:
        corpus_acc[eps], corpus_sd[eps] = corpus_accuracy(eps)
        corpus_sigma[eps] = PrivacyConfig(
            mode=PrivacyMode.CORPUS, epsilon_total=eps).corpus_gaussian_sigma()
    nonprivate_acc, nonprivate_sd = corpus_accuracy(None)

    rows = [
        {
            "epsilon": f"{eps:g}" + (" *" if eps == 1.0 else ""),
            "CORPUS DP (%)": f"{pct(corpus_acc[eps])} +- {pct(corpus_sd[eps])}",
            "sigma (sd units)": f"{corpus_sigma[eps]:.3f}",
            "LOCAL DP at inference (%)": pct(local_acc[eps]),
            "cum. eps @ n=10 (local)": f"{eps * N_INTERACTIONS:g}",
        }
        for eps in EPSILONS
    ]
    table(rows, ["epsilon", "CORPUS DP (%)", "sigma (sd units)", "LOCAL DP at inference (%)",
                 "cum. eps @ n=10 (local)"],
          "Accuracy vs privacy budget, n=10 interactions")

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
        "accuracy_corpus_dp_sd_over_releases": corpus_sd,
        "corpus_noise_sigma_sd_units": corpus_sigma,
        "accuracy_nonprivate_release": nonprivate_acc,
        "accuracy_nonprivate_release_sd": nonprivate_sd,
        "n_releases": N_RELEASES,
        "accuracy_local_dp": local_acc,
        "local_snr_at_eps1": snr,
        "n_trials": N_TRIALS,
        "n_interactions": N_INTERACTIONS,
        "verdict": (
            "High accuracy at eps=1.0 is attainable only under CORPUS-mode DP, "
            "where (eps, delta) protects the children in the calibration corpus "
            "and the release is a clipped mean with analytically calibrated "
            "Gaussian noise. Under "
            "local DP applied to the live user's own signals, eps=1.0 gives "
            "near-chance accuracy, and the cumulative loss over a 10-interaction "
            "session is eps=10 under basic composition. Any eps claim must state "
            "which reading it intends."
        ),
    }


if __name__ == "__main__":
    save("exp05_privacy", run())
