"""Experiment 10 -- bypass detection against a graded adversary.

The specification claims a bypass-detection mechanism but the original
validation never measured it: the only adversary tested sat six standard
deviations from the tier mean, which any threshold catches. That is not the
threat. The realistic case is a child who supplies text they did not write --
copied from a sibling, a parent or a model -- in order to be assigned a less
restrictive tier.

This experiment models that adversary as a convex interpolation between the
child's genuine linguistic profile and the profile of the tier they are
impersonating, at spoof strength alpha in [0, 1]. Only the linguistic channel
is spoofed: typing dynamics and device indicators are not under the child's
control in the same way, which is the whole argument for multi-signal fusion,
and this experiment is where that argument is either earned or lost.

Detection is reported against its true cost -- the rate at which genuine
atypical children are flagged -- because a detector that catches every
impersonator by suspecting every unusual child is not a safety mechanism.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.estimator import (  # noqa: E402
    MAHALANOBIS_CHI2_ALPHA01_DF5, BayesianAgeEstimator, EstimatorState,
)
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.signals import PROFILES, SignalModel  # noqa: E402
from safenest.tiers import Tier  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_TRIALS = 400
N_INTERACTIONS = 5
TRUE_TIER = Tier.T2      # an eight-year-old
TARGET_TIER = Tier.T5    # impersonating a sixteen-year-old
#: Dense enough that the adopted chi-square threshold lies on the plotted curve.
THRESHOLD_GRID = tuple(sorted(set(
    [float(x) for x in np.linspace(0.5, 60.0, 120)] + [MAHALANOBIS_CHI2_ALPHA01_DF5]
)))


def _spoofed_session(est, alpha, rng):
    """One session in which the linguistic channel is interpolated toward the
    target tier, while every other modality remains genuine."""
    genuine = SignalModel()
    mu_true, _ = genuine.linguistic_params(TRUE_TIER)
    mu_target, _ = genuine.linguistic_params(TARGET_TIER)
    state = EstimatorState()
    last = None
    for _ in range(N_INTERACTIONS):
        signals = genuine.sample(TRUE_TIER, rng)
        # Shift the linguistic vector toward the impersonated profile.
        signals["linguistic"] = (
            signals["linguistic"] + alpha * (mu_target - mu_true)
        )
        last = signals["linguistic"]
        est.observe(state, signals, rng)
    return state, last


def _genuine_session(est, profile_name, tier, rng):
    gen = SignalModel(profile=PROFILES[profile_name])
    state = EstimatorState()
    last = None
    for _ in range(N_INTERACTIONS):
        signals = gen.sample(tier, rng)
        last = signals["linguistic"]
        est.observe(state, signals, rng)
    return state, last


def run() -> dict:
    rng = rng_for("bypass")
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))

    # ---- escalation and detection as a function of spoof strength ----------
    rows, detail = [], {}
    spoof_scores: dict[float, list[float]] = {}
    for alpha in ALPHAS:
        escalated = flagged = 0
        scores = []
        for _ in range(N_TRIALS):
            state, last = _spoofed_session(est, alpha, rng)
            assigned = est.assign(state)
            d2 = est.mahalanobis(last, assigned) ** 2
            scores.append(d2)
            escalated += int(assigned) > int(TRUE_TIER)
            flagged += d2 > MAHALANOBIS_CHI2_ALPHA01_DF5
        spoof_scores[alpha] = scores
        detail[alpha] = {
            "escalation_rate": escalated / N_TRIALS,
            "detection_rate": flagged / N_TRIALS,
        }
        rows.append({
            "spoof strength": f"{alpha:.2f}",
            "escalated above t2 (%)": pct(escalated / N_TRIALS),
            "flagged by detector (%)": pct(flagged / N_TRIALS),
        })
    table(rows, ["spoof strength", "escalated above t2 (%)", "flagged by detector (%)"],
          f"Impersonation of {TARGET_TIER.label} by a genuine {TRUE_TIER.label} user "
          f"(N={N_TRIALS} sessions per strength)")

    # ---- what detection costs genuine children ----------------------------
    genuine_scores: dict[str, list[float]] = {}
    false_rows = []
    for name in ("typical", "neurodivergent_verbal", "non_weird_l2", "dialect_switching"):
        scores = []
        flagged = 0
        for _ in range(N_TRIALS):
            state, last = _genuine_session(est, name, TRUE_TIER, rng)
            assigned = est.assign(state)
            d2 = est.mahalanobis(last, assigned) ** 2
            scores.append(d2)
            flagged += d2 > MAHALANOBIS_CHI2_ALPHA01_DF5
        genuine_scores[name] = scores
        false_rows.append({
            "genuine cohort": name,
            "false-flag rate (%)": pct(flagged / N_TRIALS),
        })
    table(false_rows, ["genuine cohort", "false-flag rate (%)"],
          "Cost of detection: genuine children wrongly flagged at the adopted "
          "threshold (chi-square, alpha = 0.01)")

    # ---- ROC over the detection threshold ---------------------------------
    def roc(pos: list[float], neg: list[float]) -> tuple[list, float]:
        pts = []
        for thr in THRESHOLD_GRID:
            tpr = float(np.mean([s > thr for s in pos]))
            fpr = float(np.mean([s > thr for s in neg]))
            pts.append({"threshold": thr, "tpr": tpr, "fpr": fpr})
        # AUC by the rank statistic, ties counted as half.
        p, n = np.asarray(pos), np.asarray(neg)
        greater = (p[:, None] > n[None, :]).sum()
        equal = (p[:, None] == n[None, :]).sum()
        return pts, float((greater + 0.5 * equal) / (len(p) * len(n)))

    curves, aucs = {}, {}
    for alpha in (0.5, 0.75, 1.0):
        pts, auc = roc(spoof_scores[alpha], genuine_scores["typical"])
        curves[str(alpha)] = pts
        aucs[str(alpha)] = auc
    # The hard case: separating an impersonator from a genuinely advanced child.
    verbal_curve, auc_vs_verbal = roc(
        spoof_scores[1.0], genuine_scores["neurodivergent_verbal"])
    thr = MAHALANOBIS_CHI2_ALPHA01_DF5
    detected_at_thr = float(np.mean([x > thr for x in spoof_scores[1.0]]))
    operating_points = {
        name: {"fpr": float(np.mean([x > thr for x in genuine_scores[name]])),
               "tpr": detected_at_thr}
        for name in ("typical", "neurodivergent_verbal")
    }

    print("\n  Detector AUC, impersonator versus typical child:")
    for a, v in aucs.items():
        print(f"    spoof strength {a}: AUC = {v:.3f}")
    print(f"  Impersonator versus genuinely verbally advanced child: "
          f"AUC = {auc_vs_verbal:.3f}")
    print("  The second number is the one that matters: the detector must "
          "separate\n  a child who is pretending from a child who is simply "
          "articulate.")

    return {
        "true_tier": TRUE_TIER.label,
        "target_tier": TARGET_TIER.label,
        "alphas": list(ALPHAS),
        "by_spoof_strength": {str(k): v for k, v in detail.items()},
        "false_flag_rates": {
            k: float(np.mean([s > MAHALANOBIS_CHI2_ALPHA01_DF5 for s in v]))
            for k, v in genuine_scores.items()
        },
        "roc": curves,
        "auc_vs_typical": aucs,
        "auc_vs_verbally_advanced": auc_vs_verbal,
        "roc_vs_verbally_advanced": verbal_curve,
        "operating_points_alpha1": operating_points,
        "adopted_threshold": MAHALANOBIS_CHI2_ALPHA01_DF5,
        "n_trials": N_TRIALS,
        "n_interactions": N_INTERACTIONS,
    }


if __name__ == "__main__":
    save("exp10_bypass", run())
