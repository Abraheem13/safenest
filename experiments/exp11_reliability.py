"""Experiment 11 -- statistical reliability of the reported results.

Four things the original evaluation did not establish, each of which a careful
reader is entitled to ask for:

  seed variance     every headline figure came from one master seed, so no
                    number carried a run-to-run interval;
  calibration       the estimator's decisions are gated on posterior
                    confidence, but the posterior was never checked against
                    observed frequency, so the threshold's meaning was assumed;
  multiplicity      significance was claimed across a large family of paired
                    comparisons with no correction;
  symmetry          only the proposed framework was re-run with estimated
                    tiers, while every baseline kept oracle knowledge.

None of these changes the direction of the results. They change what may
honestly be claimed about them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.baselines import all_frameworks  # noqa: E402
from safenest.corpus import HARM_CATEGORIES, build_corpus  # noqa: E402
from safenest.estimator import BayesianAgeEstimator, EstimatorState  # noqa: E402
from safenest.labeling import rubric_label  # noqa: E402
from safenest.metrics import (  # noqa: E402
    correctness_vector, evaluate_framework, mcnemar,
)
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.tiers import ALL_TIERS  # noqa: E402

N_SEEDS = 20
N_CALIB = 3000
CONFIDENCE_BINS = np.linspace(0.2, 1.0, 9)
#: A plausible deployment mix, in place of the balanced design. Educational
#: traffic dominates; explicit harm content is rare but not negligible.
PREVALENCE = {
    "homework_assignment": 0.34, "open_ended_chat": 0.28,
    "code_generation": 0.08, "essay_creative_writing": 0.12,
    "crisis_self_harm": 0.03, "substance_body_image": 0.05,
    "age_inappropriate_content": 0.10,
}


def _holm(pvalues: dict[str, float]) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction over a family of comparisons."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        adjusted = min(1.0, max(running, (m - i) * p))
        running = adjusted
        out[name] = {"p_raw": p, "p_holm": adjusted, "significant_at_05": adjusted < 0.05}
    return out


def run() -> dict:
    rng = rng_for("reliability")

    # ---- 1. seed variance --------------------------------------------------
    per_seed: dict[str, list[float]] = {}
    for seed in range(N_SEEDS):
        prompts = build_corpus(n_per_cell=60, seed=1000 + seed)
        for name, fw in all_frameworks().items():
            r = evaluate_framework(fw, prompts, rubric_label)["overall"]["dsr"]
            per_seed.setdefault(name, []).append(r)
    variance = {
        name: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
               "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
        for name, v in per_seed.items()
    }
    table(
        [{"Framework": k,
          "Mean DSR (%)": pct(v["mean"]),
          "SD (pp)": f"{100 * v['sd']:.2f}",
          "95% range (%)": f"[{pct(v['lo'])}, {pct(v['hi'])}]"}
         for k, v in sorted(variance.items(), key=lambda kv: -kv[1]["mean"])],
        ["Framework", "Mean DSR (%)", "SD (pp)", "95% range (%)"],
        f"Across {N_SEEDS} independent corpus seeds, rubric ground truth",
    )

    # ---- 2. posterior calibration -----------------------------------------
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))
    conf, hit = [], []
    for _ in range(N_CALIB):
        tier = ALL_TIERS[int(rng.integers(len(ALL_TIERS)))]
        state = EstimatorState()
        for _ in range(int(rng.integers(1, 6))):
            est.observe(state, est.model.sample(tier, rng), rng)
        post = state.posterior
        conf.append(float(post.max()))
        hit.append(ALL_TIERS[int(np.argmax(post))] is tier)
    conf, hit = np.asarray(conf), np.asarray(hit)
    bins, ece = [], 0.0
    edges = list(zip(CONFIDENCE_BINS, CONFIDENCE_BINS[1:]))
    for i, (lo, hi) in enumerate(edges):
        # The top bin is closed so that a posterior of exactly 1.0 is counted.
        last = i == len(edges) - 1
        sel = (conf >= lo) & ((conf <= hi) if last else (conf < hi))
        if sel.sum() == 0:
            continue
        c, a = float(conf[sel].mean()), float(hit[sel].mean())
        # Every sample contributes to the ECE; sparse bins are only hidden
        # from the plotted reliability curve.
        ece += sel.sum() / len(conf) * abs(c - a)
        if sel.sum() >= 10:
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                         "mean_confidence": c, "observed_accuracy": a})
    assert sum(int(((conf >= lo) & ((conf <= hi) if i == len(edges) - 1 else (conf < hi))).sum())
               for i, (lo, hi) in enumerate(edges)) == len(conf), "calibration bins must cover [0.2, 1]"
    table(
        [{"confidence bin": f"[{b['lo']:.1f}, {b['hi']:.1f})", "n": str(b["n"]),
          "mean conf (%)": pct(b["mean_confidence"]),
          "observed acc (%)": pct(b["observed_accuracy"]),
          "gap (pp)": f"{100 * (b['observed_accuracy'] - b['mean_confidence']):+.1f}"}
         for b in bins],
        ["confidence bin", "n", "mean conf (%)", "observed acc (%)", "gap (pp)"],
        "Posterior calibration (1-5 interactions, corpus-level DP)",
    )
    print(f"\n  Expected calibration error: {ece:.4f}")
    print("  A positive gap means the posterior is under-confident, which is the "
          "protective\n  direction: the fail-safe fires more often than the "
          "posterior strictly requires.")

    # ---- 3. multiplicity ---------------------------------------------------
    prompts = build_corpus()
    frameworks = all_frameworks()
    npl_vec = correctness_vector(frameworks["NPL (ours)"], prompts, rubric_label)
    raw = {}
    for name, fw in frameworks.items():
        if name == "NPL (ours)":
            continue
        raw[name] = mcnemar(npl_vec, correctness_vector(fw, prompts, rubric_label))["p_value"]
    corrected = _holm(raw)
    table(
        [{"Comparison": f"NPL vs {k}",
          "p (raw)": f"{v['p_raw']:.3g}",
          "p (Holm)": f"{v['p_holm']:.3g}",
          "sig. at .05": "yes" if v["significant_at_05"] else "no"}
         for k, v in sorted(corrected.items(), key=lambda kv: kv[1]["p_holm"])],
        ["Comparison", "p (raw)", "p (Holm)", "sig. at .05"],
        f"Holm-Bonferroni over the family of {len(raw)} paired comparisons",
    )

    # ---- 4. symmetric end-to-end evaluation --------------------------------
    e2e_rng = rng_for("reliability_e2e")
    est2 = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))
    cache: dict[int, object] = {}

    def assigned(p):
        if p.idx not in cache:
            cache[p.idx] = est2.run_session(p.tier, 5, e2e_rng)[0]
        return cache[p.idx]

    e2e = {}
    for name, fw in frameworks.items():
        oracle = evaluate_framework(fw, prompts, rubric_label)["overall"]
        est_r = evaluate_framework(fw, prompts, rubric_label, assigned_tier=assigned)["overall"]
        e2e[name] = {
            "oracle_dsr": oracle["dsr"], "estimated_dsr": est_r["dsr"],
            "delta_pp": 100 * (est_r["dsr"] - oracle["dsr"]),
            "estimated_under": est_r["under_protection"],
        }
    table(
        [{"Framework": k, "Oracle (%)": pct(v["oracle_dsr"]),
          "Estimated (%)": pct(v["estimated_dsr"]),
          "Delta (pp)": f"{v['delta_pp']:+.1f}",
          "Under (%)": pct(v["estimated_under"])}
         for k, v in sorted(e2e.items(), key=lambda kv: -kv[1]["estimated_dsr"])],
        ["Framework", "Oracle (%)", "Estimated (%)", "Delta (pp)", "Under (%)"],
        "Every framework re-run with estimated rather than oracle tiers",
    )

    # ---- 5. prevalence-weighted DSR ---------------------------------------
    weighted = {}
    for name, fw in frameworks.items():
        r = evaluate_framework(fw, prompts, rubric_label)["by_category"]
        weighted[name] = float(sum(PREVALENCE[c] * r[c]["dsr"] for c in PREVALENCE))
    table(
        [{"Framework": k, "Balanced (%)": pct(variance[k]["mean"]),
          "Prevalence-weighted (%)": pct(v),
          "Delta (pp)": f"{100 * (v - variance[k]['mean']):+.1f}"}
         for k, v in sorted(weighted.items(), key=lambda kv: -kv[1])],
        ["Framework", "Balanced (%)", "Prevalence-weighted (%)", "Delta (pp)"],
        "DSR under a plausible deployment query mix rather than a balanced design",
    )

    return {
        "n_seeds": N_SEEDS,
        "seed_variance": variance,
        "calibration_bins": bins,
        "expected_calibration_error": ece,
        "holm_bonferroni": corrected,
        "end_to_end_all_frameworks": e2e,
        "prevalence": PREVALENCE,
        "prevalence_weighted_dsr": weighted,
    }


if __name__ == "__main__":
    save("exp11_reliability", run())
