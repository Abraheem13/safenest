"""Experiment 04 -- comparative evaluation (regenerates Tables 14, 15 and 16).

This is the experiment the reviews hit hardest, so it reports:

  * both ground truths side by side -- the circular `matrix` labeller and the
    independent `rubric` labeller ;
  * three age-aware baselines alongside the four age-agnostic ones
    ;
  * Wilson confidence intervals and paired McNemar tests, so the headline gap
    comes with a significance statement ;
  * an end-to-end condition in which tiers are *estimated* rather than known,
    which is the number that actually corresponds to a deployment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.baselines import all_frameworks  # noqa: E402
from safenest.corpus import build_corpus, corpus_summary  # noqa: E402
from safenest.estimator import BayesianAgeEstimator  # noqa: E402
from safenest.labeling import LABELLERS, agreement  # noqa: E402
from safenest.metrics import (  # noqa: E402
    DSR_DEFINITION, correctness_vector, evaluate_framework, mcnemar,
)
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.tiers import ALL_TIERS  # noqa: E402

REFERENCE_BASELINE = "Constitutional AI"       # strongest age-agnostic
STRONGEST_BASELINE = "Age-conditioned (oracle)"  # strongest age-aware


def run() -> dict:
    prompts = build_corpus()
    summary = corpus_summary(prompts)
    print(f"Corpus: {summary['n_prompts']} prompts, seed={summary['seed']}, "
          f"{len(summary['by_category'])} categories x {len(summary['by_tier'])} tiers")

    agree = agreement(prompts)
    print(f"\nGround-truth labeller agreement: exact={agree['exact_agreement']:.3f}, "
          f"Cohen's kappa={agree['cohens_kappa']:.3f}")
    print("  The two labellers are genuinely independent; the matrix labeller is")
    print("  the manuscript's own specification and is reported only for contrast.")

    frameworks = all_frameworks()
    results: dict[str, dict] = {}

    for lname, labeller in LABELLERS.items():
        print(f"\n{'=' * 78}\nGround truth: {lname.upper()} labeller")
        rows = []
        per_framework = {}
        for fname, fw in frameworks.items():
            r = evaluate_framework(fw, prompts, labeller)
            per_framework[fname] = r
            o = r["overall"]
            row = {"Framework": fname}
            for t in ALL_TIERS:
                row[t.label] = pct(r["by_tier"][t.label]["dsr"])
            row["Mean"] = pct(o["dsr"])
            row["95% CI"] = f"[{pct(o['dsr_ci_low'])}, {pct(o['dsr_ci_high'])}]"
            row["Under"] = pct(o["under_protection"])
            row["Over"] = pct(o["over_restriction"])
            rows.append(row)
        table(rows,
              ["Framework", *[t.label for t in ALL_TIERS], "Mean", "95% CI", "Under", "Over"],
              f"Tables 14+15 (regenerated) -- DSR by tier, {lname} ground truth (%)")
        results[lname] = per_framework

        npl = per_framework["NPL (ours)"]["overall"]
        for ref in (REFERENCE_BASELINE, STRONGEST_BASELINE):
            base = per_framework[ref]["overall"]
            gap = 100 * (npl["dsr"] - base["dsr"])
            test = mcnemar(
                correctness_vector(frameworks["NPL (ours)"], prompts, labeller),
                correctness_vector(frameworks[ref], prompts, labeller),
            )
            print(f"  NPL vs {ref}: {gap:+.1f} pp, "
                  f"McNemar chi2={test['chi2']:.1f}, p={test['p_value']:.2e}")

    # ---- per-category breakdown at t2, against the independent rubric -------
    rubric = LABELLERS["rubric"]
    t2 = [p for p in prompts if int(p.tier) == 2]
    rows = []
    for cat in sorted({p.category for p in t2}, key=lambda c: c.value):
        sel = [p for p in t2 if p.category is cat]
        npl_dsr = evaluate_framework(frameworks["NPL (ours)"], sel, rubric)["overall"]["dsr"]
        base_dsr = evaluate_framework(frameworks[REFERENCE_BASELINE], sel, rubric)["overall"]["dsr"]
        strong = evaluate_framework(frameworks[STRONGEST_BASELINE], sel, rubric)["overall"]["dsr"]
        rows.append({
            "Risk category": cat.value,
            "NPL": pct(npl_dsr),
            REFERENCE_BASELINE: pct(base_dsr),
            "delta vs CAI": f"{100 * (npl_dsr - base_dsr):+.1f}",
            STRONGEST_BASELINE: pct(strong),
            "delta vs age-cond": f"{100 * (npl_dsr - strong):+.1f}",
        })
    table(rows, ["Risk category", "NPL", REFERENCE_BASELINE, "delta vs CAI",
                 STRONGEST_BASELINE, "delta vs age-cond"],
          "Table 16 (regenerated) -- per-category DSR at t2, rubric ground truth (%)")

    # ---- end-to-end: estimated rather than oracle tiers ---------------------
    e2e = _end_to_end(prompts, frameworks, rubric)

    return {
        "corpus": summary,
        "dsr_definition": DSR_DEFINITION,
        "labeller_agreement": agree,
        "results": {
            lname: {
                fname: {"overall": r["overall"], "by_tier": r["by_tier"],
                        "by_category": r["by_category"]}
                for fname, r in per.items()
            }
            for lname, per in results.items()
        },
        "per_category_t2_rubric": rows,
        "end_to_end_estimated_tiers": e2e,
    }


def _end_to_end(prompts, frameworks, labeller) -> dict:
    """DSR when the tier is estimated from signals rather than given.

    Every published DSR number in the manuscript assumes the tier is known.
    A deployment never knows it, so this is the number a reader should weigh.
    """
    rng = rng_for("end_to_end")
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))
    cache: dict[int, object] = {}

    def assigned(p):
        if p.idx not in cache:
            tier, _ = est.run_session(p.tier, n_interactions=5, rng=rng)
            cache[p.idx] = tier
        return cache[p.idx]

    r = evaluate_framework(frameworks["NPL (ours)"], prompts, labeller, assigned_tier=assigned)
    oracle = evaluate_framework(frameworks["NPL (ours)"], prompts, labeller)["overall"]
    o = r["overall"]
    print(f"\nEnd-to-end with estimated tiers (5 interactions), rubric ground truth:")
    print(f"  DSR {pct(o['dsr'])}% (oracle tiers: {pct(oracle['dsr'])}%), "
          f"under-protection {pct(o['under_protection'])}%, "
          f"over-restriction {pct(o['over_restriction'])}%")
    return {"estimated": o, "oracle": oracle}


if __name__ == "__main__":
    save("exp04_comparative", run())
