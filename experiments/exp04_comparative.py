"""Experiment 04 -- comparative evaluation on the structured policy corpus.

Reports, against both ground truths (the circular policy-derived labels and the
independent rubric):

  * nine frameworks at their default settings, with Wilson intervals;
  * every baseline tuned to its best: the age-agnostic rules over their
    severity thresholds and the age-band oracle over all six contiguous
    three-band partitions, each selected on a separate tuning corpus
    (seed + 1) and scored on the evaluation corpus;
  * NPL's severity threshold tuned the same way, for symmetry;
  * paired bootstrap 95% intervals for every DSR difference, with McNemar's
    test as a secondary statistic;
  * the per-cell decomposition of NPL's margin over the age-band oracle, so
    the reader can see which (tier, category) cells produce it;
  * the earlier crisis specification (refusal above the severity threshold);
  * an end-to-end condition in which tiers are estimated, not given.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.baselines import (  # noqa: E402
    AGE_AGNOSTIC_FACTORIES,
    DEFAULT_BANDS,
    NPL_FULL,
    THRESHOLD_GRID,
    all_frameworks,
    contiguous_three_band_partitions,
    make_age_conditioned,
    make_npl,
    npl_engine,
)
from safenest.corpus import RANDOM_SEED, build_corpus, corpus_summary  # noqa: E402
from safenest.estimator import BayesianAgeEstimator  # noqa: E402
from safenest.labeling import LABELLERS, agreement, decision_to_label, rubric_label  # noqa: E402
from safenest.metrics import (  # noqa: E402
    DSR_DEFINITION,
    correctness_vector,
    evaluate_framework,
    mcnemar,
    paired_bootstrap,
)
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.tiers import ALL_TIERS  # noqa: E402

TUNING_SEED = RANDOM_SEED + 1
REFERENCE_BASELINE = "Constitutional rule"   # strongest age-agnostic at defaults
STRONGEST_BASELINE = "Age-band oracle"       # strongest age-aware
SEVERITY_TUNING_GRID = tuple(round(float(x), 2) for x in np.linspace(0.05, 0.95, 19))


_LABEL_CACHE: dict[int, list] = {}


def _dsr(framework, prompts) -> float:
    """Overall DSR only; the labels of a corpus are computed once."""
    key = id(prompts)
    if key not in _LABEL_CACHE:
        _LABEL_CACHE[key] = [rubric_label(p) for p in prompts]
    labels = _LABEL_CACHE[key]
    return float(np.mean([decision_to_label(framework(p, p.tier)) == g
                          for p, g in zip(prompts, labels)]))


def _tune_age_agnostic(tuning, evaluation) -> dict:
    out = {}
    for name, (factory, params) in AGE_AGNOSTIC_FACTORIES.items():
        best = max(
            itertools.product(THRESHOLD_GRID, repeat=len(params)),
            key=lambda combo: _dsr(factory(**dict(zip(params, combo))), tuning),
        )
        fw = factory(**dict(zip(params, best)))
        r = evaluate_framework(fw, evaluation, rubric_label)["overall"]
        out[name] = {"thresholds": dict(zip(params, best)), "tuning_dsr": _dsr(fw, tuning),
                     **r}
    return out


def _tune_bands(tuning, evaluation) -> dict:
    by_partition = {}
    for bands in contiguous_three_band_partitions():
        fw = make_age_conditioned(bands)
        by_partition[str(bands)] = {
            "tuning_dsr": _dsr(fw, tuning),
            **evaluate_framework(fw, evaluation, rubric_label)["overall"],
        }
    best = max(by_partition, key=lambda k: by_partition[k]["tuning_dsr"])
    return {"by_partition": by_partition, "selected": best, "default": str(DEFAULT_BANDS)}


def _tune_npl(tuning, evaluation) -> dict:
    best = max(SEVERITY_TUNING_GRID,
               key=lambda th: _dsr(make_npl(npl_engine(severity_threshold=th)), tuning))
    fw = make_npl(npl_engine(severity_threshold=best))
    return {"severity_threshold": best,
            **evaluate_framework(fw, evaluation, rubric_label)["overall"]}


def _cell_decomposition(a_vec, b_vec, prompts) -> list[dict]:
    """Contribution of each (tier, category) cell to DSR(a) - DSR(b), in points."""
    n = len(prompts)
    cells: dict[tuple[str, str], float] = {}
    for p, x, y in zip(prompts, a_vec, b_vec):
        key = (p.tier.label, p.category.value)
        cells[key] = cells.get(key, 0.0) + 100.0 * (int(x) - int(y)) / n
    return [{"tier": k[0], "category": k[1], "points": v}
            for k, v in sorted(cells.items(), key=lambda kv: -abs(kv[1]))]


def run() -> dict:
    prompts = build_corpus()
    tuning = build_corpus(seed=TUNING_SEED)
    summary = corpus_summary(prompts)
    print(f"Corpus: {summary['n_prompts']} prompts (seed {RANDOM_SEED}); tuning corpus "
          f"seed {TUNING_SEED}")

    agree = agreement(prompts)
    print(f"Labeller agreement: exact={agree['exact_agreement']:.3f}, "
          f"kappa={agree['cohens_kappa']:.3f}")

    frameworks = all_frameworks()
    results: dict[str, dict] = {}
    for lname, labeller in LABELLERS.items():
        per = {name: evaluate_framework(fw, prompts, labeller) for name, fw in frameworks.items()}
        results[lname] = per
        table(
            [{"Framework": n, **{t.label: pct(r["by_tier"][t.label]["dsr"]) for t in ALL_TIERS},
              "DSR": pct(r["overall"]["dsr"]),
              "95% CI": f"[{pct(r['overall']['dsr_ci_low'])}, {pct(r['overall']['dsr_ci_high'])}]",
              "Under": pct(r["overall"]["under_protection"]),
              "Over": pct(r["overall"]["over_restriction"])}
             for n, r in per.items()],
            ["Framework", *[t.label for t in ALL_TIERS], "DSR", "95% CI", "Under", "Over"],
            f"DSR by tier, {lname} ground truth (%)",
        )

    # ---- paired differences --------------------------------------------------
    rng = rng_for("comparative_bootstrap")
    npl_vec = correctness_vector(frameworks[NPL_FULL], prompts, rubric_label)
    differences = {}
    for name, fw in frameworks.items():
        if name == NPL_FULL:
            continue
        other = correctness_vector(fw, prompts, rubric_label)
        differences[name] = {**paired_bootstrap(npl_vec, other, rng),
                             "mcnemar": mcnemar(npl_vec, other)}
    table(
        [{"NPL versus": k, "Diff (pp)": f"{100 * v['difference']:+.1f}",
          "95% CI (pp)": f"[{100 * v['ci_low']:+.1f}, {100 * v['ci_high']:+.1f}]"}
         for k, v in differences.items()],
        ["NPL versus", "Diff (pp)", "95% CI (pp)"],
        "Paired bootstrap of the DSR difference (2,000 resamples)",
    )

    # ---- tuned baselines -----------------------------------------------------
    tuned_agnostic = _tune_age_agnostic(tuning, prompts)
    bands = _tune_bands(tuning, prompts)
    tuned_npl = _tune_npl(tuning, prompts)
    rows = [{"Framework": f"{k} (tuned)", "Thresholds": str(v["thresholds"]),
             "DSR": pct(v["dsr"]), "Under": pct(v["under_protection"])}
            for k, v in tuned_agnostic.items()]
    sel = bands["by_partition"][bands["selected"]]
    rows.append({"Framework": "Age-band oracle (tuned bands)", "Thresholds": bands["selected"],
                 "DSR": pct(sel["dsr"]), "Under": pct(sel["under_protection"])})
    rows.append({"Framework": "NPL (tuned severity)",
                 "Thresholds": str(tuned_npl["severity_threshold"]),
                 "DSR": pct(tuned_npl["dsr"]), "Under": pct(tuned_npl["under_protection"])})
    table(rows, ["Framework", "Thresholds", "DSR", "Under"],
          "Every framework tuned on the tuning corpus, scored on the evaluation corpus")
    table(
        [{"Bands": k, "DSR": pct(v["dsr"]), "Under": pct(v["under_protection"]),
          "Over": pct(v["over_restriction"])} for k, v in bands["by_partition"].items()],
        ["Bands", "DSR", "Under", "Over"],
        "Age-band oracle under every contiguous three-band partition",
    )

    # ---- where the margin over the oracle comes from ---------------------------
    oracle_vec = correctness_vector(frameworks[STRONGEST_BASELINE], prompts, rubric_label)
    decomposition = _cell_decomposition(npl_vec, oracle_vec, prompts)
    print("\nLargest cell contributions to NPL minus age-band oracle (points):")
    for c in decomposition[:8]:
        print(f"  {c['tier']} {c['category']:28s} {c['points']:+.2f}")

    # ---- the earlier crisis specification -------------------------------------
    earlier = make_npl(npl_engine(crisis_referral=False))
    earlier_r = evaluate_framework(earlier, prompts, rubric_label)

    # ---- per-category breakdown at t2 -----------------------------------------
    t2 = [p for p in prompts if int(p.tier) == 2]
    per_category_t2 = {
        cat.value: {name: evaluate_framework(fw, [p for p in t2 if p.category is cat],
                                             rubric_label)["overall"]["dsr"]
                    for name, fw in frameworks.items()}
        for cat in sorted({p.category for p in t2}, key=lambda c: c.value)
    }

    e2e = _end_to_end(prompts, frameworks)

    return {
        "corpus": summary,
        "tuning_seed": TUNING_SEED,
        "dsr_definition": DSR_DEFINITION,
        "labeller_agreement": agree,
        "results": {
            lname: {name: {"overall": r["overall"], "by_tier": r["by_tier"],
                           "by_category": r["by_category"]}
                    for name, r in per.items()}
            for lname, per in results.items()
        },
        "paired_differences": differences,
        "tuned_age_agnostic": tuned_agnostic,
        "band_partitions": bands,
        "tuned_npl": tuned_npl,
        "margin_decomposition_vs_oracle": decomposition,
        "earlier_crisis_specification": {"overall": earlier_r["overall"],
                                         "by_category": earlier_r["by_category"]},
        "per_category_t2_rubric_all_frameworks": per_category_t2,
        "end_to_end_estimated_tiers": e2e,
    }


def _end_to_end(prompts, frameworks) -> dict:
    """NPL's DSR when the tier is estimated from five simulated interactions."""
    rng = rng_for("end_to_end")
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))
    cache: dict[int, object] = {}

    def assigned(p):
        if p.idx not in cache:
            cache[p.idx] = est.run_session(p.tier, n_interactions=5, rng=rng)[0]
        return cache[p.idx]

    r = evaluate_framework(frameworks[NPL_FULL], prompts, rubric_label, assigned_tier=assigned)
    oracle = evaluate_framework(frameworks[NPL_FULL], prompts, rubric_label)["overall"]
    o = r["overall"]
    print(f"\nEnd to end (5 interactions): DSR {pct(o['dsr'])}% "
          f"(true tiers {pct(oracle['dsr'])}%), under {pct(o['under_protection'])}%, "
          f"over {pct(o['over_restriction'])}%")
    return {"estimated": o, "oracle": oracle}


if __name__ == "__main__":
    save("exp04_comparative", run())
