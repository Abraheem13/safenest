"""Experiment 14 -- differentially private release of parameters learned from real text.

The learned estimator's parameters (per-tier means and variances of the five
features) are released under (epsilon, delta)-DP by the two-stage mechanism of
`GaussianTierModel.fit_private`, and compared with two references: the
unclipped maximum-likelihood fit, and the same clipped pipeline without noise,
which isolates what clipping costs from what privacy costs.

The cost of privacy is measured directly, as the error of the released means
(in standard deviations of the non-private fit) and as the share of sessions on
which the private and non-private models assign the same tier. Accuracy is
reported too, but on real text it is not a clean measure of that cost: the
learned Gaussian model is miscalibrated, so perturbing its parameters changes
how often a session falls to the protective floor, and accuracy can move in
either direction. Two settings are reported: five folds within PERSUADE, and
leave-one-corpus-out. Each point averages independent releases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from experiments.exp13_real_estimation import WRITTEN, _load_written, _score  # noqa: E402
from experiments.real_common import MissingCorpus  # noqa: E402
from safenest.features import FEATURE_BOUNDS  # noqa: E402
from safenest.learning import (  # noqa: E402
    DEVIATION_CLIP,
    GaussianTierModel,
    expand_admissible,
    session_decisions,
)
from safenest.privacy import analytic_gaussian_sigma  # noqa: E402

EPSILONS = (0.1, 0.3, 1.0, 3.0, 10.0)
DELTA = 1e-5
N_RELEASES = 20
N = 10


def _param_error(private: GaussianTierModel, reference: GaussianTierModel) -> float:
    """Mean absolute error of released means, in reference standard deviations."""
    return float(np.mean([np.abs(private.mean[t] - reference.mean[t]) / reference.sd[t]
                          for t in reference.tiers]))


def _curve(train, test, test_docs, rng) -> dict:
    res = {}
    ml = GaussianTierModel().fit(train)
    clipped = GaussianTierModel().fit_private(train, None, DELTA, rng)
    ref_dec = {}
    for label, model in (("maximum_likelihood", ml), ("clipped_no_noise", clipped)):
        dec = session_decisions(test, model.window_log_scores(test), model.tiers, (N,))
        ref_dec[label] = dec.set_index("doc_id")[f"tier_n{N}"]
        res[label] = _score(dec, test_docs, N)
    for eps in EPSILONS:
        scores, agree, perr = [], [], []
        for _ in range(N_RELEASES):
            m = GaussianTierModel().fit_private(train, eps, DELTA, rng)
            dec = session_decisions(test, m.window_log_scores(test), m.tiers, (N,))
            scores.append(_score(dec, test_docs, N))
            got = dec.set_index("doc_id")[f"tier_n{N}"]
            agree.append(float((got == ref_dec["clipped_no_noise"].reindex(got.index)).mean()))
            perr.append(_param_error(m, clipped))
        res[str(eps)] = {k: {"mean": float(np.mean([s[k] for s in scores])),
                             "sd": float(np.std([s[k] for s in scores]))}
                         for k in ("accuracy", "balanced_accuracy", "under", "over")}
        res[str(eps)]["agreement_with_non_private"] = {"mean": float(np.mean(agree)),
                                                       "sd": float(np.std(agree))}
        res[str(eps)]["mean_error_in_sd"] = {"mean": float(np.mean(perr)),
                                             "sd": float(np.std(perr))}
    return res


def _print(res: dict, title: str) -> None:
    rows = [{"release": k.replace("_", " "), "Agree": "", "Mean err (SD)": "",
             "Bal. acc": pct(res[k]["balanced_accuracy"]),
             "Under": pct(res[k]["under"]), "Over": pct(res[k]["over"])}
            for k in ("maximum_likelihood", "clipped_no_noise")]
    rows += [{"release": f"epsilon = {e:g}",
              "Agree": pct(res[str(e)]["agreement_with_non_private"]["mean"]),
              "Mean err (SD)": f"{res[str(e)]['mean_error_in_sd']['mean']:.3f}",
              "Bal. acc": f"{pct(res[str(e)]['balanced_accuracy']['mean'])} "
                          f"+- {pct(res[str(e)]['balanced_accuracy']['sd'])}",
              "Under": pct(res[str(e)]["under"]["mean"]),
              "Over": pct(res[str(e)]["over"]["mean"])} for e in EPSILONS]
    table(rows, ["release", "Agree", "Mean err (SD)", "Bal. acc", "Under", "Over"], title)


def run() -> dict:
    try:
        docs, wins, _ = _load_written(50)
    except MissingCorpus as exc:
        print(f"  skipped: {exc}")
        return {"skipped": str(exc)}
    rng = rng_for("real_privacy")
    d = len(FEATURE_BOUNDS)
    out: dict = {
        "delta": DELTA, "n_releases": N_RELEASES, "deviation_clip": DEVIATION_CLIP,
        "sensitivity": {"means": float(np.sqrt(d)),
                        "variances": float(DEVIATION_CLIP ** 2 * np.sqrt(d))},
        "sigma": {str(e): [analytic_gaussian_sigma(e / 2, DELTA / 2, float(np.sqrt(d))),
                           analytic_gaussian_sigma(e / 2, DELTA / 2,
                                                   float(DEVIATION_CLIP ** 2 * np.sqrt(d)))]
                  for e in EPSILONS},
    }

    # ---- within PERSUADE -----------------------------------------------------
    p_docs = docs[docs["corpus"] == "persuade"]
    p_wins = wins[wins["corpus"] == "persuade"]
    fold_rng = np.random.default_rng(0)
    fold = {}
    for _, grp in p_docs.groupby("admissible"):
        ids = grp["doc_id"].to_numpy().copy()
        fold_rng.shuffle(ids)
        fold.update({x: i % 5 for i, x in enumerate(ids)})
    per_fold = []
    for k in range(5):
        te = {x for x, f in fold.items() if f == k}
        per_fold.append(_curve(expand_admissible(p_wins[~p_wins["doc_id"].isin(te)]),
                               p_wins[p_wins["doc_id"].isin(te)],
                               p_docs[p_docs["doc_id"].isin(te)], rng))
    within = {}
    for key in per_fold[0]:
        if key in ("maximum_likelihood", "clipped_no_noise"):
            within[key] = {m: float(np.mean([f[key][m] for f in per_fold]))
                           for m in ("accuracy", "balanced_accuracy", "under", "over")}
        else:
            within[key] = {m: {"mean": float(np.mean([f[key][m]["mean"] for f in per_fold])),
                               "sd": float(np.mean([f[key][m]["sd"] for f in per_fold]))}
                           for m in ("accuracy", "balanced_accuracy", "under", "over",
                                     "agreement_with_non_private", "mean_error_in_sd")}
    out["within_persuade"] = within
    _print(within, "Within PERSUADE, five folds (n = 10 windows), %")

    # ---- across corpora ------------------------------------------------------
    out["leave_one_corpus_out"] = {}
    for held in WRITTEN:
        res = _curve(expand_admissible(wins[wins["corpus"] != held]),
                     wins[wins["corpus"] == held], docs[docs["corpus"] == held], rng)
        out["leave_one_corpus_out"][held] = res
        _print(res, f"Trained without {held}, tested on {held} (n = {N} windows), %")
    return out


if __name__ == "__main__":
    save("exp14_real_privacy", run())
