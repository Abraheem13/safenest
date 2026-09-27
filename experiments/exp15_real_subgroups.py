"""Experiment 15 -- who the learned estimator fails, on real subgroups.

PERSUADE records, for each writer, English-learner status, identified
disability, economic disadvantage, gender and race or ethnicity. Every PERSUADE
essay is scored by estimators trained without it, on the rest of PERSUADE and
on ASAP, with and without ELLIPSE.

Folds hold out whole prompts, so no essay is scored by a model that saw its
prompt: grade and prompt are confounded in PERSUADE, and a model that has seen
the prompt can read the grade off the topic. The single grade-6 prompt cannot
be held out without removing every t3 example, so its essays are split at
random across the folds. Only the feature-based estimators are compared; the
text-based ones exploit topic (Experiment 13) and would measure it instead.

Subgroups are compared *within grade*, and only over grades in which both
groups of a comparison appear. Grade-standardised rates weight each grade by
its share of the corpus.

Training with and without ELLIPSE tests one explanation directly. ELLIPSE
writers are all English learners and mostly in grades 11-12, and PERSUADE's
grade-12 writers are mostly English learners, so the oldest tier's training
data over-represent learner English.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from experiments.exp13_real_estimation import _load_written  # noqa: E402
from experiments.real_common import MissingCorpus, model_zoo  # noqa: E402
from safenest.learning import expand_admissible, outcome, session_decisions  # noqa: E402
from safenest.metrics import wilson_interval  # noqa: E402

MODELS = ("NPL (learned)", "Gradient boosting (features)", "Logistic (features)")
TRAINING = {"with ELLIPSE": ("persuade", "asap", "ellipse"),
            "without ELLIPSE": ("persuade", "asap")}
N = 10
SUBGROUPS = {
    "ell": {1: "English learner", 0: "Not English learner"},
    "disability": {1: "Identified disability", 0: "No identified disability"},
    "econ": {1: "Economically disadvantaged", 0: "Not economically disadvantaged"},
    "gender": {"F": "Female", "M": "Male"},
}
N_BOOT = 2000


def _folds(p_docs: pd.DataFrame) -> dict[str, int]:
    """Prompt-disjoint folds; the grade-6 prompt's essays are spread at random."""
    rng = np.random.default_rng(0)
    t3_prompts = set(p_docs.loc[p_docs["admissible"] == "3", "prompt"])
    others = sorted(set(p_docs["prompt"]) - t3_prompts)
    prompt_fold = {pr: i % 5 for i, pr in enumerate(others)}
    fold = {}
    for d, pr in zip(p_docs["doc_id"], p_docs["prompt"]):
        if pr in prompt_fold:
            fold[d] = prompt_fold[pr]
    t3 = p_docs.loc[p_docs["prompt"].isin(t3_prompts), "doc_id"].to_numpy().copy()
    rng.shuffle(t3)
    fold.update({d: i % 5 for i, d in enumerate(t3)})
    return fold


def _predict_persuade(docs, wins, zoo, corpora) -> pd.DataFrame:
    p_docs = docs[docs["corpus"] == "persuade"].copy()
    fold = _folds(p_docs)
    held_prompts = {k: set(p_docs.loc[p_docs["doc_id"].map(fold) == k, "prompt"])
                    for k in range(5)}
    rows = []
    for k in range(5):
        test_ids = {d for d, f in fold.items() if f == k}
        # Exclude every PERSUADE essay on a held-out prompt, not only the test
        # essays, except for the shared grade-6 prompt.
        shared = set(p_docs.loc[p_docs["admissible"] == "3", "prompt"])
        excluded = set(p_docs.loc[p_docs["prompt"].isin(held_prompts[k] - shared), "doc_id"])
        excluded |= test_ids
        train = wins[wins["corpus"].isin(corpora) & ~wins["doc_id"].isin(excluded)]
        test = wins[wins["doc_id"].isin(test_ids)]
        train = expand_admissible(train)
        for name in MODELS:
            model = zoo[name]().fit(train)
            dec = session_decisions(test, model.window_log_scores(test), model.tiers, (N,))
            dec["model"] = name
            rows.append(dec)
    dec = pd.concat(rows).merge(p_docs, on="doc_id")
    dec["outcome"] = [outcome(a, adm) for a, adm in zip(dec[f"tier_n{N}"], dec["admissible"])]
    return dec


def _standardised(sub: pd.DataFrame, weights: pd.Series, what: str) -> float:
    by_grade = sub.groupby("grade")["outcome"].apply(lambda s: float((s == what).mean()))
    w = weights.reindex(by_grade.index).fillna(0.0)
    return float((by_grade * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def _compare(dec: pd.DataFrame, column: str, labels: dict, rng) -> dict:
    grade_weights = dec.drop_duplicates("doc_id")["grade"].value_counts(normalize=True)
    out = {}
    values = list(labels)
    present = dec[dec[column].isin(values[:2])].groupby("grade")[column].nunique()
    common = present[present == 2].index
    for name in MODELS:
        m = dec[(dec["model"] == name) & dec[column].notna() & dec["grade"].isin(common)]
        groups = {}
        for value, label in labels.items():
            sub = m[m[column] == value]
            if sub.empty:
                continue
            groups[label] = {
                "n": int(len(sub)),
                "under_raw": float((sub["outcome"] == "under").mean()),
                "over_raw": float((sub["outcome"] == "over").mean()),
                "under_std": _standardised(sub, grade_weights, "under"),
                "over_std": _standardised(sub, grade_weights, "over"),
                "under_ci": wilson_interval(int((sub["outcome"] == "under").sum()), len(sub)),
            }
        # Bootstrap the grade-standardised difference between the two groups.
        # Within each grade, counts are resampled as binomial draws at the
        # observed rate, which is the exact bootstrap of a per-grade proportion.
        diffs = {}
        for what in ("under", "over"):
            draws = []
            for value in values[:2]:
                sub = m[m[column] == value]
                stats = sub.groupby("grade")["outcome"].agg(
                    n="size", k=lambda s, w=what: int((s == w).sum()))
                w = grade_weights.reindex(stats.index).fillna(0.0).to_numpy()
                n = stats["n"].to_numpy()
                prob = stats["k"].to_numpy() / n
                sims = rng.binomial(n[None, :], prob[None, :], size=(N_BOOT, len(n))) / n
                draws.append((sims * w).sum(axis=1) / w.sum())
            diffs[what] = draws[0] - draws[1]
        out[name] = {
            "groups": groups,
            "difference": {what: {"estimate": groups[labels[values[0]]][f"{what}_std"]
                                  - groups[labels[values[1]]][f"{what}_std"],
                                  "ci": [float(np.quantile(v, 0.025)),
                                         float(np.quantile(v, 0.975))]}
                           for what, v in diffs.items()},
        }
    return out


def run() -> dict:
    try:
        docs, wins, _ = _load_written(50)
    except MissingCorpus as exc:
        print(f"  skipped: {exc}")
        return {"skipped": str(exc)}
    zoo = model_zoo()
    rng = rng_for("real_subgroups")
    out: dict = {"models": list(MODELS), "n_windows": N, "training": {}}
    for label, corpora in TRAINING.items():
        dec = _predict_persuade(docs, wins, zoo, corpora)
        results = {}
        for column, labels in SUBGROUPS.items():
            results[column] = _compare(dec, column, labels, rng)
            rows = []
            for name, r in results[column].items():
                for group, g in r["groups"].items():
                    rows.append({"Estimator": name, "Group": group, "n": str(g["n"]),
                                 "Under (std)": pct(g["under_std"]),
                                 "Over (std)": pct(g["over_std"])})
            table(rows, ["Estimator", "Group", "n", "Under (std)", "Over (std)"],
                  f"PERSUADE, training {label}: grade-standardised rates by {column} (%)")
        grade_weights = dec.drop_duplicates("doc_id")["grade"].value_counts(normalize=True)
        race = {}
        for name in MODELS:
            m = dec[dec["model"] == name]
            race[name] = {r: {"n": int(len(x)),
                              "under_std": _standardised(x, grade_weights, "under"),
                              "over_std": _standardised(x, grade_weights, "over")}
                          for r, x in m.groupby("race")}
        overall = {name: {k: float((dec[dec["model"] == name]["outcome"] == k).mean())
                          for k in ("correct", "under", "over")}
                   for name in MODELS}
        out["training"][label] = {"corpora": list(corpora), "overall": overall,
                                  "subgroups": results, "race_ethnicity": race}
    return out


if __name__ == "__main__":
    save("exp15_real_subgroups", run())
