"""Experiment 16 -- the youngest tiers, from real children's speech (CHILDES).

Two narrative corpora cover ages 4-11: Gillam (5;0-11;11) and ENNI (4-9). In
both, every child tells the same stories whatever their age, so unlike the
essay corpora there is no topic that reveals the age. Tiers t1 (3-6), t2 (7-9)
and t3 (10-12) are estimated from transcribed child utterances; an interaction
is a 50-word window of consecutive utterances, which matches the voice-first
interaction mode the policy assigns to the youngest tier.

Reported:
  * five-fold cross-validation within each corpus and on the pooled corpora,
    folds stratified by tier and grouped by child;
  * cross-corpus transfer in both directions, restricted to the tiers the
    training corpus contains;
  * under- and over-protection for children with and without language
    impairment, compared within tier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, save, table  # noqa: E402
from experiments.exp13_real_estimation import _score  # noqa: E402
from experiments.real_common import HORIZONS, MissingCorpus, load_corpus, model_zoo  # noqa: E402
from safenest.learning import expand_admissible, outcome, session_decisions  # noqa: E402
from safenest.metrics import wilson_interval  # noqa: E402

SPOKEN = ("gillam", "enni")
MODELS = ("NPL (specified)", "NPL (learned)", "Logistic (features)",
          "Gradient boosting (features)", "Logistic (TF-IDF)")
N = 10


def _cv(docs, wins, zoo, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fold = {}
    for _, grp in docs.groupby("tier"):
        ids = grp["doc_id"].to_numpy().copy()
        rng.shuffle(ids)
        fold.update({d: i % 5 for i, d in enumerate(ids)})
    decisions = {name: [] for name in MODELS}
    for k in range(5):
        test_ids = {d for d, f in fold.items() if f == k}
        train = expand_admissible(wins[~wins["doc_id"].isin(test_ids)])
        test = wins[wins["doc_id"].isin(test_ids)]
        for name in MODELS:
            model = zoo[name]().fit(train)
            decisions[name].append(
                session_decisions(test, model.window_log_scores(test), model.tiers, HORIZONS))
    scores, frames = {}, []
    for name, parts in decisions.items():
        dec = pd.concat(parts)
        scores[name] = {f"n{n}": _score(dec, docs, n) for n in HORIZONS}
        frames.append(dec.assign(model=name))
    return scores, pd.concat(frames)


def _transfer(src, dst, zoo) -> dict:
    (s_docs, s_wins), (d_docs, d_wins) = src, dst
    tiers = set(s_docs["tier"])
    d_docs = d_docs[d_docs["tier"].isin(tiers)]
    d_wins = d_wins[d_wins["doc_id"].isin(d_docs["doc_id"])]
    out = {}
    train = expand_admissible(s_wins)
    for name in MODELS:
        model = zoo[name]().fit(train)
        dec = session_decisions(d_wins, model.window_log_scores(d_wins), model.tiers, HORIZONS)
        out[name] = {f"n{n}": _score(dec, d_docs, n) for n in HORIZONS}
    return out


def _impairment(decisions: pd.DataFrame, docs: pd.DataFrame) -> dict:
    d = decisions.merge(docs[["doc_id", "tier", "admissible", "impairment"]], on="doc_id")
    d = d[d["impairment"].notna()]
    d["outcome"] = [outcome(a, adm) for a, adm in zip(d[f"tier_n{N}"], d["admissible"])]
    out = {}
    for name, m in d.groupby("model"):
        out[name] = {}
        for status, label in ((1, "language impairment"), (0, "typically developing")):
            sub = m[m["impairment"] == status]
            if sub.empty:
                continue
            k_under = int((sub["outcome"] == "under").sum())
            out[name][label] = {
                "n": int(len(sub)),
                "accuracy": float((sub["outcome"] == "correct").mean()),
                "under": float(k_under / len(sub)),
                "under_ci": wilson_interval(k_under, len(sub)),
                "over": float((sub["outcome"] == "over").mean()),
                "by_tier": {f"t{t}": {"n": int(len(s)),
                                      "under": float((s["outcome"] == "under").mean()),
                                      "over": float((s["outcome"] == "over").mean())}
                            for t, s in sub.groupby("tier")},
            }
    return out


def run() -> dict:
    loaded = {}
    for name in SPOKEN:
        try:
            loaded[name] = load_corpus(name)
        except MissingCorpus as exc:
            print(f"  {exc}")
    if not loaded:
        return {"skipped": "no CHILDES corpus has been prepared; see data/README.md"}
    zoo = model_zoo()
    out: dict = {"corpora": {}, "within_corpus": {}, "transfer": {}}
    all_decisions = []
    for name, (docs, wins) in loaded.items():
        out["corpora"][name] = {
            "children": int(len(docs)), "windows": int(len(wins)),
            "by_tier": {f"t{t}": int(n)
                        for t, n in docs["tier"].value_counts().sort_index().items()},
            "impaired": int((docs["impairment"] == 1).sum()),
            "age_range": [float(docs["age_years"].min()), float(docs["age_years"].max())],
        }
        scores, dec = _cv(docs, wins, zoo)
        out["within_corpus"][name] = scores
        all_decisions.append((docs, dec))
        table(
            [{"Estimator": m, "Acc": pct(s[f"n{N}"]["accuracy"]),
              "Bal. acc": pct(s[f"n{N}"]["balanced_accuracy"]),
              "Under": pct(s[f"n{N}"]["under"]), "Over": pct(s[f"n{N}"]["over"])}
             for m, s in scores.items()],
            ["Estimator", "Acc", "Bal. acc", "Under", "Over"],
            f"{name}: five-fold cross-validation (n = {N} windows), %",
        )
    if len(loaded) == 2:
        docs = pd.concat([d for d, _ in loaded.values()], ignore_index=True)
        wins = pd.concat([w for _, w in loaded.values()], ignore_index=True)
        pooled, pooled_dec = _cv(docs, wins, zoo)
        out["within_corpus"]["pooled"] = pooled
        out["transfer"]["gillam_to_enni"] = _transfer(loaded["gillam"], loaded["enni"], zoo)
        out["transfer"]["enni_to_gillam"] = _transfer(loaded["enni"], loaded["gillam"], zoo)
        out["impairment"] = _impairment(pooled_dec, docs)
    else:
        docs, dec = all_decisions[0]
        out["impairment"] = _impairment(dec, docs)
    return out


if __name__ == "__main__":
    save("exp16_real_spoken", run())
