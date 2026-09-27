"""Experiment 16 -- the youngest tiers, from real children's speech (CHILDES).

Two narrative corpora cover ages 4-11: Gillam (5;0-11;11) and ENNI (4-9). In
both, every child tells the same stories whatever their age, so unlike the
essay corpora there is no topic that reveals the age. Tiers t1 (3-6), t2 (7-9)
and t3 (10-12) are estimated from transcribed child utterances; an interaction
is a 50-word window of consecutive utterances, which matches the voice-first
interaction mode the policy assigns to the youngest tier.

Reported:
  * five-fold cross-validation within each corpus and on the pooled corpora,
    folds stratified by tier and grouped by child, with 95% bootstrap
    intervals over children and a paired bootstrap between the two best
    estimators;
  * cross-corpus transfer in both directions, restricted to the tiers the
    training corpus contains;
  * under- and over-protection for children with and without language
    impairment, standardised to the pooled tier distribution over the tiers
    in which both groups appear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from experiments.exp13_real_estimation import (  # noqa: E402
    _outcomes,
    _score,
    bootstrap_ci,
    paired_difference,
)
from experiments.real_common import (  # noqa: E402
    HORIZONS,
    MissingCorpus,
    embeddings_for,
    load_corpus,
    model_zoo,
)
from safenest.learning import expand_admissible, outcome, session_decisions  # noqa: E402
from safenest.metrics import wilson_interval  # noqa: E402

SPOKEN = ("gillam", "enni")
MODELS = ("NPL (specified)", "NPL (learned)", "Logistic (features)",
          "Gradient boosting (features)", "Logistic (TF-IDF)", "Logistic (MiniLM)")
N = 10
N_BOOT = 2000


def _with_intervals(scores: dict, frames: dict, rng) -> dict:
    """Attach bootstrap intervals at n = N and compare the two best estimators."""
    for name, frame in frames.items():
        scores[name][f"n{N}"]["ci"] = bootstrap_ci(frame, rng)
    bal = {m: scores[m][f"n{N}"]["balanced_accuracy"] for m in frames}
    ranked = sorted(bal, key=bal.get, reverse=True)
    scores["_best_vs_runner_up"] = {
        "best": ranked[0], "runner_up": ranked[1],
        **paired_difference(frames[ranked[0]], frames[ranked[1]], "balanced_accuracy", rng)}
    return scores


def _cv(docs, wins, zoo, rng, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    fold_rng = np.random.default_rng(seed)
    fold = {}
    for _, grp in docs.groupby("tier"):
        ids = grp["doc_id"].to_numpy().copy()
        fold_rng.shuffle(ids)
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
    scores, frames, all_dec = {}, {}, []
    for name, parts in decisions.items():
        dec = pd.concat(parts)
        scores[name] = {f"n{n}": _score(dec, docs, n) for n in HORIZONS}
        frames[name] = _outcomes(dec, docs, N)
        all_dec.append(dec.assign(model=name))
    return _with_intervals(scores, frames, rng), pd.concat(all_dec)


def _transfer(src, dst, zoo, rng) -> dict:
    (s_docs, s_wins), (d_docs, d_wins) = src, dst
    tiers = set(s_docs["tier"])
    d_docs = d_docs[d_docs["tier"].isin(tiers)]
    d_wins = d_wins[d_wins["doc_id"].isin(d_docs["doc_id"])]
    out, frames = {}, {}
    train = expand_admissible(s_wins)
    for name in MODELS:
        model = zoo[name]().fit(train)
        dec = session_decisions(d_wins, model.window_log_scores(d_wins), model.tiers, HORIZONS)
        out[name] = {f"n{n}": _score(dec, d_docs, n) for n in HORIZONS}
        frames[name] = _outcomes(dec, d_docs, N)
    return _with_intervals(out, frames, rng)


def _standardised(sub: pd.DataFrame, weights: pd.Series, what: str) -> float:
    rates = sub.groupby("tier")["outcome"].apply(lambda s: (s == what).mean())
    w = weights.reindex(rates.index).fillna(0.0)
    return float((rates * w).sum() / w.sum())


def _impairment(decisions: pd.DataFrame, docs: pd.DataFrame, rng) -> dict:
    """Impaired minus typically developing, standardised over shared tiers.

    Within each tier the per-group counts are resampled as binomial draws at
    the observed rate (the exact bootstrap of a per-tier proportion).
    """
    d = decisions.merge(docs[["doc_id", "tier", "admissible", "impairment"]], on="doc_id")
    d = d[d["impairment"].notna()].copy()
    d["outcome"] = [outcome(a, adm) for a, adm in zip(d[f"tier_n{N}"], d["admissible"])]
    both = d.groupby("tier")["impairment"].nunique()
    shared = both[both == 2].index
    d = d[d["tier"].isin(shared)]
    weights = d.drop_duplicates("doc_id")["tier"].value_counts(normalize=True)
    out: dict = {"tiers": [f"t{t}" for t in sorted(shared)]}
    for name, m in d.groupby("model"):
        res: dict = {}
        for status, label in ((1, "language impairment"), (0, "typically developing")):
            sub = m[m["impairment"] == status]
            k_under = int((sub["outcome"] == "under").sum())
            res[label] = {
                "n": int(len(sub)),
                "accuracy": float((sub["outcome"] == "correct").mean()),
                "under": float(k_under / len(sub)),
                "under_ci": wilson_interval(k_under, len(sub)),
                "over": float((sub["outcome"] == "over").mean()),
                "under_std": _standardised(sub, weights, "under"),
                "over_std": _standardised(sub, weights, "over"),
                "by_tier": {f"t{t}": {"n": int(len(s)),
                                      "under": float((s["outcome"] == "under").mean()),
                                      "over": float((s["outcome"] == "over").mean())}
                            for t, s in sub.groupby("tier")},
            }
        diff = {}
        for what in ("under", "over"):
            draws = []
            for status in (1, 0):
                stats = m[m["impairment"] == status].groupby("tier")["outcome"].agg(
                    n="size", k=lambda s, w=what: int((s == w).sum()))
                w = weights.reindex(stats.index).fillna(0.0).to_numpy()
                n = stats["n"].to_numpy()
                sims = rng.binomial(n[None, :], (stats["k"].to_numpy() / n)[None, :],
                                    size=(N_BOOT, len(n))) / n
                draws.append((sims * w).sum(axis=1) / w.sum())
            boot = draws[0] - draws[1]
            est = (res["language impairment"][f"{what}_std"]
                   - res["typically developing"][f"{what}_std"])
            diff[what] = {"estimate": est,
                          "ci": [float(np.percentile(boot, 2.5)),
                                 float(np.percentile(boot, 97.5))]}
        res["difference"] = diff
        out[name] = res
    return out


def _print(scores: dict, title: str) -> None:
    table([{"Estimator": m, "Bal. acc": pct(s[f"n{N}"]["balanced_accuracy"]),
            "Under": pct(s[f"n{N}"]["under"]), "Over": pct(s[f"n{N}"]["over"])}
           for m, s in scores.items() if not m.startswith("_")],
          ["Estimator", "Bal. acc", "Under", "Over"], title)
    b = scores["_best_vs_runner_up"]
    print(f"  {b['best']} minus {b['runner_up']}: {pct(b['difference'])} "
          f"[{pct(b['ci'][0])}, {pct(b['ci'][1])}]")


def run() -> dict:
    loaded = {}
    for name in SPOKEN:
        try:
            loaded[name] = load_corpus(name)
        except MissingCorpus as exc:
            print(f"  {exc}")
    if not loaded:
        return {"skipped": "no CHILDES corpus has been prepared; see data/README.md"}
    rng = rng_for("exp16_bootstrap")

    # One embedding matrix for the pooled windows; each corpus keeps its rows.
    names = list(loaded)
    pooled_wins = pd.concat([loaded[n][1] for n in names], ignore_index=True)
    pooled_wins["_row"] = np.arange(len(pooled_wins))
    pooled_docs = pd.concat([loaded[n][0] for n in names], ignore_index=True)
    zoo = model_zoo(embeddings_for("spoken", pooled_wins))
    per = {n: (pooled_docs[pooled_docs["corpus"] == n],
               pooled_wins[pooled_wins["corpus"] == n]) for n in names}

    out: dict = {"window_words": 50, "corpora": {}, "within_corpus": {}, "transfer": {}}
    for name, (docs, wins) in per.items():
        out["corpora"][name] = {
            "children": int(len(docs)), "windows": int(len(wins)),
            "by_tier": {f"t{t}": int(n)
                        for t, n in docs["tier"].value_counts().sort_index().items()},
            "impaired": int((docs["impairment"] == 1).sum()),
            "age_range": [float(docs["age_years"].min()), float(docs["age_years"].max())],
            "windows_per_child_median": float(wins.groupby("doc_id").size().median()),
        }
        scores, _ = _cv(docs, wins, zoo, rng)
        out["within_corpus"][name] = scores
        _print(scores, f"{name}: five-fold cross-validation (n = {N} windows), %")

    if len(per) == 2:
        pooled, pooled_dec = _cv(pooled_docs, pooled_wins, zoo, rng)
        out["within_corpus"]["pooled"] = pooled
        _print(pooled, f"pooled: five-fold cross-validation (n = {N} windows), %")
        out["transfer"]["gillam_to_enni"] = _transfer(per["gillam"], per["enni"], zoo, rng)
        out["transfer"]["enni_to_gillam"] = _transfer(per["enni"], per["gillam"], zoo, rng)
        for k, v in out["transfer"].items():
            _print(v, f"transfer {k} (n = {N} windows), %")
        out["impairment"] = _impairment(pooled_dec, pooled_docs, rng)
    return out


if __name__ == "__main__":
    save("exp16_real_spoken", run())
