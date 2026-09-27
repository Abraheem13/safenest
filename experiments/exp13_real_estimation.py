"""Experiment 13 -- learning the tier estimator from real children's writing.

Three written corpora (PERSUADE 2.0, ASAP, ELLIPSE) cover tiers t3-t5. Every
estimator is trained on two corpora and tested on the third
(leave-one-corpus-out), so no test essay shares a collection, and almost none a
prompt, with the training data. Boundary grades (7 and 10) enter training with
half weight on each adjacent tier and are scored as correct on either.

Reported for each estimator and held-out corpus, after n = 1, 3, 5 and 10
interactions (50-word windows): accuracy, balanced accuracy over admissible-tier
groups, under-protection (a less protective tier than any admissible one) and
over-protection, plus the share of sessions sent to the t1 floor.

Also reported:
  * the topic confound -- random-fold versus prompt-disjoint cross-validation
    inside PERSUADE, where every prompt was set to a single grade;
  * window-size sensitivity (25 and 100 words) for the learned estimators;
  * Proposition 1 evaluated at the learned parameters against the observed
    error, which tests the bound where its independence assumption fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from experiments.real_common import (  # noqa: E402
    HORIZONS,
    MissingCorpus,
    embeddings_for,
    load_corpus,
    model_zoo,
)
from safenest.data.corpora import text_key  # noqa: E402
from safenest.learning import (  # noqa: E402
    GaussianTierModel,
    expand_admissible,
    outcome,
    session_decisions,
)

WRITTEN = ("persuade", "asap", "ellipse")
REPORT_N = 10
N_BOOT = 2000


def _load_written(window: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    docs, wins = [], []
    for name in WRITTEN:
        d, w = load_corpus(name, window)
        docs.append(d)
        wins.append(w)
    docs = pd.concat(docs, ignore_index=True)
    wins = pd.concat(wins, ignore_index=True)
    # ELLIPSE shares essays with PERSUADE; drop them from ELLIPSE so that no
    # essay is ever on both sides of a split.
    first = wins[(wins["corpus"] == "persuade") & (wins["window"] == 0)]
    persuade_keys = set(first["text"].map(text_key))
    e = wins[(wins["corpus"] == "ellipse") & (wins["window"] == 0)]
    dup = set(e.loc[e["text"].map(text_key).isin(persuade_keys), "doc_id"])
    wins = wins[~wins["doc_id"].isin(dup)].reset_index(drop=True)
    docs = docs[~docs["doc_id"].isin(dup)].reset_index(drop=True)
    wins["_row"] = np.arange(len(wins))
    return docs, wins, len(dup)


def shared_prompts(docs: pd.DataFrame) -> dict:
    """Prompts that two corpora share, matched by name, and the essays on them.

    ASAP names its prompts by set number and shares none with the others.
    """
    names = {c: set(d["prompt"].dropna().str.strip().str.lower())
             for c, d in docs.groupby("corpus")}
    shared = sorted(names["persuade"] & names["ellipse"])
    on_shared = {c: float(docs.loc[docs["corpus"] == c, "prompt"].str.strip().str.lower()
                          .isin(shared).mean())
                 for c in ("persuade", "ellipse")}
    return {"persuade_prompts": len(names["persuade"]), "ellipse_prompts": len(names["ellipse"]),
            "shared": shared, "share_of_essays_on_shared": on_shared}


def _score(decisions: pd.DataFrame, docs: pd.DataFrame, n: int) -> dict:
    d = decisions.merge(docs[["doc_id", "admissible"]], on="doc_id")
    oc = np.array([outcome(a, adm) for a, adm in zip(d[f"tier_n{n}"], d["admissible"])])
    adm = d["admissible"].to_numpy()
    groups = {g: (oc[adm == g] == "correct").mean() for g in sorted(d["admissible"].unique())}
    return {
        "n_docs": int(len(d)),
        "accuracy": float((oc == "correct").mean()),
        "balanced_accuracy": float(np.mean(list(groups.values()))),
        "under": float((oc == "under").mean()),
        "over": float((oc == "over").mean()),
        "floor_share": float((d[f"tier_n{n}"] == 1).mean()),
        "by_admissible": {g: {"n": int((adm == g).sum()),
                              "accuracy": float(groups[g]),
                              "under": float((oc[adm == g] == "under").mean()),
                              "over": float((oc[adm == g] == "over").mean())}
                          for g in groups},
        "mean_windows_used": float(np.minimum(d["n_windows"], n).mean()),
    }


def _outcomes(decisions: pd.DataFrame, docs: pd.DataFrame, n: int) -> pd.DataFrame:
    """One row per writer: admissible tier set and outcome after n interactions."""
    d = decisions.merge(docs[["doc_id", "admissible"]], on="doc_id")
    d["outcome"] = [outcome(a, adm) for a, adm in zip(d[f"tier_n{n}"], d["admissible"])]
    return d[["doc_id", "admissible", "outcome"]].sort_values("doc_id").reset_index(drop=True)


def _rates(correct, under, over, group, n_groups, idx) -> np.ndarray:
    """Balanced accuracy, accuracy, under- and over-protection on a resample."""
    g = group[idx]
    per_group = (np.bincount(g, weights=correct[idx], minlength=n_groups)
                 / np.maximum(np.bincount(g, minlength=n_groups), 1))
    present = np.bincount(g, minlength=n_groups) > 0
    return np.array([per_group[present].mean(), correct[idx].mean(),
                     under[idx].mean(), over[idx].mean()])


METRICS = ("balanced_accuracy", "accuracy", "under", "over")
#: Filled by leave_one_corpus_out when it bootstraps.
BEST_VS_RUNNER_UP: dict = {}


def _arrays(frame: pd.DataFrame):
    oc = frame["outcome"].to_numpy()
    codes, group = np.unique(frame["admissible"].to_numpy(), return_inverse=True)
    return ((oc == "correct").astype(float), (oc == "under").astype(float),
            (oc == "over").astype(float), group, len(codes))


def bootstrap_ci(frame: pd.DataFrame, rng: np.random.Generator, b: int = N_BOOT) -> dict:
    """95% percentile intervals over writers for each headline rate."""
    c, u, o, g, k = _arrays(frame)
    n = len(frame)
    draws = np.array([_rates(c, u, o, g, k, rng.integers(0, n, n)) for _ in range(b)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {m: [float(lo[i]), float(hi[i])] for i, m in enumerate(METRICS)}


def paired_difference(a: pd.DataFrame, b_: pd.DataFrame, metric: str,
                      rng: np.random.Generator, b: int = N_BOOT) -> dict:
    """Difference a - b in one rate on the same writers, with a paired bootstrap."""
    m = a.merge(b_, on=["doc_id", "admissible"], suffixes=("_a", "_b"))
    i = METRICS.index(metric)
    fa = _arrays(m.rename(columns={"outcome_a": "outcome"}))
    fb = _arrays(m.rename(columns={"outcome_b": "outcome"}))
    n = len(m)
    full = np.arange(n)
    diff = _rates(*fa, full)[i] - _rates(*fb, full)[i]
    draws = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        draws.append(_rates(*fa, idx)[i] - _rates(*fb, idx)[i])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"metric": metric, "n_writers": int(n), "difference": float(diff),
            "ci": [float(lo), float(hi)]}


def leave_one_corpus_out(docs, wins, zoo, horizons=HORIZONS, rng=None) -> dict:
    """Train on two corpora, test on the third. With `rng`, every estimator's
    headline rates carry bootstrap intervals over writers, and the estimator
    with the highest balanced accuracy is compared with the runner-up."""
    out: dict = {}
    for held in WRITTEN:
        train = expand_admissible(wins[wins["corpus"] != held])
        test = wins[wins["corpus"] == held]
        test_docs = docs[docs["corpus"] == held]
        out[held] = {}
        frames = {}
        for name, factory in zoo.items():
            model = factory().fit(train)
            dec = session_decisions(test, model.window_log_scores(test), model.tiers, horizons)
            out[held][name] = {f"n{n}": _score(dec, test_docs, n) for n in horizons}
            if rng is not None:
                frames[name] = _outcomes(dec, test_docs, REPORT_N)
                out[held][name][f"n{REPORT_N}"]["ci"] = bootstrap_ci(frames[name], rng)
        if rng is not None:
            bal = {m: out[held][m][f"n{REPORT_N}"]["balanced_accuracy"] for m in frames}
            ranked = sorted(bal, key=bal.get, reverse=True)
            BEST_VS_RUNNER_UP[held] = {"best": ranked[0], "runner_up": ranked[1],
                                       **paired_difference(frames[ranked[0]], frames[ranked[1]],
                                                           "balanced_accuracy", rng)}
    return out



def topic_confound(docs, wins, zoo, rng) -> dict:
    """Inside PERSUADE: random folds versus folds that hold out whole prompts.

    Prompt-disjoint folds keep the single grade-6 prompt in every training set
    (holding it out would leave no t3 data), so they test t4 and t5 only.
    """
    p = wins[wins["corpus"] == "persuade"]
    pdocs = docs[docs["corpus"] == "persuade"]
    prompt = pdocs.set_index("doc_id")["prompt"]
    rng = np.random.default_rng(0)
    ids = pdocs["doc_id"].to_numpy().copy()
    rng.shuffle(ids)
    random_fold = {d: i % 5 for i, d in enumerate(ids)}
    t3_prompts = set(pdocs.loc[pdocs["admissible"] == "3", "prompt"])
    others = sorted(set(pdocs["prompt"]) - t3_prompts)
    prompt_fold = {pr: i % 5 for i, pr in enumerate(others)}
    out = {}
    for name in ("Gradient boosting (features)", "Logistic (TF-IDF)", "NPL (learned)"):
        res = {}
        frames = {}
        for scheme in ("random", "prompt_disjoint"):
            decs = []
            for k in range(5):
                if scheme == "random":
                    te_ids = {d for d, f in random_fold.items() if f == k}
                    tr = p[~p["doc_id"].isin(te_ids)]
                else:
                    te_ids = {d for d in pdocs["doc_id"]
                              if prompt[d] in prompt_fold and prompt_fold[prompt[d]] == k}
                    tr = p[~p["doc_id"].isin(te_ids)]
                te = p[p["doc_id"].isin(te_ids)]
                if te.empty:
                    continue
                model = zoo[name]().fit(expand_admissible(tr))
                decs.append(session_decisions(te, model.window_log_scores(te), model.tiers,
                                              (REPORT_N,)))
            dec = pd.concat(decs)
            scored = pdocs[pdocs["doc_id"].isin(dec["doc_id"])]
            if scheme == "random":
                scored = scored[scored["admissible"] != "3"]  # same population as prompt-disjoint
                dec = dec[dec["doc_id"].isin(scored["doc_id"])]
            res[scheme] = _score(dec, scored, REPORT_N)
            frames[scheme] = _outcomes(dec, scored, REPORT_N)
            res[scheme]["ci"] = bootstrap_ci(frames[scheme], rng)
        res["random_minus_disjoint"] = paired_difference(
            frames["random"], frames["prompt_disjoint"], "accuracy", rng)
        out[name] = res
    return out


def bound_check(docs, wins) -> dict:
    """Proposition 1 at the learned parameters versus the observed error.

    The bound assumes windows drawn independently from the fitted model. Real
    windows from one essay are neither Gaussian nor independent, so the
    comparison measures how far the guarantee survives misspecification.
    """
    from safenest.estimator import DEFAULT_GAMMA

    out = {}
    for held in WRITTEN:
        model = GaussianTierModel().fit(expand_admissible(wins[wins["corpus"] != held]))
        test = wins[(wins["corpus"] == held) & (wins["tier"] > 0)]
        dec = session_decisions(test, model.window_log_scores(test), model.tiers, (1, 3, 5, 10))
        dec = dec.merge(docs[["doc_id", "tier"]], on="doc_id")
        rows = {}
        for t in sorted(dec["tier"].unique()):
            rows[f"t{t}"] = {}
            for n in (1, 3, 5, 10):
                # Only sessions that actually supply n windows; the bound is for n.
                sub = dec[(dec["tier"] == t) & (dec["n_windows"] >= n)]
                if sub.empty:
                    continue
                observed = float((sub[f"tier_n{n}"] != t).mean())
                bound = _gaussian_bound(model, int(t), n, DEFAULT_GAMMA)
                rows[f"t{t}"][f"n{n}"] = {"observed_error": observed, "bound": bound,
                                          "sessions": int(len(sub)),
                                          "violated": bool(observed > bound + 1e-12)}
        out[held] = rows
    return out


def _gaussian_bound(model: GaussianTierModel, true: int, n: int, gamma: float) -> float:
    """Proposition 1 for diagonal Gaussians, per-rival optimisation of s."""
    from safenest.signals import _gaussian_log_chernoff

    k = len(model.tiers)
    c = (k - 1) * gamma / (1.0 - gamma)
    total = 0.0
    for r in model.tiers:
        if r == true:
            continue
        best = np.inf
        for s in np.linspace(0.005, 0.995, 199):
            g = sum(_gaussian_log_chernoff(model.mean[true][j], model.sd[true][j],
                                           model.mean[r][j], model.sd[r][j], s)
                    for j in range(len(model.mean[true])))
            best = min(best, c ** s * np.exp(n * g))
        total += best
    return float(min(total, 1.0))


def run() -> dict:
    try:
        docs, wins, n_dup = _load_written(50)
    except MissingCorpus as exc:
        print(f"  skipped: {exc}")
        return {"skipped": str(exc)}
    emb = embeddings_for("written", wins)
    zoo = model_zoo(emb)

    summary = (docs.groupby(["corpus", "admissible"]).size().unstack(fill_value=0)
               .to_dict(orient="index"))
    print(f"Documents per corpus and admissible tier set: {summary}")
    print(f"ELLIPSE essays also in PERSUADE, removed: {n_dup}")
    prompts = shared_prompts(docs)
    print(f"Prompts shared by PERSUADE and ELLIPSE: {prompts}")

    rng = rng_for("exp13_bootstrap")
    loco = leave_one_corpus_out(docs, wins, zoo, rng=rng)
    for held, cmp in BEST_VS_RUNNER_UP.items():
        print(f"  {held}: {cmp['best']} minus {cmp['runner_up']} balanced accuracy "
              f"{pct(cmp['difference'])} [{pct(cmp['ci'][0])}, {pct(cmp['ci'][1])}]")
    for held, models in loco.items():
        rows = []
        for name, res in models.items():
            r = res[f"n{REPORT_N}"]
            rows.append({"Estimator": name, "Acc": pct(r["accuracy"]),
                         "Bal. acc": pct(r["balanced_accuracy"]), "Under": pct(r["under"]),
                         "Over": pct(r["over"]), "Floor": pct(r["floor_share"])})
        table(rows, ["Estimator", "Acc", "Bal. acc", "Under", "Over", "Floor"],
              f"Held-out corpus {held} (n = {REPORT_N} windows), %")

    confound = topic_confound(docs, wins, zoo, rng)
    print("\nTopic confound inside PERSUADE (t4/t5 essays, n = 10):")
    for name, res in confound.items():
        print(f"  {name:30s} random folds {pct(res['random']['accuracy'])}%  "
              f"prompt-disjoint {pct(res['prompt_disjoint']['accuracy'])}%")

    sensitivity = {}
    for w in (25, 100):
        d_w, wins_w, _ = _load_written(w)
        small_zoo = {k: v for k, v in model_zoo().items()
                     if k in ("NPL (learned)", "Gradient boosting (features)")}
        sensitivity[f"w{w}"] = leave_one_corpus_out(d_w, wins_w, small_zoo, (REPORT_N,))

    bound = bound_check(docs, wins)

    return {
        "window_words": 50,
        "horizons": list(HORIZONS),
        "documents": summary,
        "ellipse_duplicates_removed": n_dup,
        "shared_prompts": prompts,
        "best_vs_runner_up": BEST_VS_RUNNER_UP,
        "leave_one_corpus_out": loco,
        "topic_confound": confound,
        "window_sensitivity": sensitivity,
        "proposition1_real": bound,
    }


if __name__ == "__main__":
    save("exp13_real_estimation", run())
