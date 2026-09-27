"""Learning the tier estimator from real children's language.

The specification in `signals.py` fixes the tier-conditional likelihoods by
hand. This module learns them from data instead and evaluates the result the
same way the simulation does: evidence is accumulated over a child's successive
interactions (feature windows), the posterior is thresholded at gamma, and an
unconfident session falls back to the most protective tier.

Estimators
----------
GaussianTierModel      the NPL estimator with learned parameters: one diagonal
                       Gaussian per tier over the five linguistic features,
                       fitted by maximum likelihood (child-weighted), or
                       released under (epsilon, delta)-DP by `fit_private`.
SpecifiedTierModel     the hand-set parameters of Table 3, applied unchanged to
                       real data; measures how far the specification is from it.
DiscriminativeTierModel  any scikit-learn classifier on window features or
                       window text; its per-window posteriors are combined
                       across windows by the same product rule.

All models expose `window_log_scores(frame) -> (n_windows, K)` over the tiers in
`self.tiers`; `session_decisions` turns those into tier assignments.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .estimator import DEFAULT_GAMMA
from .features import FEATURE_BOUNDS, from_unit, to_unit
from .privacy import analytic_gaussian_sigma
from .signals import LINGUISTIC_FEATURES, LINGUISTIC_MEAN, LINGUISTIC_STD
from .tiers import ALL_TIERS

FEATURES = list(LINGUISTIC_FEATURES)
#: Variance floor in unit space, so a fitted variance is never zero.
UNIT_VAR_FLOOR = 1e-6
#: Clip on a window's deviation from its tier mean, as a fraction of the public
#: feature range, in the private variance release.
DEVIATION_CLIP = 0.35


def _gauss_loglik(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (x - mean) / sd
    return -0.5 * np.sum(z**2, axis=1) - np.sum(np.log(sd)) - 0.5 * x.shape[1] * np.log(2 * np.pi)


def expand_admissible(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (window, admissible tier), weighted 1/|admissible|.

    A document from a boundary grade (for example grade 10, ages 15-16) is
    evidence about both adjacent tiers; it enters training once per admissible
    tier with half weight rather than being forced into one label or dropped.
    """
    adm = frame["admissible"].str.split(",")
    out = frame.assign(_adm=adm).explode("_adm")
    out["tier"] = out["_adm"].astype(int)
    out["weight"] = 1.0 / out.groupby(level=0)["_adm"].transform("size")
    return out.drop(columns="_adm").reset_index(drop=True)


def _weights(frame: pd.DataFrame) -> np.ndarray:
    return frame["weight"].to_numpy() if "weight" in frame else np.ones(len(frame))


# ------------------------------------------------------------ generative
@dataclass
class GaussianTierModel:
    """Diagonal Gaussian per tier, the estimator of Section 3.3 with learned parameters."""

    tiers: list[int] = field(default_factory=list)
    mean: dict[int, np.ndarray] = field(default_factory=dict)
    sd: dict[int, np.ndarray] = field(default_factory=dict)
    released_sigma: tuple[float, float] | None = None  # DP noise per stage, unit space

    @staticmethod
    def _child_moments(frame: pd.DataFrame) -> pd.DataFrame:
        """Per child: mean of unit-scaled features and of their squares.

        A child contributes one vector however many windows they wrote, which is
        what makes the child, not the window, the privacy unit.
        """
        u = to_unit(frame[FEATURES].to_numpy())
        cols = [f"u_{f}" for f in FEATURES] + [f"u2_{f}" for f in FEATURES]
        df = pd.DataFrame(np.hstack([u, u**2]), columns=cols)
        df["doc_id"] = frame["doc_id"].to_numpy()
        df["tier"] = frame["tier"].to_numpy()
        df["weight"] = _weights(frame)
        grouped = df.groupby(["doc_id", "tier"], sort=False)
        mom = grouped[cols].mean()
        mom["weight"] = grouped["weight"].first()
        return mom.reset_index()

    def _set_from_unit(self, tier: int, m1: np.ndarray, m2: np.ndarray) -> None:
        m1 = np.clip(m1, 0.0, 1.0)
        var = np.maximum(m2 - m1**2, UNIT_VAR_FLOOR)
        lo = np.array([FEATURE_BOUNDS[f][0] for f in FEATURES])
        hi = np.array([FEATURE_BOUNDS[f][1] for f in FEATURES])
        self.mean[tier] = from_unit(m1)
        self.sd[tier] = np.sqrt(var) * (hi - lo)

    def fit(self, frame: pd.DataFrame) -> GaussianTierModel:
        """Maximum-likelihood fit, each child weighted equally."""
        mom = self._child_moments(frame)
        self.tiers = sorted(int(t) for t in mom["tier"].unique())
        for t in self.tiers:
            sub = mom[mom["tier"] == t]
            w = sub["weight"].to_numpy()[:, None]
            self._set_from_unit(
                t,
                (w * sub[[f"u_{f}" for f in FEATURES]].to_numpy()).sum(0) / w.sum(),
                (w * sub[[f"u2_{f}" for f in FEATURES]].to_numpy()).sum(0) / w.sum())
        self.released_sigma = None
        return self

    def fit_private(self, frame: pd.DataFrame, epsilon: float | None, delta: float,
                    rng: np.random.Generator,
                    deviation_clip: float = DEVIATION_CLIP) -> GaussianTierModel:
        """(epsilon, delta)-DP release of every tier's mean and variance.

        Privacy unit: one child in the calibration corpus. Adjacency: replace
        one child's text by another's; each child's admissible tiers, and hence
        the per-tier weight totals, are public (they come from recorded age or
        grade). Features are clipped to the public `FEATURE_BOUNDS` and mapped
        to [0, 1]^d. The release has two stages, each (epsilon/2, delta/2)-DP,
        so the whole is (epsilon, delta)-DP by sequential composition:

          1. Means. A child contributes w_k * ubar to tier k, where ubar in
             [0, 1]^d is the mean of their windows and sum_k w_k = 1, so a
             replacement moves the stacked per-tier sums by at most sqrt(d).
          2. Variances, around the means released in stage 1 (post-processing
             of an already private output). A child contributes w_k * v, where
             v is the mean over their windows of min((u - mean_k)^2, b^2), so
             a replacement moves the sums by at most b^2 sqrt(d).

        Clipping squared deviations at b (default 0.35 of the feature range)
        is what makes the variance stage affordable: its sensitivity is b^2
        rather than 1. It biases variances down when deviations exceed b.
        `epsilon=None` runs the same clipped pipeline without noise, which
        separates the cost of clipping from the cost of privacy.
        """
        mom = self._child_moments(frame)
        d = len(FEATURES)
        b2 = deviation_clip ** 2
        if epsilon is None:
            sigma1 = sigma2 = 0.0
        else:
            sigma1 = analytic_gaussian_sigma(epsilon / 2, delta / 2, float(np.sqrt(d)))
            sigma2 = analytic_gaussian_sigma(epsilon / 2, delta / 2, float(b2 * np.sqrt(d)))
        self.tiers = sorted(int(t) for t in mom["tier"].unique())
        ucols = [f"u_{f}" for f in FEATURES]
        released_means = {}
        for t in self.tiers:
            sub = mom[mom["tier"] == t]
            w = sub["weight"].to_numpy()[:, None]
            s1 = (w * sub[ucols].to_numpy()).sum(0) + rng.normal(0, sigma1, d) * (sigma1 > 0)
            released_means[t] = np.clip(s1 / w.sum(), 0.0, 1.0)
        # Stage 2 needs each window's deviation from its tier's released mean.
        u = to_unit(frame[FEATURES].to_numpy())
        tiers = frame["tier"].to_numpy().astype(int)
        centre = np.stack([released_means[int(t)] for t in tiers])
        dev = pd.DataFrame(np.minimum((u - centre) ** 2, b2), columns=FEATURES)
        dev["doc_id"] = frame["doc_id"].to_numpy()
        dev["tier"] = tiers
        dev["weight"] = _weights(frame)
        g = dev.groupby(["doc_id", "tier"], sort=False)
        per_child = g[FEATURES].mean()
        per_child["weight"] = g["weight"].first()
        per_child = per_child.reset_index()
        lo = np.array([FEATURE_BOUNDS[f][0] for f in FEATURES])
        hi = np.array([FEATURE_BOUNDS[f][1] for f in FEATURES])
        for t in self.tiers:
            sub = per_child[per_child["tier"] == t]
            w = sub["weight"].to_numpy()[:, None]
            total = float(w.sum())
            s2 = (w * sub[FEATURES].to_numpy()).sum(0) + rng.normal(0, sigma2, d) * (sigma2 > 0)
            # A released variance below three noise standard deviations is not
            # distinguishable from zero; the floor uses public quantities only,
            # so it is post-processing.
            var = np.maximum(s2 / total, max(UNIT_VAR_FLOOR, 3.0 * sigma2 / total))
            self.mean[t] = from_unit(released_means[t])
            self.sd[t] = np.sqrt(var) * (hi - lo)
        self.released_sigma = (sigma1, sigma2) if epsilon is not None else None
        return self

    def window_log_scores(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[FEATURES].to_numpy()
        return np.column_stack([_gauss_loglik(x, self.mean[t], self.sd[t]) for t in self.tiers])

    def parameters(self) -> dict:
        return {f"t{t}": {"mean": self.mean[t].tolist(), "sd": self.sd[t].tolist()}
                for t in self.tiers}


@dataclass
class SpecifiedTierModel:
    """The author-specified likelihoods of Table 3, restricted to `tiers`."""

    tiers: list[int] = field(default_factory=lambda: [int(t) for t in ALL_TIERS])

    def fit(self, frame: pd.DataFrame) -> SpecifiedTierModel:
        present = sorted(int(t) for t in frame["tier"].unique())
        self.tiers = present
        return self

    def window_log_scores(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[FEATURES].to_numpy()
        by_int = {int(t): t for t in ALL_TIERS}
        return np.column_stack([
            _gauss_loglik(x, LINGUISTIC_MEAN[by_int[t]], LINGUISTIC_STD[by_int[t]])
            for t in self.tiers
        ])


# --------------------------------------------------------- discriminative
@dataclass
class DiscriminativeTierModel:
    """A probabilistic classifier on windows, combined across windows by the
    product rule: log p(t | x_1..n) = sum_i log p(t | x_i) - (n-1) log p(t).

    `kind` is "features" (the five features, standardised) or "text" (the
    window's words), and `make` builds the scikit-learn estimator.
    """

    make: object
    kind: str = "features"
    tiers: list[int] = field(default_factory=list)
    _model: object = None
    _log_prior: np.ndarray | None = None
    _embed: object = None

    def _inputs(self, frame: pd.DataFrame):
        if self.kind == "features":
            return frame[FEATURES].to_numpy()
        if self.kind == "text":
            return frame["text"].tolist()
        if self.kind == "embedding":
            return self._embed(frame["text"].tolist())
        raise ValueError(self.kind)

    def fit(self, frame: pd.DataFrame) -> DiscriminativeTierModel:
        y = frame["tier"].to_numpy().astype(int)
        # Weight so that every child counts once and every tier counts equally,
        # as in the generative fit; the prior the product rule divides out is
        # then uniform over tiers.
        base = _weights(frame)
        per_child = frame.groupby(["doc_id", "tier"])["doc_id"].transform("size").to_numpy()
        child_w = base / per_child
        tier_total = pd.Series(child_w).groupby(y).transform("sum").to_numpy()
        weight = child_w / tier_total
        self._model = self.make()
        weight = weight * len(y) / weight.sum()
        self._model.fit(self._inputs(frame), y, **{self._weight_arg(): weight})
        self.tiers = [int(t) for t in self._model.classes_]
        self._log_prior = np.log(np.full(len(self.tiers), 1.0 / len(self.tiers)))
        return self

    def _weight_arg(self) -> str:
        model = self.make()
        if hasattr(model, "steps"):
            return f"{model.steps[-1][0]}__sample_weight"
        return "sample_weight"

    def window_log_scores(self, frame: pd.DataFrame) -> np.ndarray:
        p = self._model.predict_proba(self._inputs(frame))
        return np.log(np.clip(p, 1e-12, 1.0)) - self._log_prior


# ---------------------------------------------------------------- sessions
def session_decisions(frame: pd.DataFrame, scores: np.ndarray, tiers: list[int],
                      horizons: tuple[int, ...], gamma: float = DEFAULT_GAMMA,
                      floor_tier: int = 1) -> pd.DataFrame:
    """Tier assigned to each document after n = 1, 3, ... windows.

    Windows are read in order; a document with fewer than n windows is scored
    on all it has, and `n_used` records that. Scores are summed (independent
    windows under the model), normalised with a uniform prior, and the MAP
    tier is assigned when its posterior reaches `gamma`, else `floor_tier`.
    """
    frame = frame.reset_index(drop=True)
    order = frame.sort_values(["doc_id", "window"]).index.to_numpy()
    f = frame.loc[order]
    s = scores[order]
    out = []
    starts = np.flatnonzero(np.r_[True, f["doc_id"].to_numpy()[1:] != f["doc_id"].to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(f)]
    for a, b in zip(starts, ends):
        row = {"doc_id": f["doc_id"].iloc[a], "n_windows": b - a}
        cum = np.cumsum(s[a:b], axis=0)
        for n in horizons:
            k = min(n, b - a)
            logp = cum[k - 1] - cum[k - 1].max()
            post = np.exp(logp) / np.exp(logp).sum()
            j = int(np.argmax(post))
            row[f"tier_n{n}"] = tiers[j] if post[j] >= gamma else floor_tier
            row[f"conf_n{n}"] = float(post[j])
            row[f"map_n{n}"] = tiers[j]
        out.append(row)
    return pd.DataFrame(out)


def outcome(assigned: int, admissible: str) -> str:
    """'correct', 'under' (less protective than warranted) or 'over'."""
    adm = [int(t) for t in admissible.split(",")]
    if assigned in adm:
        return "correct"
    return "under" if assigned > max(adm) else "over"


def summarise(decisions: pd.DataFrame, docs: pd.DataFrame, n: int,
              by: str | None = None) -> pd.DataFrame:
    """Accuracy, under- and over-protection at horizon n, optionally by group."""
    cols = ["doc_id", "admissible"] + ([by] if by else [])
    d = decisions.merge(docs[list(dict.fromkeys(cols))], on="doc_id")
    d["outcome"] = [outcome(a, adm) for a, adm in zip(d[f"tier_n{n}"], d["admissible"])]
    keys = [by] if by else []
    g = d.groupby(keys) if keys else [((), d)]
    rows = []
    for key, sub in g:
        m = len(sub)
        rows.append({
            **({by: key[0] if isinstance(key, tuple) else key} if by else {}),
            "n": m,
            "accuracy": float((sub["outcome"] == "correct").mean()),
            "under": float((sub["outcome"] == "under").mean()),
            "over": float((sub["outcome"] == "over").mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
