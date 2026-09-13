"""Tier-conditional signal models and synthetic user generation.

Four modalities, matching the manuscript:

  linguistic   5-dim multivariate Gaussian  (MTLD, Flesch-Kincaid grade, mean
               sentence length, parse-tree depth, spelling error rate)
  behavioural  log-normal typing speed in WPM
  device       Bernoulli child-account / parental-controls flag
  contextual   Bernoulli presence of an external attestation

Parameters are author-specified simulation assumptions. The choice of features
is motivated by developmental-linguistics work (CHILDES-db; the Oxford
Children's Corpus; MTLD), but no source-to-parameter calibration is claimed and
no child-produced text is used. They are declared here in one place so the
paper's parameter table can be regenerated from code rather than transcribed.

`UserProfile` lets an experiment perturb a synthetic cohort away from the
corpus-derived norms, which is what Experiment 06 (atypical and
non-Western populations) needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .tiers import ALL_TIERS, K_TIERS, Tier

LINGUISTIC_FEATURES: tuple[str, ...] = (
    "mtld",
    "fk_grade",
    "mean_sentence_length",
    "parse_depth",
    "spelling_error_rate",
)
MODALITIES: tuple[str, ...] = ("linguistic", "behavioural", "device", "contextual")

# Linguistic means per tier, in feature order above.
LINGUISTIC_MEAN: dict[Tier, np.ndarray] = {
    Tier.T1: np.array([12.0, 0.8, 4.0, 2.2, 0.35]),
    Tier.T2: np.array([22.0, 2.5, 6.5, 3.0, 0.18]),
    Tier.T3: np.array([34.0, 5.0, 9.0, 3.8, 0.09]),
    Tier.T4: np.array([48.0, 7.5, 12.0, 4.5, 0.05]),
    Tier.T5: np.array([60.0, 9.5, 14.5, 5.1, 0.03]),
}
# Per-tier standard deviations (diagonal covariance; correlations are added by
# `correlation` below so the conditional-independence assumption can be relaxed).
LINGUISTIC_STD: dict[Tier, np.ndarray] = {
    Tier.T1: np.array([4.0, 0.6, 1.2, 0.5, 0.10]),
    Tier.T2: np.array([5.5, 0.9, 1.6, 0.6, 0.07]),
    Tier.T3: np.array([7.0, 1.2, 2.0, 0.7, 0.04]),
    Tier.T4: np.array([9.0, 1.5, 2.6, 0.8, 0.03]),
    Tier.T5: np.array([11.0, 1.8, 3.0, 0.9, 0.02]),
}
# Typing speed, WPM: log-normal with these medians and log-scale sigma.
TYPING_MEDIAN_WPM: dict[Tier, float] = {
    Tier.T1: 5.0, Tier.T2: 10.0, Tier.T3: 18.0, Tier.T4: 28.0, Tier.T5: 38.0,
}
TYPING_LOG_SIGMA: dict[Tier, float] = {
    Tier.T1: 0.45, Tier.T2: 0.40, Tier.T3: 0.35, Tier.T4: 0.32, Tier.T5: 0.30,
}
# P(child-account / parental-controls flag present).
DEVICE_P: dict[Tier, float] = {
    Tier.T1: 0.90, Tier.T2: 0.80, Tier.T3: 0.60, Tier.T4: 0.35, Tier.T5: 0.15,
}
# P(external attestation present); when absent the likelihood ratio is neutral.
CONTEXT_P: dict[Tier, float] = {
    Tier.T1: 0.70, Tier.T2: 0.60, Tier.T3: 0.45, Tier.T4: 0.30, Tier.T5: 0.20,
}


@dataclass(frozen=True)
class UserProfile:
    """A departure from the corpus-derived norms for a synthetic cohort.

    linguistic_tier_shift
        Shifts the *linguistic* modality to look like a user this many tiers
        older (positive) or younger (negative), while the protective tier the
        user actually needs stays the nominal one. A verbally precocious or
        highly verbal autistic 8-year-old is `+2`.
    linguistic_scale
        Multiplies linguistic standard deviations (heterogeneity / dialect
        variation / L2 speakers).
    typing_scale
        Multiplies median typing speed (motor differences, input device).
    device_p_override
        Replaces the device-flag probability (e.g. shared family device, or a
        deployment context where child accounts are not used).
    """

    name: str = "typical"
    linguistic_tier_shift: float = 0.0
    linguistic_scale: float = 1.0
    typing_scale: float = 1.0
    device_p_override: float | None = None
    context_p_override: float | None = None


TYPICAL = UserProfile("typical")
NEURODIVERGENT_VERBAL = UserProfile(
    "neurodivergent_verbal", linguistic_tier_shift=+2.0, linguistic_scale=1.4
)
NEURODIVERGENT_MOTOR = UserProfile(
    "neurodivergent_motor", linguistic_tier_shift=+1.0, typing_scale=0.5, linguistic_scale=1.3
)
NON_WEIRD_L2 = UserProfile(
    "non_weird_l2", linguistic_tier_shift=-1.0, linguistic_scale=1.6, device_p_override=0.25
)
DIALECT_SWITCHING = UserProfile("dialect_switching", linguistic_scale=1.8)

PROFILES: dict[str, UserProfile] = {
    p.name: p
    for p in (TYPICAL, NEURODIVERGENT_VERBAL, NEURODIVERGENT_MOTOR, NON_WEIRD_L2, DIALECT_SWITCHING)
}


def _interp_tier_params(tier: Tier, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Linguistic (mean, std) for a tier displaced by a fractional `shift`.

    Linear interpolation between neighbouring tier parameter sets, clamped at
    the ends, so a shift of +2 from t2 gives exactly the t4 profile.
    """
    target = float(np.clip(int(tier) + shift, 1, K_TIERS))
    lo = Tier(int(np.floor(target)))
    hi = Tier(int(np.ceil(target)))
    w = target - int(lo)
    mean = (1 - w) * LINGUISTIC_MEAN[lo] + w * LINGUISTIC_MEAN[hi]
    std = (1 - w) * LINGUISTIC_STD[lo] + w * LINGUISTIC_STD[hi]
    return mean, std


@dataclass(frozen=True)
class SignalModel:
    """Tier-conditional distributions the estimator uses as its likelihoods.

    `correlation` is the equicorrelation coefficient among linguistic features.
    The estimator's likelihood always assumes independence; setting
    `correlation > 0` on the *generating* model is how Experiment 07 measures
    the cost of that assumption.
    """

    profile: UserProfile = TYPICAL
    correlation: float = 0.0
    param_noise: np.ndarray | None = None  # CORPUS-DP perturbation of means
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    # -- parameters --------------------------------------------------------
    def linguistic_params(self, tier: Tier) -> tuple[np.ndarray, np.ndarray]:
        key = ("ling", int(tier))
        if key not in self._cache:
            mean, std = _interp_tier_params(tier, self.profile.linguistic_tier_shift)
            std = std * self.profile.linguistic_scale
            if self.param_noise is not None:
                mean = mean + self.param_noise[int(tier) - 1]
            self._cache[key] = (mean, std)
        return self._cache[key]

    def linguistic_cov(self, tier: Tier) -> np.ndarray:
        _, std = self.linguistic_params(tier)
        d = len(std)
        corr = np.full((d, d), self.correlation)
        np.fill_diagonal(corr, 1.0)
        return corr * np.outer(std, std)

    def typing_params(self, tier: Tier) -> tuple[float, float]:
        median = TYPING_MEDIAN_WPM[tier] * self.profile.typing_scale
        return float(np.log(median)), TYPING_LOG_SIGMA[tier]

    def device_p(self, tier: Tier) -> float:
        if self.profile.device_p_override is not None:
            return self.profile.device_p_override
        return DEVICE_P[tier]

    def context_p(self, tier: Tier) -> float:
        if self.profile.context_p_override is not None:
            return self.profile.context_p_override
        return CONTEXT_P[tier]

    # -- sampling ----------------------------------------------------------
    def sample(self, tier: Tier, rng: np.random.Generator) -> dict[str, np.ndarray | float]:
        """Draw one interaction's worth of signals for a user at `tier`."""
        mean, _ = self.linguistic_params(tier)
        ling = rng.multivariate_normal(mean, self.linguistic_cov(tier))
        mu, sigma = self.typing_params(tier)
        wpm = float(rng.lognormal(mu, sigma))
        device = float(rng.random() < self.device_p(tier))
        context = float(rng.random() < self.context_p(tier))
        return {"linguistic": ling, "behavioural": wpm, "device": device, "contextual": context}

    # -- log-likelihoods used by the estimator -----------------------------
    def log_likelihood(self, modality: str, value, tier: Tier) -> float:
        if modality == "linguistic":
            mean, std = self.linguistic_params(tier)
            # Estimator-side likelihood is diagonal by construction.
            z = (np.asarray(value) - mean) / std
            return float(-0.5 * np.sum(z**2) - np.sum(np.log(std)) - 0.5 * len(std) * np.log(2 * np.pi))
        if modality == "behavioural":
            mu, sigma = self.typing_params(tier)
            x = max(float(value), 1e-6)
            return float(
                -np.log(x * sigma * np.sqrt(2 * np.pi)) - (np.log(x) - mu) ** 2 / (2 * sigma**2)
            )
        if modality == "device":
            p = self.device_p(tier)
            return float(np.log(p if value >= 0.5 else 1.0 - p))
        if modality == "contextual":
            p = self.context_p(tier)
            # Absent attestation is deliberately near-neutral: a missing external
            # signal must not by itself drive the tier estimate.
            return float(np.log(p)) if value >= 0.5 else float(np.log(1.0 - p) * 0.25)
        raise KeyError(f"unknown modality {modality!r}")

    def llr_vector(self, modality: str, value) -> np.ndarray:
        """Log-likelihood vector across all tiers, centred at its own mean.

        Centring fixes the reference point used in the sensitivity analysis in
        `privacy.PrivacyConfig.llr_sensitivity`.
        """
        ll = np.array([self.log_likelihood(modality, value, t) for t in ALL_TIERS])
        return ll - ll.mean()


def release_corpus_parameters(
    model: SignalModel, config, rng: np.random.Generator, private: bool = True
) -> SignalModel:
    """Simulate the CORPUS-mode release end to end and return the released model.

    For each tier, `config.corpus_n_per_tier` synthetic corpus children are drawn
    from `model`. Each child's linguistic vector is standardised by the tier's
    public design scale, centred on the public prior mean, clipped to
    [-c, c] per feature, and averaged. Gaussian noise with the analytically
    calibrated sigma (`config.corpus_gaussian_sigma()`) is added to the mean, and
    the result is mapped back to feature units. Both the sampling error of a
    finite corpus and the privacy noise therefore reach the estimator. The
    noise is applied once, offline, to released parameters, never to a live
    user's signals. `private=False` omits the noise, isolating sampling error.
    """
    c = config.corpus_clip_sd
    n = config.corpus_n_per_tier
    sigma = config.corpus_gaussian_sigma() if private else 0.0
    d = len(LINGUISTIC_FEATURES)
    shift = np.zeros((K_TIERS, d))
    for i, tier in enumerate(ALL_TIERS):
        mean, std = model.linguistic_params(tier)
        draws = rng.multivariate_normal(mean, model.linguistic_cov(tier), size=n)
        z = np.clip((draws - mean) / std, -c, c)
        released_z = z.mean(axis=0) + rng.normal(0.0, sigma, size=d)
        shift[i] = released_z * std
    base = model.param_noise if model.param_noise is not None else 0.0
    return replace(model, param_noise=base + shift, _cache={})


# -- analytic divergences -------------------------------------------------
def kl_gaussian(m0: np.ndarray, s0: np.ndarray, m1: np.ndarray, s1: np.ndarray) -> float:
    """KL(N(m0, diag s0^2) || N(m1, diag s1^2)) in nats."""
    v0, v1 = s0**2, s1**2
    return float(0.5 * np.sum(np.log(v1 / v0) + (v0 + (m0 - m1) ** 2) / v1 - 1.0))


def linguistic_kl_matrix(model: SignalModel | None = None) -> np.ndarray:
    """Pairwise KL divergence over the linguistic feature space."""
    model = model or SignalModel()
    out = np.zeros((K_TIERS, K_TIERS))
    for i, ti in enumerate(ALL_TIERS):
        mi, si = model.linguistic_params(ti)
        for j, tj in enumerate(ALL_TIERS):
            if i == j:
                continue
            mj, sj = model.linguistic_params(tj)
            out[i, j] = kl_gaussian(mi, si, mj, sj)
    return out


def min_adjacent_kl(model: SignalModel | None = None) -> float:
    """D_min: smallest divergence between *adjacent* tiers, which governs the
    the KL heuristic; see `chernoff_exponent` for the exponent that governs the proved bound."""
    m = linguistic_kl_matrix(model)
    return float(min(min(m[i, i + 1], m[i + 1, i]) for i in range(K_TIERS - 1)))


def sanov_interactions(d_min: float, delta: float, k: int = K_TIERS) -> float:
    """n >= (1/D_min) ln((K-1)/delta), a KL-based heuristic; Proposition 1 gives the proved Chernoff bound."""
    return float(np.log((k - 1) / delta) / d_min)


# -- Chernoff error exponents ---------------------------------------------
def _gaussian_log_chernoff(a: float, sa: float, b: float, sb: float, s: float) -> float:
    """log E_{x~N(a,sa^2)} [ (N(x;b,sb^2) / N(x;a,sa^2))^s ], in closed form."""
    A = (1.0 - s) / sa**2 + s / sb**2
    B = (1.0 - s) * a / sa**2 + s * b / sb**2
    return float(
        -0.5 * ((1.0 - s) * a**2 / sa**2 + s * b**2 / sb**2 - B**2 / A)
        + 0.5 * np.log(2.0 * np.pi / A)
        - (1.0 - s) * np.log(sa * np.sqrt(2.0 * np.pi))
        - s * np.log(sb * np.sqrt(2.0 * np.pi))
    )


def log_mgf_llr(model: SignalModel, true_tier: Tier, rival: Tier, s: float) -> float:
    """g(s) = log E_{x ~ true_tier} [ exp(s * l(x)) ] for one interaction, where
    l(x) is the log-likelihood ratio of `rival` over `true_tier` exactly as the
    estimator computes it: diagonal linguistic Gaussian, log-normal typing,
    Bernoulli device flag, and the attenuated (weight 0.25) absent-attestation
    term. Modalities are independent under the model, so their terms add.
    """
    ma, sa = model.linguistic_params(true_tier)
    mb, sb = model.linguistic_params(rival)
    g = sum(_gaussian_log_chernoff(ma[k], sa[k], mb[k], sb[k], s) for k in range(len(ma)))
    mu_a, sig_a = model.typing_params(true_tier)
    mu_b, sig_b = model.typing_params(rival)
    g += _gaussian_log_chernoff(mu_a, sig_a, mu_b, sig_b, s)
    p, q = model.device_p(true_tier), model.device_p(rival)
    g += float(np.log(p ** (1 - s) * q ** s + (1 - p) ** (1 - s) * (1 - q) ** s))
    p, q = model.context_p(true_tier), model.context_p(rival)
    g += float(np.log(p * (q / p) ** s + (1 - p) * ((1 - q) / (1 - p)) ** (0.25 * s)))
    return g


_S_GRID = np.linspace(0.005, 0.995, 199)


def chernoff_exponent(model: SignalModel, true_tier: Tier, rival: Tier) -> float:
    """C = -min_{s in (0,1)} g(s): per-interaction error exponent for one rival."""
    return float(-min(log_mgf_llr(model, true_tier, rival, s) for s in _S_GRID))


def misassignment_bound(
    model: SignalModel, true_tier: Tier, n: int, gamma: float, k: int = K_TIERS
) -> float:
    """Upper bound on P(assigned tier != true tier) after n interactions.

    Proposition 1 of the manuscript. With a uniform prior, the confidence-
    thresholded MAP rule assigns the true tier whenever
    sum_{j != *} exp(Lambda_j) <= (1 - gamma)/gamma, where Lambda_j is the summed
    log-likelihood ratio of rival j. A union bound and Markov's inequality on
    exp(s Lambda_j) give, for every s in (0, 1),
        P(error) <= sum_j ((K-1) gamma / (1-gamma))^s exp(n g_j(s)).
    The bound is minimised over s on a grid and capped at 1.
    """
    c = (k - 1) * gamma / (1.0 - gamma)
    rivals = [t for t in ALL_TIERS if t is not true_tier]
    best = min(
        sum(c ** s * np.exp(n * log_mgf_llr(model, true_tier, r, s)) for r in rivals)
        for s in _S_GRID
    )
    return float(min(best, 1.0))
