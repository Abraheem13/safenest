"""Differential privacy for the age-assurance module.

A differential-privacy claim is only meaningful once the privacy unit, the
adjacency relation, the sensitivity bound and the protected output are all
fixed. This module fixes them in code, so the guarantee is auditable rather
than asserted.

Every DP claim needs four things pinned down. Each has a name here:

  privacy unit      what one protected "individual" is
  adjacency         which pairs of inputs the guarantee ranges over
  sensitivity       how much one unit can move the released quantity
  protected output  the exact quantity the noise is added to

--------------------------------------------------------------------------
The tension this module makes explicit
--------------------------------------------------------------------------
The age-assurance module's *purpose* is to infer a property of the live user
from that user's own signals. If the privacy unit is the live user's own data,
then eps-DP directly bounds how much the released tier may depend on that data,
and the achievable accuracy is capped near chance. Concretely, for a clipped
per-modality log-likelihood-ratio vector the per-release signal-to-noise ratio
is  eps_m / (2 (K-1))  and is *independent of the clip bound C* -- tightening
the clip shrinks noise and signal by the same factor. At eps = 1.0 and K = 5 that
ratio is at most 0.125 (whole budget on one modality) and 0.05 for the linguistic
share eps_m = 0.40, so tens of interactions cannot recover a reliable five-way
decision. `LOCAL_INFERENCE` implements this honestly and Experiment 05
reports the resulting curve. The released vector is the K-1 log-likelihood
ratios against the reference tier t1, each clipped to [-C, C]; that is what
makes the L1 sensitivity 2C(K-1).

The defensible reading of the architecture is therefore `CORPUS` (plus
`NONE`'s data minimisation, which is architectural rather than statistical):

  CORPUS  The (eps, delta)-DP guarantee protects the children in a
          *calibration corpus* whose data would parameterise the tier-conditional
          linguistic likelihoods. The privacy unit is one corpus child, who
          contributes one averaged feature vector. Tier sizes n_k are public, so
          adjacency is replace-one (bounded DP). Each child's vector is
          standardised by the public design scale of its tier, centred on a
          public prior mean and clipped to [-c, c] per feature; the protected
          output is the vector of per-tier clipped means. Replacing one child can
          change two tier means (if the replacement sits in another tier), so the
          L2 sensitivity is sqrt(2) * 2c * sqrt(d) / n_k. Noise is calibrated by
          the analytic Gaussian mechanism (Balle and Wang, 2018), which is exact
          for every eps > 0, unlike the classical bound that needs eps < 1.
          Dispersions are public design constants and are not released.
          Live inference then runs on the released parameters.

  NONE    No statistical noise; the guarantee is architectural only. Raw
          features never leave the per-modality enclave, only clipped LLRs
          cross the boundary, and they are discarded at session end. This is
          data minimisation (COPPA) and not differential privacy; it must not
          be reported as an eps guarantee.

Use `PrivacyConfig.describe()` to emit the exact statement for the methods
section, so the paper's wording and the code cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import math

import numpy as np


class PrivacyMode(str, Enum):
    NONE = "none"                        # data minimisation only, no eps claim
    LOCAL_INFERENCE = "local_inference"  # local DP on the live user's own signals
    CORPUS = "corpus"                    # central DP over calibration-corpus children


class PrivacyUnit(str, Enum):
    INTERACTION = "interaction"
    SESSION = "session"
    CORPUS_CHILD = "corpus_child"


class Accounting(str, Enum):
    PER_INTERACTION = "per_interaction"
    PER_SESSION = "per_session"
    ONE_SHOT = "one_shot"  # corpus release happens once, offline


# Local-DP budget split across modalities. Shares must sum to 1.0.
DEFAULT_BUDGET: dict[str, float] = {
    "linguistic": 0.40,
    "behavioural": 0.30,
    "device": 0.20,
    "contextual": 0.10,
}

#: Clipping bound C, in nats, on a per-modality log-likelihood ratio. Bounds
#: sensitivity and caps the influence of any single (possibly spoofed) modality.
DEFAULT_CLIP: float = 3.0


@dataclass
class PrivacyConfig:
    """Privacy parameters plus the accounting needed to state them precisely."""

    mode: PrivacyMode = PrivacyMode.CORPUS
    epsilon_total: float = 1.0
    budget: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BUDGET))
    clip: float = DEFAULT_CLIP
    n_tiers: int = 5
    #: CORPUS mode: delta, public per-tier corpus size, per-feature clip bound
    #: in units of the tier's public design standard deviation, and the number
    #: of released linguistic features per tier.
    corpus_delta: float = 1e-5
    corpus_n_per_tier: int = 400
    corpus_clip_sd: float = 3.0
    corpus_n_features: int = 5

    def __post_init__(self) -> None:
        total = sum(self.budget.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"budget shares must sum to 1.0, got {total:.4f}")
        if self.epsilon_total <= 0:
            raise ValueError("epsilon_total must be positive")

    # -- unit / accounting -------------------------------------------------
    @property
    def unit(self) -> PrivacyUnit:
        return {
            PrivacyMode.NONE: PrivacyUnit.SESSION,
            PrivacyMode.LOCAL_INFERENCE: PrivacyUnit.INTERACTION,
            PrivacyMode.CORPUS: PrivacyUnit.CORPUS_CHILD,
        }[self.mode]

    @property
    def accounting(self) -> Accounting:
        return {
            PrivacyMode.NONE: Accounting.PER_SESSION,
            PrivacyMode.LOCAL_INFERENCE: Accounting.PER_INTERACTION,
            PrivacyMode.CORPUS: Accounting.ONE_SHOT,
        }[self.mode]

    # -- inference-time (LOCAL_INFERENCE) ---------------------------------
    def llr_sensitivity(self) -> float:
        """L1 sensitivity of one clipped LLR vector.

        The released vector holds the K-1 ratios log P(s|t_k) / P(s|t_1),
        k = 2..K, each clipped to [-C, C]. One privacy unit can move every
        coordinate by at most 2C, so the L1 sensitivity is 2C(K-1).
        """
        return 2.0 * self.clip * (self.n_tiers - 1)

    def noise_scale(self, modality: str) -> float:
        """Laplace scale b = Delta / eps_m for `modality`."""
        eps_m = self.epsilon_total * self.budget[modality]
        if eps_m <= 0:
            raise ValueError(f"modality {modality!r} has zero privacy budget")
        return self.llr_sensitivity() / eps_m

    def per_release_snr(self, modality: str) -> float:
        """Signal-to-noise ratio of one LLR release: eps_m / (2 (K-1)).

        Independent of the clip bound C -- the reason LOCAL_INFERENCE cannot be
        made accurate by re-tuning the clip.
        """
        eps_m = self.epsilon_total * self.budget[modality]
        return eps_m / (2.0 * (self.n_tiers - 1))

    def basic_composition(self, n_interactions: int) -> float:
        """Total eps over n interactions under sequential composition."""
        if self.accounting is not Accounting.PER_INTERACTION:
            return self.epsilon_total
        return self.epsilon_total * n_interactions

    def advanced_composition(self, n_interactions: int, delta: float = 1e-5) -> float:
        """(eps', delta) total under advanced composition (Dwork et al., 2010)."""
        if self.accounting is not Accounting.PER_INTERACTION:
            return self.epsilon_total
        e = self.epsilon_total
        return float(
            np.sqrt(2 * n_interactions * np.log(1.0 / delta)) * e
            + n_interactions * e * (np.exp(e) - 1.0)
        )

    # -- corpus-time (CORPUS) ---------------------------------------------
    def corpus_l2_sensitivity(self) -> float:
        """L2 sensitivity of the vector of per-tier clipped means.

        Replace-one adjacency with public tier sizes. A replacement can remove a
        child from one tier and add one to another, changing two tier means;
        within one tier each of the d standardised coordinates moves by at most
        2c / n_k. Hence sqrt(2) * 2c * sqrt(d) / n_k, in standard-deviation units.
        """
        return float(
            math.sqrt(2.0) * 2.0 * self.corpus_clip_sd * math.sqrt(self.corpus_n_features)
            / self.corpus_n_per_tier
        )

    def corpus_gaussian_sigma(self) -> float:
        """Noise standard deviation, in standard-deviation units, for the corpus
        release, calibrated by the analytic Gaussian mechanism."""
        return analytic_gaussian_sigma(
            self.epsilon_total, self.corpus_delta, self.corpus_l2_sensitivity()
        )

    def corpus_classical_sigma(self) -> float:
        """Classical calibration Delta sqrt(2 ln(1.25/delta)) / eps (Dwork and
        Roth, 2014, Theorem 3.22). Valid only for eps < 1; reported for contrast."""
        return float(
            self.corpus_l2_sensitivity()
            * math.sqrt(2.0 * math.log(1.25 / self.corpus_delta))
            / self.epsilon_total
        )

    # -- reporting ---------------------------------------------------------
    def describe(self) -> dict[str, object]:
        """Machine-readable privacy statement for the paper's methods section."""
        common: dict[str, object] = {
            "mode": self.mode.value,
            "privacy_unit": self.unit.value,
            "accounting": self.accounting.value,
            "post_processing": "posterior, tier assignment and response are post-processing",
        }
        if self.mode is PrivacyMode.NONE:
            common.update(
                epsilon_total=None,
                adjacency="not applicable",
                protected_output="none (architectural data minimisation only)",
                mechanism="none",
                claim="raw features never cross the enclave boundary; LLRs discarded "
                      "at session end. This is data minimisation, NOT an eps guarantee.",
            )
            return common
        if self.mode is PrivacyMode.LOCAL_INFERENCE:
            common.update(
                epsilon_total=self.epsilon_total,
                adjacency="two signal streams differing in a single interaction of "
                          "the live user, for one modality",
                protected_output="per-modality clipped log-likelihood-ratio vector "
                                 "released to the fusion engine",
                mechanism="Laplace",
                clip_bound_nats=self.clip,
                l1_sensitivity=self.llr_sensitivity(),
                per_modality_epsilon={k: self.epsilon_total * v for k, v in self.budget.items()},
                per_modality_noise_scale={k: self.noise_scale(k) for k in self.budget},
                per_modality_snr={k: self.per_release_snr(k) for k in self.budget},
                cumulative_epsilon_10_interactions=self.basic_composition(10),
                cumulative_epsilon_10_advanced=self.advanced_composition(10),
                caveat="the released tier is an inference about the same user whose "
                       "data is protected; accuracy is therefore bounded by eps",
            )
            return common
        common.update(
            epsilon_total=self.epsilon_total,
            delta=self.corpus_delta,
            adjacency="replace one child in the calibration corpus; tier sizes "
                      "are public",
            protected_output="per-tier means of clipped, standardised linguistic "
                             "feature vectors, computed offline; dispersions are "
                             "public design constants and are not released",
            mechanism="analytic Gaussian (Balle and Wang, 2018)",
            clip_sd=self.corpus_clip_sd,
            n_per_tier=self.corpus_n_per_tier,
            l2_sensitivity=self.corpus_l2_sensitivity(),
            sigma=self.corpus_gaussian_sigma(),
            sigma_classical=self.corpus_classical_sigma(),
            caveat="live-user signals are not themselves DP-protected; they are "
                   "protected architecturally (see PrivacyMode.NONE) and by retention limits",
        )
        return common


def clip_llr(llr: np.ndarray, clip: float) -> np.ndarray:
    """Clip a log-likelihood-ratio vector to [-clip, clip]."""
    return np.clip(llr, -clip, clip)


def laplace_noise(scale: float, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return rng.laplace(loc=0.0, scale=scale, size=shape)


def _std_normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def gaussian_mechanism_delta(epsilon: float, sigma: float, sensitivity: float) -> float:
    """Exact delta of the Gaussian mechanism at a given (epsilon, sigma).

    Balle and Wang (2018), Theorem 8:
        delta = Phi(D/(2 sigma) - eps sigma / D) - e^eps Phi(-D/(2 sigma) - eps sigma / D).
    When the second term underflows the returned delta is an over-estimate, so a
    sigma calibrated against it is conservative.
    """
    a = sensitivity / (2.0 * sigma)
    b = epsilon * sigma / sensitivity
    second = _std_normal_cdf(-a - b)
    tail = math.exp(epsilon) * second if second > 0.0 else 0.0
    return max(_std_normal_cdf(a - b) - tail, 0.0)


def analytic_gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Smallest sigma for which the Gaussian mechanism is (epsilon, delta)-DP.

    delta(sigma) is decreasing in sigma, so bisection on a log scale suffices.
    """
    if epsilon <= 0 or not 0 < delta < 1 or sensitivity <= 0:
        raise ValueError("need epsilon > 0, 0 < delta < 1 and sensitivity > 0")
    lo, hi = 1e-12 * sensitivity, 1e6 * sensitivity
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if gaussian_mechanism_delta(epsilon, mid, sensitivity) > delta:
            lo = mid
        else:
            hi = mid
    return float(hi)
