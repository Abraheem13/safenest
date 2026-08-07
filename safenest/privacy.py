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
the clip shrinks noise and signal by the same factor. At eps = 1.0 and K = 5
that ratio is at most 0.125, so tens of interactions cannot recover a reliable
five-way decision. `LOCAL_INFERENCE` implements this honestly and Experiment 05
reports the resulting curve.

The defensible reading of the architecture is therefore `CORPUS` (plus
`NONE`'s data minimisation, which is architectural rather than statistical):

  CORPUS  The eps-DP guarantee protects the children in the *calibration
          corpora* (CHILDES-db, Oxford Children's Language Corpus) whose data
          parameterise the tier-conditional likelihoods. The privacy unit is
          one corpus child; adjacency is add/remove one child's transcripts;
          the protected output is the released likelihood parameter vector.
          Live inference then runs on noise-free released parameters, so
          classification accuracy is preserved.

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


# Table 3 budget split across modalities. Shares must sum to 1.0.
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
    #: CORPUS mode: (delta, per-child L2 sensitivity of the parameter vector).
    corpus_delta: float = 1e-5
    corpus_l2_sensitivity: float = 1.0
    corpus_n_children: int = 2000

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

        The vector has K-1 free coordinates once a reference tier is fixed;
        each is a difference of two clipped log-likelihoods and so spans
        [-2C, 2C], and one privacy unit can move all of them at once.
        """
        return 2.0 * self.clip * (self.n_tiers - 1)

    def noise_scale(self, modality: str) -> float:
        """Laplace scale b = Delta / eps_m for `modality` (Equation 8)."""
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
    def corpus_gaussian_sigma(self) -> float:
        """Gaussian-mechanism sigma for a one-shot release of the tier-conditional
        likelihood parameters, per the analytic calibration
        sigma = Delta_2 sqrt(2 ln(1.25/delta)) / eps."""
        return float(
            self.corpus_l2_sensitivity
            * np.sqrt(2.0 * np.log(1.25 / self.corpus_delta))
            / self.epsilon_total
        )

    def corpus_parameter_noise_std(self) -> float:
        """Per-parameter noise std after averaging over `corpus_n_children`.

        A parameter estimated as a mean over N children has per-child
        sensitivity Delta_2 / N, so the released estimate is perturbed by
        sigma / N -- negligible for corpus sizes in the thousands, which is why
        CORPUS mode preserves inference accuracy.
        """
        return self.corpus_gaussian_sigma() / max(self.corpus_n_children, 1)

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
            adjacency="add or remove all transcripts of one child from the "
                      "calibration corpus",
            protected_output="released tier-conditional likelihood parameters "
                             "(means and covariances), computed offline",
            mechanism="Gaussian",
            l2_sensitivity=self.corpus_l2_sensitivity,
            sigma=self.corpus_gaussian_sigma(),
            per_parameter_noise_std=self.corpus_parameter_noise_std(),
            corpus_n_children=self.corpus_n_children,
            caveat="live-user signals are not themselves DP-protected; they are "
                   "protected architecturally (see PrivacyMode.NONE) and by retention limits",
        )
        return common


def clip_llr(llr: np.ndarray, clip: float) -> np.ndarray:
    """Clip a log-likelihood-ratio vector to [-clip, clip]."""
    return np.clip(llr, -clip, clip)


def laplace_noise(scale: float, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return rng.laplace(loc=0.0, scale=scale, size=shape)
