"""Multi-signal Bayesian age assurance.

Sequential posterior update over tiers from per-modality log-likelihood ratios,
with a fail-safe default to t1, Mahalanobis bypass detection and an explicit
profile/attested-age discordance check for atypically developing users.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .privacy import PrivacyConfig, PrivacyMode, clip_llr, laplace_noise
from .signals import MODALITIES, SignalModel
from .tiers import ALL_TIERS, K_TIERS, Tier

#: Confidence threshold gamma below which the estimator falls back to t1.
DEFAULT_GAMMA: float = 0.55
#: Chi-square critical value at alpha = 0.01 with 5 degrees of freedom, used as
#: the Mahalanobis bypass-detection threshold.
MAHALANOBIS_CHI2_ALPHA01_DF5: float = 15.086


@dataclass
class EstimatorState:
    log_posterior: np.ndarray = field(
        default_factory=lambda: np.log(np.full(K_TIERS, 1.0 / K_TIERS))
    )
    n_interactions: int = 0
    bypass_flags: int = 0
    discordance_flags: int = 0
    held_tier: Tier | None = None  # tier frozen pending additional evidence

    @property
    def posterior(self) -> np.ndarray:
        p = np.exp(self.log_posterior - self.log_posterior.max())
        return p / p.sum()


@dataclass
class BayesianAgeEstimator:
    """Sequential MAP tier estimator with a fail-safe floor.

    `model` supplies the likelihoods. Under `PrivacyMode.CORPUS` the likelihood
    parameters are the DP-released ones and no noise is added at inference;
    under `LOCAL_INFERENCE` calibrated Laplace noise is added to every released
    per-modality LLR vector.
    """

    model: SignalModel = field(default_factory=SignalModel)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    gamma: float = DEFAULT_GAMMA
    #: Two consecutive sessions must agree before a tier *upgrade* takes effect.
    upgrade_confirmations: int = 2
    _pending_upgrade: tuple[Tier, int] | None = field(default=None, repr=False)

    # -- cross-session tier authority --------------------------------------
    def confirm_upgrade(self, current: Tier, proposed: Tier) -> Tier:
        """Two-session confirmation for tier *upgrades* (Invariant II).

        A downgrade takes effect immediately: moving to a more restrictive tier
        is always safe. An upgrade must be proposed by `upgrade_confirmations`
        consecutive sessions before it is granted, so a single anomalous session
        cannot raise a child's tier. Any intervening disagreement resets the
        count. This is session-level state and is therefore exercised across
        sessions rather than within `run_session`.
        """
        if int(proposed) <= int(current):
            self._pending_upgrade = None
            return proposed
        pending, count = self._pending_upgrade or (proposed, 0)
        count = count + 1 if pending is proposed else 1
        if count >= self.upgrade_confirmations:
            self._pending_upgrade = None
            return proposed
        self._pending_upgrade = (proposed, count)
        return current

    # -- single interaction ------------------------------------------------
    def observe(
        self, state: EstimatorState, signals: dict, rng: np.random.Generator
    ) -> EstimatorState:
        """Fold one interaction's signals into the posterior."""
        total = np.zeros(K_TIERS)
        for modality in MODALITIES:
            llr = self.model.llr_vector(modality, signals[modality])
            if self.privacy.mode is PrivacyMode.LOCAL_INFERENCE:
                # The released quantity is the K-1 ratios against the reference
                # tier t1, each clipped to [-C, C], which is what fixes the L1
                # sensitivity at 2C(K-1). The t1 coordinate is the reference and
                # is not released. Clipping applies only here, where a
                # sensitivity bound is needed; clipping unconditionally would
                # saturate the t4/t5 contrast for no privacy benefit.
                ratios = clip_llr(llr[1:] - llr[0], self.privacy.clip)
                ratios = ratios + laplace_noise(
                    self.privacy.noise_scale(modality), ratios.shape, rng
                )
                llr = np.concatenate(([0.0], ratios))
            total += llr
        state.log_posterior = state.log_posterior + total
        state.log_posterior -= state.log_posterior.max()
        state.n_interactions += 1
        return state

    # -- decision ----------------------------------------------------------
    def assign(self, state: EstimatorState) -> Tier:
        """MAP tier when confident, else the most restrictive tier."""
        post = state.posterior
        k = int(np.argmax(post))
        if post[k] >= self.gamma:
            return ALL_TIERS[k]
        return Tier.T1

    def confidence(self, state: EstimatorState) -> float:
        return float(state.posterior.max())

    # -- bypass and discordance -------------------------------------------
    def mahalanobis(self, linguistic: np.ndarray, tier: Tier) -> float:
        mean, std = self.model.linguistic_params(tier)
        z = (np.asarray(linguistic) - mean) / std
        return float(np.sqrt(np.sum(z**2)))

    def bypass_detected(self, linguistic: np.ndarray, tier: Tier) -> bool:
        """Flag a session whose signals are implausible for the assigned tier."""
        return self.mahalanobis(linguistic, tier) ** 2 > MAHALANOBIS_CHI2_ALPHA01_DF5

    def discordance_flag(self, linguistic_tier: Tier, attested_tier: Tier | None) -> bool:
        """Profile/attested-age discordance.

        When an external attestation (parent, school device certificate,
        verified account) states a tier that the linguistic profile contradicts
        by more than one tier, the mismatch -- not the linguistic estimate --
        is the signal. This is the mechanism that protects a highly verbal
        8-year-old whose MTLD and parse depth look like t4.
        """
        if attested_tier is None:
            return False
        return abs(int(linguistic_tier) - int(attested_tier)) >= 2

    def resolve(
        self,
        state: EstimatorState,
        attested_tier: Tier | None = None,
        linguistic: np.ndarray | None = None,
    ) -> Tier:
        """Final tier after fail-safe, discordance and bypass handling.

        Protective policy: whenever two sources disagree materially, take the
        meet (most restrictive) rather than the estimate.
        """
        estimated = self.assign(state)
        if attested_tier is not None and self.discordance_flag(estimated, attested_tier):
            state.discordance_flags += 1
            # Trust the externally attested tier, and never exceed it.
            return Tier(min(int(attested_tier), int(estimated)))
        if linguistic is not None and self.bypass_detected(linguistic, estimated):
            state.bypass_flags += 1
            # Hold at the last confirmed tier; t1 when nothing is yet confirmed.
            return state.held_tier or Tier.T1
        # Record the assignment so a later flagged session is held here rather
        # than dropped to t1 (Section "Convergence and bypass detection").
        state.held_tier = estimated
        return estimated

    # -- convenience -------------------------------------------------------
    def run_session(
        self,
        true_tier: Tier,
        n_interactions: int,
        rng: np.random.Generator,
        generating_model: SignalModel | None = None,
        attested_tier: Tier | None = None,
    ) -> tuple[Tier, EstimatorState]:
        """Simulate one user session and return the final tier decision."""
        gen = generating_model or self.model
        state = EstimatorState()
        last_ling = None
        for _ in range(n_interactions):
            signals = gen.sample(true_tier, rng)
            last_ling = signals["linguistic"]
            self.observe(state, signals, rng)
        return self.resolve(state, attested_tier=attested_tier, linguistic=last_ling), state
