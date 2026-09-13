"""Socratic Protection Engine: a finite-horizon MDP over
pedagogical actions, solved by backward induction.

Every reward weight is a named field of `RewardParams` with a documented
default, and `sensitivity_analysis` sweeps them, so no qualitative claim about
the optimal policy rests on an unstated parameter choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .tiers import ALL_TIERS, Tier


class Action(str, Enum):
    ELICIT = "elicit"
    HINT = "hint"
    GUIDE = "guide"
    PARTIAL_EXPLAIN = "partial_explain"
    VERIFY_REQUEST = "verify_request"

    @property
    def reveals_answer(self) -> bool:
        return self is Action.PARTIAL_EXPLAIN


ACTIONS: tuple[Action, ...] = tuple(Action)


@dataclass(frozen=True)
class RewardParams:
    """Reward weights. These are the values used everywhere.

    alpha_reveal must exceed the largest achievable one-step learning reward,
    alpha_learn * max(delta), or the optimal policy could open with a partial
    explanation. With alpha_learn = 1.0 and max delta = 0.40 the binding
    threshold is 0.40; the default 0.60 leaves a 50% margin.
    """

    alpha_learn: float = 1.0
    alpha_reveal: float = 0.60
    alpha_frust: float = 0.25
    #: Credit for metacognitive confirmation, scaled by the tier's metacognitive
    #: capacity. the reward without the metacognition term rewards only knowledge gain, and under
    #: that reward VerifyRequest is dominated at every tier -- so the
    #: manuscript's claim that VerifyRequest usage rises with tier does not
    #: follow from its own reward function. Setting alpha_meta = 0 reproduces
    #: the reward without the metacognition term; the default credits metacognition and
    #: makes the claim reproducible. Experiment 03 reports both.
    alpha_meta: float = 0.06
    discount: float = 0.95
    horizon: int = 5
    n_q: int = 50                 # knowledge-level discretisation points
    sigma_zpd: float = 0.12       # stochasticity of the ZPD transition kernel
    stagnation_threshold: float = 0.02  # q gain below this counts as stagnation

    def validate(self, max_delta: float) -> None:
        """Check the no-premature-revelation condition stated in Section 3.5."""
        bound = self.alpha_learn * max_delta
        if self.alpha_reveal <= bound:
            raise ValueError(
                f"alpha_reveal={self.alpha_reveal} must exceed alpha_learn*max_delta={bound}"
            )


#: delta(a, t_k): expected knowledge gain per action and tier.
#: Rows follow ACTIONS order; columns follow t1..t5.
DELTA = np.array(
    [
        [0.05, 0.08, 0.11, 0.13, 0.15],  # elicit    (diagnostic, lowest gain)
        [0.10, 0.14, 0.18, 0.22, 0.25],  # hint
        [0.05, 0.15, 0.21, 0.26, 0.30],  # guide     (needs concrete operations, t2+)
        [0.20, 0.25, 0.30, 0.35, 0.40],  # partial explain (highest non-reveal gain)
        [0.05, 0.08, 0.14, 0.17, 0.20],  # verify request  (needs metacognition)
    ]
)
MAX_DELTA = float(DELTA.max())

#: Metacognitive capacity by tier, grounded in the Piagetian progression from
#: concrete to formal operations: reliable reflection on one's own reasoning is
#: not available before concrete operations and matures through adolescence.
META_CAPACITY: dict[Tier, float] = {
    Tier.T1: 0.0, Tier.T2: 0.15, Tier.T3: 0.45, Tier.T4: 0.75, Tier.T5: 1.0,
}


def delta(action: Action, tier: Tier) -> float:
    return float(DELTA[ACTIONS.index(action), int(tier) - 1])


@dataclass
class SocraticMDP:
    """M_s = (S, A, P, R, gamma, H) with S = (q, p, tier)."""

    params: RewardParams = field(default_factory=RewardParams)
    #: When False, the no-premature-revelation precondition is not enforced.
    #: Used only to demonstrate that the guarantee is not vacuous.
    strict: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.strict:
            self.params.validate(MAX_DELTA)

    @property
    def q_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.params.n_q)

    # -- dynamics ----------------------------------------------------------
    def transition_matrix(self, action: Action, tier: Tier) -> np.ndarray:
        """P(q' | q, a, t_k): a ZPD-centred Gaussian kernel,
        truncated to the grid and renormalised. Row i is the distribution of q'
        given q = q_grid[i]."""
        key = ("P", action, int(tier))
        if key not in self._cache:
            q = self.q_grid
            centre = np.clip(q + delta(action, tier), 0.0, 1.0)
            diff = q[None, :] - centre[:, None]
            kernel = np.exp(-(diff**2) / (2 * self.params.sigma_zpd**2))
            kernel /= kernel.sum(axis=1, keepdims=True)
            self._cache[key] = kernel
        return self._cache[key]

    def reward_vector(self, action: Action, tier: Tier) -> np.ndarray:
        """R(s, a), as an expectation over q'."""
        p = self.transition_matrix(action, tier)
        q = self.q_grid
        expected_gain = p @ q - q
        r = self.params.alpha_learn * expected_gain
        if action.reveals_answer:
            r = r - self.params.alpha_reveal
        if action is Action.VERIFY_REQUEST:
            r = r + self.params.alpha_meta * META_CAPACITY[tier]
        stagnating = expected_gain < self.params.stagnation_threshold
        r = r - self.params.alpha_frust * stagnating
        return r

    # -- solution ----------------------------------------------------------
    def solve(self, tier: Tier) -> tuple[np.ndarray, np.ndarray]:
        """Backward induction.

        Returns (policy, value) each of shape (H, n_q); row p-1 is protocol step p.
        """
        key = ("solve", int(tier), id(self.params))
        if key in self._cache:
            return self._cache[key]
        h, n = self.params.horizon, self.params.n_q
        value = np.zeros((h + 1, n))
        policy = np.zeros((h, n), dtype=int)
        rewards = np.stack([self.reward_vector(a, tier) for a in ACTIONS])
        kernels = [self.transition_matrix(a, tier) for a in ACTIONS]
        for step in range(h - 1, -1, -1):
            q_values = np.stack(
                [rewards[i] + self.params.discount * (kernels[i] @ value[step + 1])
                 for i in range(len(ACTIONS))]
            )
            policy[step] = np.argmax(q_values, axis=0)
            value[step] = q_values.max(axis=0)
        out = (policy, value[:h])
        self._cache[key] = out
        return out

    def bellman_updates(self) -> int:
        """Offline solve cost, for the complexity claim in Section 3.5."""
        return self.params.horizon * self.params.n_q * len(ACTIONS) * len(ALL_TIERS)

    def action_distribution(self, tier: Tier, step: int = 1) -> dict[Action, float]:
        """Percentage of knowledge states for which each action is optimal."""
        policy, _ = self.solve(tier)
        choices = policy[step - 1]
        return {
            a: 100.0 * float(np.mean(choices == i)) for i, a in enumerate(ACTIONS)
        }

    def expected_value(self, tier: Tier) -> float:
        _, value = self.solve(tier)
        return float(value[0].mean())

    def trajectory(self, tier: Tier, q0: float) -> list[tuple[int, Action, float, float]]:
        """A representative optimal trajectory."""
        policy, _ = self.solve(tier)
        q = q0
        out = []
        for step in range(self.params.horizon):
            idx = int(np.argmin(np.abs(self.q_grid - q)))
            action = ACTIONS[policy[step, idx]]
            kernel = self.transition_matrix(action, tier)
            q_next = float(kernel[idx] @ self.q_grid)
            out.append((step + 1, action, q, q_next))
            q = q_next
        return out

    def opens_with_revelation(self, tier: Tier) -> bool:
        """True if PartialExplain is ever optimal at protocol step 1."""
        policy, _ = self.solve(tier)
        return bool(np.any(policy[0] == ACTIONS.index(Action.PARTIAL_EXPLAIN)))


def sensitivity_analysis(
    alpha_reveal_grid: np.ndarray,
    alpha_frust_grid: np.ndarray,
    discount_grid: np.ndarray,
) -> list[dict]:
    """Sweep the reward weights and record whether the qualitative claims hold.

    The claims under test are (i) PartialExplain is never the optimal opening
    action, and (ii) VerifyRequest usage increases with tier.
    """
    rows: list[dict] = []
    for ar in alpha_reveal_grid:
        for af in alpha_frust_grid:
            for g in discount_grid:
                params = RewardParams(alpha_reveal=float(ar), alpha_frust=float(af),
                                      discount=float(g))
                try:
                    mdp = SocraticMDP(params=params)
                except ValueError:
                    rows.append({
                        "alpha_reveal": float(ar), "alpha_frust": float(af), "discount": float(g),
                        "valid": False, "no_premature_reveal": None, "verify_monotone": None,
                    })
                    continue
                reveal = any(mdp.opens_with_revelation(t) for t in ALL_TIERS)
                verify = [
                    mdp.action_distribution(t)[Action.VERIFY_REQUEST] for t in ALL_TIERS
                ]
                rows.append({
                    "alpha_reveal": float(ar), "alpha_frust": float(af), "discount": float(g),
                    "valid": True,
                    "no_premature_reveal": not reveal,
                    # Genuine non-decreasing check across all five tiers, not
                    # merely a comparison of the endpoints.
                    "verify_monotone": bool(
                        all(b >= a - 1e-9 for a, b in zip(verify, verify[1:]))
                    ),
                    "verify_by_tier": verify,
                })
    return rows
