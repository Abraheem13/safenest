"""The Nested Policy Engine (Section 3.4) and the four safety invariants
(Section 4).

Five layers with strictly ordered update frequencies f0 >> f1 >> ... >> f4.
Each layer is a constraint function C_i : (response, tier) -> Decision, and the
composite is the conjunction (Equation 12): a response is emitted only if every
layer accepts it. No layer can weaken another layer's decision -- that property
is what Invariant II rests on, and `test_invariants.py` checks it by exhaustive
search rather than by assertion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .lattice import Access, Capability, access_level, allowed_set, validate_lattice
from .tiers import TIER_SPECS, Tier


class Decision(str, Enum):
    ACCEPT = "accept"
    MODIFY = "modify"   # emit, but only via Socratic scaffolding
    REJECT = "reject"

    @property
    def emits_direct_answer(self) -> bool:
        return self is Decision.ACCEPT


#: Conjunction order on decisions: REJECT dominates MODIFY dominates ACCEPT.
_SEVERITY = {Decision.ACCEPT: 0, Decision.MODIFY: 1, Decision.REJECT: 2}


def conjoin(*decisions: Decision) -> Decision:
    """Equation 12. The most severe decision wins; no layer can relax another."""
    return max(decisions, key=lambda d: _SEVERITY[d])


@dataclass(frozen=True)
class Response:
    """A candidate model output, as the policy layers see it."""

    capability: Capability
    tokens: tuple[str, ...] = ()
    fk_grade: float = 0.0
    is_direct_answer: bool = True
    is_academic_query: bool = False
    session_minutes: float = 0.0
    #: Severity of harmful content in [0, 1]. The manuscript's feature-gating
    #: matrix is indexed by capability alone, which cannot distinguish a
    #: general question about alcohol from instructions for acquiring it; the
    #: severity gate below closes that gap.
    harm_severity: float = 0.0


#: Capabilities for which severity, not merely tier, determines admissibility.
SEVERITY_GATED: frozenset[Capability] = frozenset(
    {Capability.CRISIS_CONTENT, Capability.SUBSTANCE_BODYIMAGE, Capability.AGE_INAPPROPRIATE}
)
#: Above this severity the content is refused at every tier, t5 included.
SEVERITY_REJECT_THRESHOLD: float = 0.5


#: Flesch-Kincaid readability ceiling g_k enforced by L0.
READABILITY_CEILING: dict[Tier, float] = {
    Tier.T1: 1.0, Tier.T2: 3.0, Tier.T3: 6.0, Tier.T4: 9.0, Tier.T5: 12.0,
}
#: Tokens outside the tier-appropriate lexicon W_k, keyed by the lowest tier at
#: which the token becomes admissible.
_RESTRICTED_LEXICON: dict[str, Tier] = {
    "suicide": Tier.T3, "self-harm": Tier.T3, "overdose": Tier.T4,
    "alcohol": Tier.T4, "vaping": Tier.T4, "calories": Tier.T4, "diet": Tier.T4,
    "liability": Tier.T5, "prosecution": Tier.T5,
}


class Layer:
    """Base class. `name`, `update_frequency_hz` and `evaluate` are the contract."""

    name: str = "layer"
    update_frequency_hz: float = 0.0

    def evaluate(self, response: Response, tier: Tier) -> Decision:  # pragma: no cover
        raise NotImplementedError

    def degraded_decision(self) -> Decision:
        """Fail-closed behaviour when this layer is unavailable (Invariant IV)."""
        return Decision.REJECT


class L0TokenFilter(Layer):
    """Per-token vocabulary and readability gate. O(1) per token."""

    name = "L0_token_filter"
    update_frequency_hz = 1_000.0

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        if response.fk_grade > READABILITY_CEILING[tier]:
            return Decision.MODIFY  # simplify rather than refuse
        for token in response.tokens:
            required = _RESTRICTED_LEXICON.get(token.lower())
            if required is not None and int(tier) < int(required):
                return Decision.REJECT
        limit = TIER_SPECS[tier].session_limit_min
        if limit is not None and response.session_minutes > limit:
            return Decision.REJECT
        return Decision.ACCEPT

    def degraded_decision(self) -> Decision:
        return Decision.REJECT  # suppress all output


class L1SocraticGuard(Layer):
    """Per-response direct-answer interception. O(|y|)."""

    name = "L1_socratic_guard"
    update_frequency_hz = 1.0

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        # Severity gate first: applied uniformly across tiers, so it tightens
        # every tier equally and cannot break monotonicity (Invariant I).
        if (
            response.capability in SEVERITY_GATED
            and response.harm_severity > SEVERITY_REJECT_THRESHOLD
        ):
            return Decision.REJECT
        level = access_level(response.capability, tier)
        if level is Access.BLOCKED:
            return Decision.REJECT
        if level is Access.SOCRATIC:
            return Decision.MODIFY
        if level is Access.LIMITED:
            # Direct answers to academic queries are withheld below t5.
            if response.is_academic_query and response.is_direct_answer and tier < Tier.T5:
                return Decision.MODIFY
            return Decision.ACCEPT
        return Decision.ACCEPT

    def degraded_decision(self) -> Decision:
        return Decision.MODIFY  # route everything through the t1 Socratic protocol


class L2TierRefiner(Layer):
    """Session-level re-estimation. Asynchronous: contributes no per-response gate."""

    name = "L2_tier_refiner"
    update_frequency_hz = 1.0 / 600.0

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        return Decision.ACCEPT

    def degraded_decision(self) -> Decision:
        return Decision.ACCEPT  # retain the last confirmed tier; no new emission risk


class L3MemoryLayer(Layer):
    """Cross-session competency tracking; unlocks capability *within* a tier."""

    name = "L3_memory_layer"
    update_frequency_hz = 1.0 / 86_400.0

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        return Decision.ACCEPT

    def degraded_decision(self) -> Decision:
        return Decision.ACCEPT  # stateless operation


class L4PolicyStore(Layer):
    """Versioned tier definitions, validated against lattice monotonicity."""

    name = "L4_policy_store"
    update_frequency_hz = 1.0 / (365 * 86_400.0)

    def __init__(self) -> None:
        validate_lattice()  # compile-time check; raises on a bad policy edit

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        return (
            Decision.ACCEPT
            if response.capability in allowed_set(tier) or access_level(response.capability, tier)
            is not Access.BLOCKED
            else Decision.REJECT
        )

    def degraded_decision(self) -> Decision:
        return Decision.REJECT  # hardcoded minimal policy == t1 constraints


@dataclass
class NestedPolicyEngine:
    layers: list[Layer] = field(
        default_factory=lambda: [
            L0TokenFilter(), L1SocraticGuard(), L2TierRefiner(),
            L3MemoryLayer(), L4PolicyStore(),
        ]
    )
    #: Names of layers currently failed, for Invariant IV case analysis.
    failed: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        freqs = [layer.update_frequency_hz for layer in self.layers]
        if any(a <= b for a, b in zip(freqs, freqs[1:])):
            raise ValueError(f"layers must be strictly frequency-ordered, got {freqs}")

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        decisions = [
            layer.degraded_decision() if layer.name in self.failed
            else layer.evaluate(response, tier)
            for layer in self.layers
        ]
        return conjoin(*decisions)

    def with_failures(self, *names: str) -> "NestedPolicyEngine":
        return NestedPolicyEngine(layers=self.layers, failed=frozenset(names))

    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]
