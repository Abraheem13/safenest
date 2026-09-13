"""The Nested Policy Engine and the three safety invariants.

Five layers with strictly ordered update frequencies f0 >> f1 >> ... >> f4.
Each layer is a constraint function C_i : (response, tier) -> Decision, and the
composite is the conjunction: a response is emitted only if every
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
    """the conjunction rule. The most severe decision wins; no layer can relax another."""
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

    def degraded_decision(self, response: Response, tier: Tier) -> Decision:
        """Fail-closed behaviour when this layer is unavailable (Invariant III).

        The fallback is evaluated against the response, because a decision that
        ignores it can be *less* restrictive than the one the layer would have
        made -- which breaks Invariant III rather than upholding it.
        """
        return Decision.REJECT


class L0TokenFilter(Layer):
    """Per-token vocabulary and readability gate. O(1) per token."""

    name = "L0_token_filter"
    update_frequency_hz = 1_000.0

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        """Conjoin all three gates rather than returning on the first that fires.

        Returning early on readability would skip the session-limit check, and
        because the readability ceiling and the session limit both rise with
        tier, that produces a decision sequence that is *not* monotone in tier
        -- a direct violation of Invariant I. Conjoining is what makes the
        invariant hold; `test_invariants.py` covers the case that exposed it.
        """
        decisions = [Decision.ACCEPT]
        if response.fk_grade > READABILITY_CEILING[tier]:
            decisions.append(Decision.MODIFY)  # simplify rather than refuse
        for token in response.tokens:
            required = _RESTRICTED_LEXICON.get(token.lower())
            if required is not None and int(tier) < int(required):
                decisions.append(Decision.REJECT)
                break
        limit = TIER_SPECS[tier].session_limit_min
        if limit is not None and response.session_minutes > limit:
            decisions.append(Decision.REJECT)
        return conjoin(*decisions)

    def degraded_decision(self, response: Response, tier: Tier) -> Decision:
        return Decision.REJECT  # suppress all output


class L1SocraticGuard(Layer):
    """Per-response direct-answer interception. O(|y|).

    `scaffold_below_t3` toggles the amendment described in the manuscript: when
    False the engine reproduces the pre-amendment specification, in which a
    direct answer to a non-academic request was delivered to t1-t2 users. It
    exists so both reported DSR figures are regenerable from this package.
    """

    name = "L1_socratic_guard"
    update_frequency_hz = 1.0

    def __init__(self, scaffold_below_t3: bool = True,
                 severity_gate: bool = True) -> None:
        self.scaffold_below_t3 = scaffold_below_t3
        #: Disabling the gate reproduces the capability-indexed matrix alone,
        #: which is what the component ablation isolates.
        self.severity_gate = severity_gate

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        # Severity gate first: applied uniformly across tiers, so it tightens
        # every tier equally and cannot break monotonicity (Invariant I).
        if (
            self.severity_gate
            and response.capability in SEVERITY_GATED
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
            # Below concrete operations (t3), a direct answer to *any* request
            # is scaffolded rather than delivered. A capability-indexed gating
            # matrix cannot express this on its own: "Limited" is one bit, and
            # the distinction that matters at t1-t2 is between answering a
            # child's question and helping them reason toward it. Grounded in
            # Piaget's account of preoperational and early concrete reasoning,
            # where an authoritative answer is accepted without evaluation.
            if self.scaffold_below_t3 and response.is_direct_answer and tier < Tier.T3:
                return Decision.MODIFY
            return Decision.ACCEPT
        return Decision.ACCEPT

    def degraded_decision(self, response: Response, tier: Tier) -> Decision:
        """Route everything through the t1 Socratic protocol.

        Returning a bare MODIFY would be unsafe: L1 is the only layer applying
        the severity gate, so a flat MODIFY lets high-severity crisis and
        substance content be scaffolded when it should be refused. Evaluating
        at t1 -- the most restrictive tier -- is both what the manuscript
        describes and what Invariant III requires.
        """
        return conjoin(Decision.MODIFY, self.evaluate(response, Tier.T1))


class L2TierRefiner(Layer):
    """Session-level re-estimation. Asynchronous: contributes no per-response gate."""

    name = "L2_tier_refiner"
    update_frequency_hz = 1.0 / 600.0

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        return Decision.ACCEPT

    def degraded_decision(self, response: Response, tier: Tier) -> Decision:
        return Decision.ACCEPT  # retain the last confirmed tier; no new emission risk


class L3MemoryLayer(Layer):
    """Cross-session competency tracking; unlocks capability *within* a tier."""

    name = "L3_memory_layer"
    update_frequency_hz = 1.0 / 86_400.0

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        return Decision.ACCEPT

    def degraded_decision(self, response: Response, tier: Tier) -> Decision:
        return Decision.ACCEPT  # stateless operation


class L4PolicyStore(Layer):
    """Versioned tier definitions, validated against lattice monotonicity."""

    name = "L4_policy_store"
    update_frequency_hz = 1.0 / (365 * 86_400.0)

    def __init__(self) -> None:
        validate_lattice()  # compile-time check; raises on a bad policy edit

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        blocked = access_level(response.capability, tier) is Access.BLOCKED
        return Decision.REJECT if blocked else Decision.ACCEPT

    def degraded_decision(self, response: Response, tier: Tier) -> Decision:
        # Hardcoded minimal policy: the t1 constraint set.
        return self.evaluate(response, Tier.T1)


@dataclass
class NestedPolicyEngine:
    layers: list[Layer] = field(
        default_factory=lambda: [
            L0TokenFilter(), L1SocraticGuard(), L2TierRefiner(),
            L3MemoryLayer(), L4PolicyStore(),
        ]
    )
    #: Names of layers currently failed, for Invariant III case analysis.
    failed: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        freqs = [layer.update_frequency_hz for layer in self.layers]
        if any(a <= b for a, b in zip(freqs, freqs[1:])):
            raise ValueError(f"layers must be strictly frequency-ordered, got {freqs}")

    def evaluate(self, response: Response, tier: Tier) -> Decision:
        decisions = [
            layer.degraded_decision(response, tier) if layer.name in self.failed
            else layer.evaluate(response, tier)
            for layer in self.layers
        ]
        return conjoin(*decisions)

    def with_failures(self, *names: str) -> "NestedPolicyEngine":
        return NestedPolicyEngine(layers=self.layers, failed=frozenset(names))

    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]
