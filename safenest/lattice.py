"""Developmental constraint lattice and the feature-gating matrix.

The lattice is over constraint sets C_k subset of U. Tier order is
t_i <= t_j  iff  C_j subset-or-equal C_i, so t1 (most restrictive) is bottom and
t5 is top. Meet = union of constraints, join = intersection.
"""
from __future__ import annotations

from enum import Enum

from .tiers import ALL_TIERS, Tier


class Capability(str, Enum):
    """Elements of the universe U that the foundation model can produce."""

    HOMEWORK_ANSWER = "homework_answer"
    OPEN_ENDED_CHAT = "open_ended_chat"
    CODE_GENERATION = "code_generation"
    ESSAY_WRITING = "essay_writing"
    FREEFORM_TEXT_OUTPUT = "freeform_text_output"
    CRISIS_CONTENT = "crisis_content"
    SUBSTANCE_BODYIMAGE = "substance_bodyimage"
    AGE_INAPPROPRIATE = "age_inappropriate"
    LEGAL_RISK_CONTENT = "legal_risk_content"


UNIVERSE: frozenset[Capability] = frozenset(Capability)


class Access(str, Enum):
    """Access levels of the feature-gating matrix."""

    BLOCKED = "blocked"
    SOCRATIC = "socratic"      # guided scaffolding only, never a direct answer
    LIMITED = "limited"        # permitted but monitored / rate-limited
    AVAILABLE = "available"

    @property
    def is_constrained(self) -> bool:
        """A capability counts as constrained unless it is fully available."""
        return self is not Access.AVAILABLE


# Feature-gating matrix. Rows are capabilities, columns are tiers.
FEATURE_GATING: dict[Capability, dict[Tier, Access]] = {
    # An earlier manuscript table recorded "Blocked" for t2-t4 here, but the prose
    # of Section 3.5 says a direct answer to assessed work is *intercepted and
    # replaced* by guided questioning -- which is SOCRATIC, not BLOCKED. The
    # two readings differ: BLOCKED refuses the child outright and scores as
    # over-restriction against any external rubric. The prose is followed.
    Capability.HOMEWORK_ANSWER: {
        Tier.T1: Access.BLOCKED, Tier.T2: Access.SOCRATIC, Tier.T3: Access.SOCRATIC,
        Tier.T4: Access.SOCRATIC, Tier.T5: Access.LIMITED,  # explicit toggle
    },
    Capability.OPEN_ENDED_CHAT: {
        Tier.T1: Access.BLOCKED, Tier.T2: Access.LIMITED, Tier.T3: Access.LIMITED,
        Tier.T4: Access.AVAILABLE, Tier.T5: Access.AVAILABLE,
    },
    Capability.CODE_GENERATION: {
        Tier.T1: Access.BLOCKED, Tier.T2: Access.BLOCKED, Tier.T3: Access.SOCRATIC,
        Tier.T4: Access.SOCRATIC, Tier.T5: Access.AVAILABLE,
    },
    Capability.ESSAY_WRITING: {
        Tier.T1: Access.BLOCKED, Tier.T2: Access.BLOCKED, Tier.T3: Access.SOCRATIC,
        Tier.T4: Access.AVAILABLE, Tier.T5: Access.AVAILABLE,
    },
    Capability.FREEFORM_TEXT_OUTPUT: {
        Tier.T1: Access.BLOCKED, Tier.T2: Access.LIMITED, Tier.T3: Access.LIMITED,
        Tier.T4: Access.AVAILABLE, Tier.T5: Access.AVAILABLE,
    },
    Capability.CRISIS_CONTENT: {
        # Never "available": crisis material is always routed to a safe protocol.
        Tier.T1: Access.BLOCKED, Tier.T2: Access.BLOCKED, Tier.T3: Access.SOCRATIC,
        Tier.T4: Access.SOCRATIC, Tier.T5: Access.SOCRATIC,
    },
    Capability.SUBSTANCE_BODYIMAGE: {
        Tier.T1: Access.BLOCKED, Tier.T2: Access.BLOCKED, Tier.T3: Access.BLOCKED,
        Tier.T4: Access.BLOCKED, Tier.T5: Access.LIMITED,
    },
    Capability.AGE_INAPPROPRIATE: {
        Tier.T1: Access.BLOCKED, Tier.T2: Access.BLOCKED, Tier.T3: Access.BLOCKED,
        Tier.T4: Access.BLOCKED, Tier.T5: Access.BLOCKED,
    },
    Capability.LEGAL_RISK_CONTENT: {
        Tier.T1: Access.BLOCKED, Tier.T2: Access.BLOCKED, Tier.T3: Access.BLOCKED,
        Tier.T4: Access.BLOCKED, Tier.T5: Access.LIMITED,
    },
}


def access_level(cap: Capability, tier: Tier) -> Access:
    return FEATURE_GATING[cap][tier]


def constraint_set(tier: Tier) -> frozenset[Capability]:
    """C_k: capabilities that are restricted (not fully available) at `tier`."""
    return frozenset(c for c in Capability if FEATURE_GATING[c][tier].is_constrained)


def allowed_set(tier: Tier) -> frozenset[Capability]:
    """A(t_k) = U \\ C_k."""
    return UNIVERSE - constraint_set(tier)


def leq(a: Tier, b: Tier) -> bool:
    """Lattice order: a <= b iff C_b subset-or-equal C_a."""
    return constraint_set(b) <= constraint_set(a)


def meet(a: Tier, b: Tier) -> Tier:
    """t_i meet t_j: most restrictive, C = C_i union C_j.

    The tier set is a chain under `leq`, so the union of two constraint sets is
    realised by the lower tier; `validate_lattice` checks that the chain holds.
    """
    return a if int(a) <= int(b) else b


def join(a: Tier, b: Tier) -> Tier:
    """t_i join t_j: least restrictive, C = C_i intersect C_j."""
    return a if int(a) >= int(b) else b


def validate_lattice() -> None:
    """Compile-time validation performed by the Policy Store (L4).

    Enforces the chain C_1 superset C_2 superset ... superset C_5, which is what
    Invariant I (tier monotonicity) rests on. Raises if a policy edit breaks it.
    """
    for lo, hi in zip(ALL_TIERS, ALL_TIERS[1:]):
        c_lo, c_hi = constraint_set(lo), constraint_set(hi)
        if not c_hi <= c_lo:
            offending = sorted(c.value for c in (c_hi - c_lo))
            raise ValueError(
                f"lattice violation: C({hi.label}) is not a subset of C({lo.label}); "
                f"offending capabilities: {offending}"
            )
    # Monotonicity of the *access level* per capability, not just set membership:
    # a capability may never become more restricted as the tier increases.
    order = {Access.BLOCKED: 0, Access.SOCRATIC: 1, Access.LIMITED: 2, Access.AVAILABLE: 3}
    for cap in Capability:
        levels = [order[FEATURE_GATING[cap][t]] for t in ALL_TIERS]
        if any(b < a for a, b in zip(levels, levels[1:])):
            raise ValueError(f"non-monotone access levels for capability {cap.value}: {levels}")
