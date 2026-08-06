"""Developmental tier definitions (Table 2 of the manuscript).

Tiers are the atomic unit of the framework: everything downstream (constraint
lattice, Bayesian estimator, Socratic MDP, feature gating) is indexed by tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Tier(IntEnum):
    """Developmental tiers t1..t5. IntEnum so the lattice order is the int order."""

    T1 = 1  # ages 3-6,   late preoperational
    T2 = 2  # ages 7-9,   early concrete operational
    T3 = 3  # ages 10-12, concrete operational
    T4 = 4  # ages 13-15, early formal operational
    T5 = 5  # ages 16-17, late formal operational

    @property
    def label(self) -> str:
        return f"t{int(self)}"


ALL_TIERS: tuple[Tier, ...] = tuple(Tier)
K_TIERS = len(ALL_TIERS)


@dataclass(frozen=True)
class TierSpec:
    tier: Tier
    age_low: int
    age_high: int
    piaget_stage: str
    erikson_crisis: str
    interaction_mode: str
    session_limit_min: int | None  # None == no limit
    crisis_detection: bool


TIER_SPECS: dict[Tier, TierSpec] = {
    Tier.T1: TierSpec(
        Tier.T1, 3, 6, "late preoperational", "initiative vs. guilt",
        "closed-domain, voice-first", 15, False,
    ),
    Tier.T2: TierSpec(
        Tier.T2, 7, 9, "early concrete operational", "industry vs. inferiority",
        "Socratic-only, simplified language", 30, False,
    ),
    Tier.T3: TierSpec(
        Tier.T3, 10, 12, "concrete operational", "industry vs. inferiority",
        "guided Socratic with metacognitive prompts", 45, True,
    ),
    Tier.T4: TierSpec(
        Tier.T4, 13, 15, "early formal operational", "identity vs. role confusion",
        "semi-autonomous with step verification", 60, True,
    ),
    Tier.T5: TierSpec(
        Tier.T5, 16, 17, "late formal operational", "identity vs. role confusion",
        "near-adult with transparent guardrails", None, True,
    ),
}


def tier_for_age(age: int) -> Tier:
    """Chronological age -> nominal tier. Used only to generate synthetic users
    and for the profile/age-discordance check; the runtime system never sees age."""
    for spec in TIER_SPECS.values():
        if spec.age_low <= age <= spec.age_high:
            return spec.tier
    if age < 3:
        return Tier.T1
    raise ValueError(f"age {age} is outside the 3-17 scope of the framework")
