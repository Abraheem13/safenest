"""A region-complete finite abstraction of the response space.

Every synchronous layer of the Nested Policy Engine reads a response only
through a finite set of predicates:

  capability                 one of |U| values
  fk_grade > g_k             for each tier's readability ceiling g_k
  session_minutes > l_k      for each tier's session limit l_k
  harm_severity > theta_sev  the severity gate
  is_direct_answer, is_academic_query
  max over tokens of the lowest tier at which the token is admissible

The asynchronous layers (L2, L3) return a constant, and every degraded
(fail-closed) decision is built from the same predicates. The thresholds on
each numeric attribute are nested, so the realisable predicate combinations for
that attribute are exactly the intervals between consecutive thresholds. One
representative per interval, crossed over all attributes, therefore realises
every input the engine can distinguish, and checking an invariant on these
representatives checks it on *every* response. This is what makes the
enumeration in `tests/test_invariants.py` a verification of the implementation
rather than a sample of it.
"""
from __future__ import annotations

from collections.abc import Iterator

from .lattice import Capability
from .policy import (
    _RESTRICTED_LEXICON, READABILITY_CEILING, SEVERITY_REJECT_THRESHOLD, Response,
)
from .tiers import TIER_SPECS, Tier


def interval_representatives(thresholds: list[float], top: float) -> tuple[float, ...]:
    """One point in each interval of (-inf, t1], (t1, t2], ..., (tn, top].

    Layers compare with a strict `>`, so a value equal to a threshold falls in
    the interval below it; midpoints avoid the boundary altogether.
    """
    ts = sorted(set(thresholds))
    lows = [ts[0] - 1.0] + ts
    highs = ts + [top]
    return tuple(round((lo + hi) / 2.0, 6) for lo, hi in zip(lows, highs))


def readability_representatives() -> tuple[float, ...]:
    return interval_representatives(list(READABILITY_CEILING.values()),
                                    top=max(READABILITY_CEILING.values()) + 2.0)


def session_representatives() -> tuple[float, ...]:
    limits = [s.session_limit_min for s in TIER_SPECS.values() if s.session_limit_min]
    return interval_representatives([float(x) for x in limits], top=max(limits) + 20.0)


def severity_representatives() -> tuple[float, ...]:
    return (SEVERITY_REJECT_THRESHOLD / 2.0, (1.0 + SEVERITY_REJECT_THRESHOLD) / 2.0)


def lexicon_representatives() -> tuple[tuple[str, ...], ...]:
    """No restricted token, then one token for each distinct admissibility tier."""
    by_tier: dict[Tier, str] = {}
    for token, tier in sorted(_RESTRICTED_LEXICON.items()):
        by_tier.setdefault(tier, token)
    return ((),) + tuple((by_tier[t],) for t in sorted(by_tier))


def abstract_responses() -> Iterator[Response]:
    """The full product of region representatives."""
    for cap in Capability:
        for fk in readability_representatives():
            for direct in (True, False):
                for academic in (True, False):
                    for sev in severity_representatives():
                        for minutes in session_representatives():
                            for tokens in lexicon_representatives():
                                yield Response(
                                    capability=cap, tokens=tokens, fk_grade=fk,
                                    is_direct_answer=direct, is_academic_query=academic,
                                    harm_severity=sev, session_minutes=minutes,
                                )


def abstraction_size() -> int:
    return (len(Capability) * len(readability_representatives()) * 2 * 2
            * len(severity_representatives()) * len(session_representatives())
            * len(lexicon_representatives()))
