"""SafeNest / Nested Policy Learning reference implementation.

Every quantitative claim in the manuscript is regenerated from this package by
the scripts in `experiments/`; nothing is transcribed by hand.
"""
from .corpus import Category, Prompt, build_corpus
from .estimator import BayesianAgeEstimator, EstimatorState
from .labeling import Label, matrix_label, rubric_label
from .lattice import Access, Capability, allowed_set, constraint_set, validate_lattice
from .metrics import DSR_DEFINITION, evaluate_framework
from .policy import Decision, NestedPolicyEngine, Response
from .privacy import PrivacyConfig, PrivacyMode, PrivacyUnit
from .signals import PROFILES, SignalModel, linguistic_kl_matrix, min_adjacent_kl
from .socratic import Action, RewardParams, SocraticMDP
from .tiers import ALL_TIERS, TIER_SPECS, Tier, tier_for_age

__version__ = "2.0.0"

__all__ = [
    "Tier", "TIER_SPECS", "ALL_TIERS", "tier_for_age",
    "Capability", "Access", "allowed_set", "constraint_set", "validate_lattice",
    "PrivacyConfig", "PrivacyMode", "PrivacyUnit",
    "SignalModel", "PROFILES", "linguistic_kl_matrix", "min_adjacent_kl",
    "BayesianAgeEstimator", "EstimatorState",
    "NestedPolicyEngine", "Response", "Decision",
    "SocraticMDP", "RewardParams", "Action",
    "build_corpus", "Prompt", "Category",
    "matrix_label", "rubric_label", "Label",
    "evaluate_framework", "DSR_DEFINITION",
]
