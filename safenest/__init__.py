"""SafeNest / Nested Policy Learning reference implementation.

Every quantitative claim in the manuscript is regenerated from this package by
the scripts in `experiments/`; nothing is transcribed by hand.
"""
from .tiers import Tier, TIER_SPECS, ALL_TIERS, tier_for_age
from .lattice import Capability, Access, allowed_set, constraint_set, validate_lattice
from .privacy import PrivacyConfig, PrivacyMode, PrivacyUnit
from .signals import SignalModel, PROFILES, linguistic_kl_matrix, min_adjacent_kl
from .estimator import BayesianAgeEstimator, EstimatorState
from .policy import NestedPolicyEngine, Response, Decision
from .socratic import SocraticMDP, RewardParams, Action
from .corpus import build_corpus, Prompt, Category
from .labeling import matrix_label, rubric_label, Label
from .metrics import evaluate_framework, DSR_DEFINITION

__version__ = "1.0.0"

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
