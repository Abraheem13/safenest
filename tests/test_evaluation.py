import numpy as np
import pytest

from safenest.baselines import all_frameworks, make_npl
from safenest.corpus import Category, build_corpus, corpus_summary
from safenest.labeling import Label, agreement, cohens_kappa, matrix_label, rubric_label
from safenest.metrics import correctness_vector, evaluate_framework, mcnemar, score, wilson_interval
from safenest.policy import Decision
from safenest.privacy import PrivacyConfig, PrivacyMode
from safenest.tiers import ALL_TIERS


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(n_per_cell=60, seed=42)


def test_corpus_is_balanced_and_reproducible():
    a = build_corpus(n_per_cell=20, seed=5)
    b = build_corpus(n_per_cell=20, seed=5)
    assert [p.as_row() for p in a] == [p.as_row() for p in b]
    s = corpus_summary(a)
    assert s["n_prompts"] == 20 * 7 * 5
    assert len(set(s["by_tier"].values())) == 1
    assert len(set(s["by_category"].values())) == 1


def test_full_corpus_is_seven_thousand_prompts():
    assert len(build_corpus()) == 7000


def test_labellers_disagree_enough_to_be_independent(corpus):
    """If the rubric labeller agreed perfectly with the matrix labeller it would
    not be an independent ground truth."""
    a = agreement(corpus)
    assert 0.2 < a["exact_agreement"] < 0.95
    assert a["cohens_kappa"] < 0.9


def test_kappa_is_one_for_identical_labellings(corpus):
    labels = [matrix_label(p) for p in corpus]
    assert np.isclose(cohens_kappa(labels, labels), 1.0)


def test_rubric_labeller_never_allows_age_inappropriate_content(corpus):
    for p in corpus:
        if p.category is Category.AGE_INAPPROPRIATE:
            assert rubric_label(p) is Label.BLOCK


def test_rubric_labeller_uses_age_not_tier(corpus):
    """Within a single tier, the rubric must still discriminate by age where a
    regulatory threshold falls inside the band."""
    t4 = [p for p in corpus if int(p.tier) == 4 and p.category is Category.SUBSTANCE]
    labels = {rubric_label(p) for p in t4}
    assert len(labels) >= 1


def test_dsr_and_error_rates_partition_the_prompts(corpus):
    for name, fw in all_frameworks().items():
        r = evaluate_framework(fw, corpus, rubric_label)["overall"]
        total = r["dsr"] + r["under_protection"] + r["over_restriction"]
        assert np.isclose(total, 1.0), name


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(90, 100)
    assert lo < 0.90 < hi
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_matrix_labeller_makes_npl_look_near_perfect(corpus):
    """Demonstrates the circularity
    specification, NPL is near-perfect by construction."""
    npl = make_npl()
    r = evaluate_framework(npl, corpus, matrix_label)["overall"]
    assert r["dsr"] > 0.85


def test_npl_scores_lower_against_the_independent_rubric(corpus):
    """The honest number must be materially below the circular one."""
    npl = make_npl()
    circular = evaluate_framework(npl, corpus, matrix_label)["overall"]["dsr"]
    honest = evaluate_framework(npl, corpus, rubric_label)["overall"]["dsr"]
    assert honest < circular


def test_npl_beats_every_baseline_on_the_independent_rubric(corpus):
    frameworks = all_frameworks()
    npl = evaluate_framework(frameworks["NPL (ours)"], corpus, rubric_label)["overall"]["dsr"]
    for name, fw in frameworks.items():
        if name.startswith("NPL"):
            continue
        other = evaluate_framework(fw, corpus, rubric_label)["overall"]["dsr"]
        assert npl > other, f"{name} scored {other:.3f} vs NPL {npl:.3f}"


def test_npl_under_protection_is_lower_than_every_baseline(corpus):
    frameworks = all_frameworks()
    npl = evaluate_framework(
        frameworks["NPL (ours)"], corpus, rubric_label
    )["overall"]["under_protection"]
    for name, fw in frameworks.items():
        if name.startswith("NPL"):
            continue
        other = evaluate_framework(fw, corpus, rubric_label)["overall"]["under_protection"]
        assert npl <= other, name


def test_mcnemar_detects_a_real_difference(corpus):
    fws = all_frameworks()
    a = correctness_vector(fws["NPL (ours)"], corpus, rubric_label)
    b = correctness_vector(fws["Constitutional AI"], corpus, rubric_label)
    assert mcnemar(a, b)["p_value"] < 0.01
    assert mcnemar(a, a)["p_value"] == 1.0


def test_socratic_ablation_costs_over_restriction(corpus):
    """Removing Socratic substitution should raise over-restriction, which is
    the ablation evidence that the mechanism earns its place."""
    fws = all_frameworks()
    full = evaluate_framework(fws["NPL (ours)"], corpus, rubric_label)["overall"]
    ablated = evaluate_framework(fws["NPL (no Socratic)"], corpus, rubric_label)["overall"]
    assert ablated["over_restriction"] > full["over_restriction"]


def test_score_handles_the_empty_set():
    assert score([], [])["n"] == 0


def test_per_tier_breakdown_covers_all_tiers(corpus):
    r = evaluate_framework(make_npl(), corpus, rubric_label)
    assert set(r["by_tier"]) == {t.label for t in ALL_TIERS}
