"""Linguistic features and the corpus-to-tier mapping."""
import pytest

pytest.importorskip("pandas", reason="real-data modules need the [real] extra")

import numpy as np

from safenest.data.corpora import _child_utterances, text_key, tiers_for_grade
from safenest.features import (
    FEATURE_BOUNDS,
    MTLD_CAP,
    Featurizer,
    Sentence,
    clip_features,
    fk_grade,
    from_unit,
    mtld,
    to_unit,
)
from safenest.signals import LINGUISTIC_FEATURES


# ------------------------------------------------------------------ MTLD
def test_mtld_of_a_repeated_token_is_two():
    # Every second token drops the type-token ratio to 0.5, completing a factor.
    assert mtld(["a"] * 100) == pytest.approx(2.0)


def test_mtld_is_capped_when_every_token_is_distinct():
    assert mtld([f"w{i}" for i in range(40)]) == MTLD_CAP


def test_mtld_rises_with_lexical_diversity():
    low = mtld(("the cat sat on the mat and the cat sat " * 5).split())
    high = mtld(("the quick brown fox jumps over a lazy dog while seven "
                 "curious owls watch silently from ancient twisted branches").split() * 2)
    assert high > low


def test_mtld_of_nothing_is_zero():
    assert mtld([]) == 0.0


# ------------------------------------------------------------ readability
def test_flesch_kincaid_matches_the_formula():
    # 0.39 * 100/5 + 11.8 * 150/100 - 15.59
    assert fk_grade(100, 5, 150) == pytest.approx(9.91)


def test_flesch_kincaid_of_empty_text_is_the_floor():
    assert fk_grade(0, 0, 0) == FEATURE_BOUNDS["fk_grade"][0]


# ---------------------------------------------------------------- windows
def _sentence(n: int) -> Sentence:
    return Sentence(tuple(["w"] * n), tuple(["w"] * n), 2)


def test_windows_reach_the_target_length():
    ws = Featurizer.windows([_sentence(30), _sentence(30), _sentence(30), _sentence(30)], 50)
    assert [sum(len(s.words) for s in w) for w in ws] == [60, 60]


def test_a_short_remainder_is_merged_not_dropped():
    ws = Featurizer.windows([_sentence(30), _sentence(30), _sentence(10)], 50)
    assert len(ws) == 1 and sum(len(s.words) for s in ws[0]) == 70


def test_a_long_remainder_is_its_own_window():
    ws = Featurizer.windows([_sentence(60), _sentence(30)], 50)
    assert [sum(len(s.words) for s in w) for w in ws] == [60, 30]


# --------------------------------------------------------- public bounds
def test_unit_scaling_round_trips_inside_the_bounds():
    x = np.array([55.0, 7.2, 14.0, 4.5, 0.03])
    assert np.allclose(from_unit(to_unit(x)), x)


def test_unit_scaling_clips_to_the_public_bounds():
    u = to_unit(np.array([1e9, -1e9, 1e9, -1e9, 2.0]))
    assert np.all((u >= 0.0) & (u <= 1.0))


def test_clip_features_respects_every_bound():
    x = clip_features(np.array([500.0, -50.0, 0.0, 99.0, 3.0]))
    for value, name in zip(x, LINGUISTIC_FEATURES):
        lo, hi = FEATURE_BOUNDS[name]
        assert lo <= value <= hi


# -------------------------------------------------------- grade to tier
@pytest.mark.parametrize("grade,tiers", [
    (6, (3,)), (7, (3, 4)), (8, (4,)), (9, (4,)), (10, (4, 5)), (11, (5,)), (12, (5,)),
])
def test_grades_map_to_the_tiers_of_the_ages_they_span(grade, tiers):
    assert tiers_for_grade(float(grade)) == tiers


def test_a_missing_grade_admits_no_tier():
    assert tiers_for_grade(float("nan")) == ()


def test_the_duplicate_key_ignores_case_and_whitespace():
    assert text_key("Hello   World\n again") == text_key("hello world again")


# ------------------------------------------------------------- CHILDES
CHAT = (
    "@Begin\n"
    "@Participants:\tCHI Target_Child, EXA Investigator\n"
    "@ID:\teng|Gillam|CHI|7;06.|male|TD||Target_Child|||\n"
    "*EXA:\ttell me a story .\n"
    "*CHI:\t<the boy> [/] the boy went to &-um the store .\n"
    "*CHI:\the saw a xxx dog@c <and a> [//] and a cat !\n"
    "@End\n"
)


def test_chat_cleaning_keeps_only_the_childs_words():
    utts, age = _child_utterances(CHAT)
    assert age == pytest.approx(7.5)
    assert utts == ["the boy went to the store", "he saw a dog and a cat"]


def test_chat_without_a_child_age_returns_none():
    utts, age = _child_utterances("*CHI:\thello .\n")
    assert utts == ["hello"] and age is None


# ------------------------------------------------------------- parsing
def test_the_featuriser_parses_depth_and_spelling():
    pytest.importorskip("spacy")
    try:
        f = Featurizer()
    except OSError:
        pytest.skip("spaCy model en_core_web_sm is not installed")
    sents = f.parse(["The cat sat on the mat. My camputer is broken today."], spoken=False)[0]
    assert len(sents) == 2
    assert all(s.depth >= 2 for s in sents)
    vec = f.window_features(sents)
    assert vec[4] == pytest.approx(1 / 11)  # one of eleven checkable words is misspelled
