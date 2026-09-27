"""The checks behind the decision diamonds of the measurement protocol.

The article's protocol figure draws each check as a diamond that can fail;
these tests are the checks not already covered by the lattice and invariant
suites.
"""
import pytest

from safenest.labeling import matrix_label, rubric_label

#: Names through which a labeller could read the specification under test.
POLICY_NAMES = {"access_level", "FEATURE_GATING", "Access", "tier", "capability",
                "matrix_label", "npl_engine"}


def test_rubric_labeller_never_reads_the_policy():
    """The rubric decides from age and the record's own attributes, never from
    the gating matrix, the tier index or the capability the policy gates."""
    code = rubric_label.__code__
    assert not (set(code.co_names) | set(code.co_varnames)) & POLICY_NAMES


def test_the_policy_check_would_catch_a_circular_labeller():
    """The same inspection flags the circular labeller, so it can fail."""
    code = matrix_label.__code__
    assert (set(code.co_names) | set(code.co_varnames)) & POLICY_NAMES


def test_prompt_disjoint_folds_never_split_a_prompt():
    """Every prompt except the grade-6 one lands in a single fold, and every
    essay is assigned."""
    pd = pytest.importorskip("pandas")
    exp15 = pytest.importorskip("experiments.exp15_real_subgroups")
    docs = pd.DataFrame({
        "doc_id": [f"d{i}" for i in range(40)],
        "prompt": [f"p{i % 8}" for i in range(40)],
        "admissible": ["3" if i % 8 == 0 else "4" for i in range(40)],
    })
    fold = exp15._folds(docs)
    assert set(fold) == set(docs["doc_id"])
    per_prompt = docs.assign(fold=docs["doc_id"].map(fold)).groupby("prompt")["fold"].nunique()
    assert (per_prompt.drop("p0") == 1).all()
    assert per_prompt["p0"] > 1  # the grade-6 prompt is spread, as the article states


def test_bootstrap_intervals_cover_the_estimate_and_pair_correctly():
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    exp13 = pytest.importorskip("experiments.exp13_real_estimation")
    rng = np.random.default_rng(0)
    n = 1500
    frame = pd.DataFrame({"doc_id": [f"d{i}" for i in range(n)],
                          "admissible": rng.choice(["3", "4", "5"], n),
                          "outcome": rng.choice(["correct", "under", "over"], n, p=[.6, .2, .2])})
    ci = exp13.bootstrap_ci(frame, rng, b=200)
    acc = (frame["outcome"] == "correct").mean()
    assert ci["accuracy"][0] <= acc <= ci["accuracy"][1]
    same = exp13.paired_difference(frame, frame, "balanced_accuracy", rng, b=50)
    assert same["difference"] == 0 and same["ci"] == [0.0, 0.0]
