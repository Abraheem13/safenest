"""Shared helpers for the real-data experiments (Experiments 13-16).

The processed tables are produced by `scripts/prepare_features.py`; each
experiment skips cleanly, with a message, if a corpus it needs is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safenest.data.registry import PROCESSED  # noqa: E402
from safenest.learning import (  # noqa: E402
    DiscriminativeTierModel,
    GaussianTierModel,
    SpecifiedTierModel,
)

#: Window size (words per interaction) for the headline analyses; 25 and 100
#: are reported as a sensitivity analysis.
WINDOW = 50
HORIZONS = (1, 3, 5, 10)
N_FOLDS = 5
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


class MissingCorpus(RuntimeError):
    pass


def load_corpus(name: str, window: int = WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(documents, windows) for a processed corpus, windows carrying doc metadata."""
    docs_path = PROCESSED / f"{name}_docs.parquet"
    win_path = PROCESSED / f"{name}_w{window}.parquet"
    if not docs_path.exists() or not win_path.exists():
        raise MissingCorpus(f"{name}: run scripts/prepare_features.py first")
    docs = pd.read_parquet(docs_path)
    wins = pd.read_parquet(win_path).merge(
        docs[["doc_id", "tier", "admissible"]], on="doc_id", how="inner")
    # Corpora reuse identifiers (ELLIPSE and PERSUADE share Kaggle essay IDs),
    # so every identifier is prefixed with its corpus.
    docs["doc_id"] = name + ":" + docs["doc_id"].astype(str)
    wins["doc_id"] = name + ":" + wins["doc_id"].astype(str)
    docs = docs[docs["doc_id"].isin(wins["doc_id"])].reset_index(drop=True)
    return docs, wins


def folds(docs: pd.DataFrame, k: int = N_FOLDS, seed: int = 0) -> dict[str, int]:
    """Assign each document to a fold, stratified by admissible-tier label."""
    rng = np.random.default_rng(seed)
    out: dict[str, int] = {}
    for _, grp in docs.groupby("admissible"):
        ids = grp["doc_id"].to_numpy().copy()
        rng.shuffle(ids)
        for i, d in enumerate(ids):
            out[d] = i % k
    return out


def embeddings_for(name: str, wins: pd.DataFrame, window: int = WINDOW) -> np.ndarray:
    """Embeddings of every window of a corpus, cached as .npy next to the features."""
    path = PROCESSED / f"{name}_w{window}_minilm.npy"
    if path.exists():
        e = np.load(path)
        if len(e) == len(wins):
            return e.astype(np.float32)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDER, device="mps")
    e = model.encode(wins["text"].tolist(), batch_size=256, show_progress_bar=False,
                     normalize_embeddings=True)
    np.save(path, e.astype(np.float16))  # half precision: disk space, not accuracy
    return e.astype(np.float32)


def model_zoo(embed_matrix: np.ndarray | None = None) -> dict[str, callable]:
    """Factories for every tier estimator compared on real data."""
    from lightgbm import LGBMClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    zoo = {
        "NPL (specified)": lambda: SpecifiedTierModel(),
        "NPL (learned)": lambda: GaussianTierModel(),
        "Logistic (features)": lambda: DiscriminativeTierModel(
            make=lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            kind="features"),
        "Gradient boosting (features)": lambda: DiscriminativeTierModel(
            make=lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                        min_child_samples=50, subsample=0.8,
                                        subsample_freq=1, colsample_bytree=1.0,
                                        random_state=0, verbose=-1),
            kind="features"),
        "Logistic (TF-IDF)": lambda: DiscriminativeTierModel(
            make=lambda: make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000,
                                sublinear_tf=True),
                LogisticRegression(max_iter=2000, C=2.0)),
            kind="text"),
    }
    if embed_matrix is not None:
        zoo["Logistic (MiniLM)"] = lambda: _EmbeddingModel(embed_matrix)
    return zoo


class _EmbeddingModel(DiscriminativeTierModel):
    """Logistic regression on precomputed sentence embeddings, indexed by row."""

    def __init__(self, matrix: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        super().__init__(make=lambda: LogisticRegression(max_iter=3000, C=4.0),
                         kind="rows")
        self._matrix = matrix

    def _inputs(self, frame: pd.DataFrame):
        return self._matrix[frame["_row"].to_numpy()]

    def _weight_arg(self) -> str:
        return "sample_weight"
