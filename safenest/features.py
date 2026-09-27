"""The five linguistic features of the tier estimator, computed from real text.

Features, in the order used everywhere in the package (`signals.LINGUISTIC_FEATURES`):

    mtld                  Measure of Textual Lexical Diversity (McCarthy and
                          Jarvis, 2010), bidirectional, TTR threshold 0.72
    fk_grade              Flesch-Kincaid grade level
    mean_sentence_length  words per sentence (per utterance for speech)
    parse_depth           mean over sentences of the maximum dependency depth
    spelling_error_rate   share of lower-case dictionary-checkable words not
                          found in the English word list of `pyspellchecker`

A child's text is split into *interactions*: consecutive sentences grouped until
a window holds at least `window_words` words, which stands in for one message
to an assistant. Every feature is computed per window, so a child contributes
a sequence of feature vectors, and the estimator accumulates evidence over it
exactly as it does over simulated interactions.

Public bounds (`FEATURE_BOUNDS`) clip each feature to a fixed, data-independent
range. They matter for differential privacy: the sensitivity of a released
statistic is computed from them, so they must not be estimated from the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .signals import LINGUISTIC_FEATURES

#: Data-independent clipping range per feature, fixed before any data was seen.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "mtld": (0.0, 200.0),
    "fk_grade": (-3.4, 25.0),       # -3.40 is the formula's floor
    "mean_sentence_length": (1.0, 60.0),
    "parse_depth": (1.0, 20.0),
    "spelling_error_rate": (0.0, 1.0),
}

MTLD_THRESHOLD = 0.72
MTLD_CAP = FEATURE_BOUNDS["mtld"][1]


def _mtld_pass(tokens: list[str]) -> float:
    factors, types, count = 0.0, set(), 0
    for tok in tokens:
        count += 1
        types.add(tok)
        if len(types) / count <= MTLD_THRESHOLD:
            factors += 1.0
            types, count = set(), 0
    if count > 0:
        ttr = len(types) / count
        factors += (1.0 - ttr) / (1.0 - MTLD_THRESHOLD)
    if factors <= 0.0:
        return MTLD_CAP  # every token distinct: diversity beyond the scale
    return min(len(tokens) / factors, MTLD_CAP)


def mtld(tokens: list[str]) -> float:
    """Bidirectional MTLD, capped at the public bound."""
    if not tokens:
        return 0.0
    return 0.5 * (_mtld_pass(tokens) + _mtld_pass(tokens[::-1]))


@lru_cache(maxsize=200_000)
def syllables(word: str) -> int:
    import textstat

    return max(1, int(textstat.syllable_count(word)))


def fk_grade(n_words: int, n_sentences: int, n_syllables: int) -> float:
    if n_words == 0 or n_sentences == 0:
        return FEATURE_BOUNDS["fk_grade"][0]
    return 0.39 * n_words / n_sentences + 11.8 * n_syllables / n_words - 15.59


@dataclass(frozen=True)
class Sentence:
    words: tuple[str, ...]        # lower-cased word tokens
    checkable: tuple[str, ...]    # words eligible for the spelling check
    depth: int                    # maximum dependency depth, root = 1


class _Speller:
    def __init__(self) -> None:
        from spellchecker import SpellChecker

        self._known = SpellChecker().word_frequency
        self._cache: dict[str, bool] = {}

    def unknown(self, word: str) -> bool:
        if word not in self._cache:
            self._cache[word] = word not in self._known
        return self._cache[word]


class Featurizer:
    """Parse documents once and featurise them at any window size."""

    def __init__(self, n_process: int = 1, batch_size: int = 64) -> None:
        import spacy

        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        self.nlp.max_length = 2_000_000
        self.n_process = n_process
        self.batch_size = batch_size
        self._speller = _Speller()

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _depth(token) -> int:
        depth, stack = 0, [(token, 1)]
        while stack:
            tok, d = stack.pop()
            depth = max(depth, d)
            stack.extend((c, d + 1) for c in tok.children)
        return depth

    def _sentences_from_doc(self, doc, as_one: bool) -> list[Sentence]:
        spans = [doc[:]] if as_one else list(doc.sents)
        out = []
        for span in spans:
            words, checkable = [], []
            for tok in span:
                if tok.is_punct or tok.is_space or not any(ch.isalnum() for ch in tok.text):
                    continue
                w = tok.text.lower()
                words.append(w)
                if (tok.is_alpha and "'" not in tok.text and tok.pos_ != "PROPN"
                        and (tok.text.islower() or tok.is_sent_start)):
                    checkable.append(w)
            if not words:
                continue
            roots = [t for t in span if t.head == t] or [span.root]
            out.append(Sentence(tuple(words), tuple(checkable),
                                max(self._depth(r) for r in roots)))
        return out

    def parse(self, texts: list[str], spoken: bool) -> list[list[Sentence]]:
        """Sentences per document. For speech each line is one utterance."""
        if spoken:
            docs_lines = [t.split("\n") for t in texts]
            flat = [u for lines in docs_lines for u in lines]
            parsed = list(self.nlp.pipe(flat, n_process=self.n_process,
                                        batch_size=self.batch_size * 8))
            out, i = [], 0
            for lines in docs_lines:
                sents = []
                for _ in lines:
                    sents.extend(self._sentences_from_doc(parsed[i], as_one=True))
                    i += 1
                out.append(sents)
            return out
        return [self._sentences_from_doc(doc, as_one=False)
                for doc in self.nlp.pipe(texts, n_process=self.n_process,
                                         batch_size=self.batch_size)]

    # -- windows and features ---------------------------------------------
    @staticmethod
    def windows(sentences: list[Sentence], window_words: int) -> list[list[Sentence]]:
        """Group consecutive sentences into windows of at least `window_words`.

        A trailing remainder shorter than half a window is merged into the
        previous window, so no window is a fragment.
        """
        out, cur, n = [], [], 0
        for s in sentences:
            cur.append(s)
            n += len(s.words)
            if n >= window_words:
                out.append(cur)
                cur, n = [], 0
        if cur:
            if out and n < window_words / 2:
                out[-1].extend(cur)
            else:
                out.append(cur)
        return out

    def window_features(self, window: list[Sentence]) -> np.ndarray:
        words = [w for s in window for w in s.words]
        n_words, n_sent = len(words), len(window)
        n_syll = sum(syllables(w) for w in words if w.isalpha())
        checkable = [w for s in window for w in s.checkable]
        spell = (sum(self._speller.unknown(w) for w in checkable) / len(checkable)
                 if checkable else 0.0)
        vec = np.array([
            mtld(words),
            fk_grade(n_words, n_sent, n_syll),
            n_words / n_sent,
            float(np.mean([s.depth for s in window])),
            spell,
        ])
        return clip_features(vec)

    def featurise(self, docs: pd.DataFrame, window_words: int,
                  parsed: list[list[Sentence]] | None = None) -> pd.DataFrame:
        """One row per (document, window): the five features and the window's words."""
        if parsed is None:
            parsed = self.parse(docs["text"].tolist(), spoken=docs["modality"].iloc[0] == "spoken")
        rows = []
        for (_, doc), sents in zip(docs.iterrows(), parsed):
            for k, window in enumerate(self.windows(sents, window_words)):
                n_words = sum(len(s.words) for s in window)
                if n_words < 10:
                    continue
                f = self.window_features(window)
                rows.append({"corpus": doc["corpus"], "doc_id": doc["doc_id"],
                             "window": k, "n_words": n_words,
                             **dict(zip(LINGUISTIC_FEATURES, f)),
                             "text": " ".join(w for s in window for w in s.words)})
        return pd.DataFrame(rows)


def clip_features(x: np.ndarray) -> np.ndarray:
    lo = np.array([FEATURE_BOUNDS[f][0] for f in LINGUISTIC_FEATURES])
    hi = np.array([FEATURE_BOUNDS[f][1] for f in LINGUISTIC_FEATURES])
    return np.clip(x, lo, hi)


def to_unit(x: np.ndarray) -> np.ndarray:
    """Map features to [0, 1]^d with the public bounds (for the DP release)."""
    lo = np.array([FEATURE_BOUNDS[f][0] for f in LINGUISTIC_FEATURES])
    hi = np.array([FEATURE_BOUNDS[f][1] for f in LINGUISTIC_FEATURES])
    return (np.clip(x, lo, hi) - lo) / (hi - lo)


def from_unit(u: np.ndarray) -> np.ndarray:
    lo = np.array([FEATURE_BOUNDS[f][0] for f in LINGUISTIC_FEATURES])
    hi = np.array([FEATURE_BOUNDS[f][1] for f in LINGUISTIC_FEATURES])
    return lo + u * (hi - lo)
