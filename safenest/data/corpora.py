"""Loaders that turn each external corpus into one common document table.

Every loader returns a `pandas.DataFrame` with one row per child document and
these columns:

    corpus         registry key
    doc_id         unique within the corpus
    modality       "written" or "spoken"
    text           full text (written) or utterances joined by newlines (spoken)
    grade          US school grade, or NaN
    age_years      chronological age in years, or NaN
    admissible     developmental tiers consistent with the age or grade, as a
                   string such as "4" or "4,5"; boundary grades admit two tiers
    tier           the single tier when `admissible` has one element, else 0
    plus corpus-specific subgroup columns (ell, disability, econ, gender,
    race, impairment) where the source records them.

Tier bands follow `safenest.tiers`: t1 3-6, t2 7-9, t3 10-12, t4 13-15, t5 16-17.
A US student in grade g is aged g+5 at the start of the school year and g+6 at
its end, so grades 7 and 10 straddle a tier boundary and admit two tiers.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..tiers import tier_for_age
from .registry import RAW

GRADE_AGE_OFFSET = 5


def tiers_for_grade(grade: float) -> tuple[int, ...]:
    """Tiers consistent with a US grade, from the ages g+5 to g+6."""
    if grade is None or not np.isfinite(grade):
        return ()
    ages = (int(grade) + GRADE_AGE_OFFSET, int(grade) + GRADE_AGE_OFFSET + 1)
    tiers = sorted({int(tier_for_age(min(a, 17))) for a in ages})
    return tuple(tiers)


def _admissible(tiers: tuple[int, ...]) -> tuple[str, int]:
    label = ",".join(str(t) for t in tiers)
    return label, (tiers[0] if len(tiers) == 1 else 0)


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def text_key(text: str, prefix: int = 150) -> str:
    """Key used to detect the same essay appearing in two corpora."""
    return hashlib.md5(normalise_text(text)[:prefix].encode()).hexdigest()


# --------------------------------------------------------------- written
def load_persuade() -> pd.DataFrame:
    """PERSUADE 2.0, one row per essay.

    The released CSV repeats each essay once per discourse element; the
    essay-level table is cached as a parquet file on first use.
    """
    cache = RAW / "persuade" / "persuade_essays.parquet"
    if not cache.exists():
        raw = pd.read_csv(RAW / "persuade" / "persuade_corpus_2.0_train.csv", low_memory=False)
        cols = ["essay_id_comp", "full_text", "holistic_essay_score", "provider", "task",
                "prompt_name", "gender", "grade_level", "ell_status", "race_ethnicity",
                "economically_disadvantaged", "student_disability_status",
                "essay_word_count"]
        raw[cols].drop_duplicates("essay_id_comp").to_parquet(cache, index=False)
    e = pd.read_parquet(cache)
    rows = []
    for r in e.itertuples(index=False):
        tiers = tiers_for_grade(r.grade_level)
        adm, tier = _admissible(tiers) if tiers else ("", -1)
        rows.append({
            "corpus": "persuade", "doc_id": r.essay_id_comp, "modality": "written",
            "text": r.full_text, "grade": r.grade_level, "age_years": np.nan,
            "admissible": adm, "tier": tier,
            "ell": {"Yes": 1, "No": 0}.get(str(r.ell_status).strip(), np.nan),
            "disability": {"Identified as having disability": 1,
                           "Not identified as having disability": 0}.get(
                               r.student_disability_status, np.nan),
            "econ": {"Economically disadvantaged": 1,
                     "Not economically disadvantaged": 0}.get(
                         r.economically_disadvantaged, np.nan),
            "gender": r.gender, "race": r.race_ethnicity,
            "impairment": np.nan, "prompt": r.prompt_name,
        })
    return pd.DataFrame(rows)


def load_ellipse() -> pd.DataFrame:
    """ELLIPSE public training split; every writer is an English learner."""
    e = pd.read_csv(RAW / "ellipse" / "ELLIPSE_Final_github_train.csv")
    rows = []
    for r in e.itertuples(index=False):
        adm, tier = _admissible(tiers_for_grade(r.grade))
        rows.append({
            "corpus": "ellipse", "doc_id": r.text_id_kaggle, "modality": "written",
            "text": r.full_text, "grade": float(r.grade), "age_years": np.nan,
            "admissible": adm, "tier": tier, "ell": 1, "disability": np.nan,
            "econ": {"Economically disadvantaged": 1,
                     "Not economically disadvantaged": 0}.get(r.SES, np.nan),
            "gender": {"Male": "M", "Female": "F"}.get(r.gender, r.gender),
            "race": r.race_ethnicity, "impairment": np.nan, "prompt": r.prompt,
        })
    return pd.DataFrame(rows)


#: Grade of each ASAP prompt, from the competition's data description.
ASAP_GRADE: dict[int, int] = {1: 8, 2: 10, 3: 10, 4: 10, 5: 8, 6: 10, 7: 7, 8: 10}
_ASAP_FILES = ("asappp-1-2-original", "asappp-3-6-original", "asap-7-original",
               "asap-8-original")
#: ASAP replaces named entities with tags such as @PERSON1; a neutral
#: dictionary word keeps the syntax intact without inventing content.
_ASAP_TAG = re.compile(r"@[A-Z]+\d*")


def load_asap() -> pd.DataFrame:
    frames = [pd.read_parquet(RAW / "asap" / f"{name}.parquet")[["essay_set", "essay"]]
              for name in _ASAP_FILES]
    a = pd.concat(frames, ignore_index=True)
    rows = []
    for i, r in enumerate(a.itertuples(index=False)):
        grade = ASAP_GRADE[int(r.essay_set)]
        adm, tier = _admissible(tiers_for_grade(grade))
        rows.append({
            "corpus": "asap", "doc_id": f"asap-{i:05d}", "modality": "written",
            "text": _ASAP_TAG.sub("something", r.essay), "grade": float(grade),
            "age_years": np.nan, "admissible": adm, "tier": tier, "ell": np.nan,
            "disability": np.nan, "econ": np.nan, "gender": np.nan, "race": np.nan,
            "impairment": np.nan, "prompt": f"set{int(r.essay_set)}",
        })
    return pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)


# ---------------------------------------------------------------- spoken
_CHAT_AGE = re.compile(r"(\d+);(\d{1,2})")


def _chat_files(zip_path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(zip_path) as z:
        return [(n, z.read(n).decode("utf-8", errors="replace"))
                for n in sorted(z.namelist()) if n.endswith(".cha")]


def _child_utterances(chat: str) -> tuple[list[str], float | None]:
    """Target-child utterances and age from one CHAT transcript.

    Utterances are cleaned of CHAT annotation: retraced material, bracketed
    codes, fillers (&-), untranscribed material (xxx, yyy, www), form markers
    (@c, @b) and terminators, leaving the words the child finally produced.
    """
    age = None
    for line in chat.splitlines():
        if line.startswith("@ID:") and "|CHI|" in line:
            m = _CHAT_AGE.search(line.split("|CHI|", 1)[1])
            if m:
                age = int(m.group(1)) + int(m.group(2)) / 12.0
    utts = []
    for line in chat.replace("\n\t", " ").splitlines():
        if not line.startswith("*CHI:"):
            continue
        u = line[5:]
        u = re.sub(r"\x15[^\x15]*\x15", " ", u)            # media bullets
        # Retraced material (repetitions and self-corrections marked [/], [//],
        # [///] or [/-]) is removed, as CHILDES analyses conventionally do.
        u = re.sub(r"<[^>]*>\s*\[/+-?\]", " ", u)
        u = re.sub(r"\S+\s*\[/+-?\]", " ", u)
        u = re.sub(r"\[[^\]]*\]", " ", u)                   # [: x], [*], [+ gram] ...
        u = re.sub(r"<([^>]*)>", r"\1", u)
        u = re.sub(r"&[-=+~]?\S+", " ", u)                  # fillers, gestures
        u = re.sub(r"\b(xxx|yyy|www)\b", " ", u)
        u = re.sub(r"@\S+", "", u)                          # form markers
        u = re.sub(r"\(([a-z']*)\)", r"\1", u)              # omitted sounds
        u = re.sub(r"[+/.,!?\"()]+", " ", u)
        u = re.sub(r"\s+", " ", u).strip()
        if u:
            utts.append(u)
    return utts, age


def load_childes(name: str) -> pd.DataFrame:
    """Gillam or ENNI transcripts, one row per child.

    Language-impairment status is read from the corpus folder structure
    (SLI or TD folders), recorded as `impairment` = 1 or 0.
    """
    zip_path = RAW / "childes" / f"{name.capitalize() if name != 'enni' else 'ENNI'}.zip"
    rows = []
    for path, chat in _chat_files(zip_path):
        utts, age = _child_utterances(chat)
        if not utts or age is None:
            continue
        parts = path.lower().split("/")
        impaired = 1 if any(p in ("sli", "li", "impaired", "dld") for p in parts) else (
            0 if any(p in ("td", "typical", "control") for p in parts) else np.nan)
        years = int(np.floor(age))
        if not 3 <= years <= 17:
            continue
        t = int(tier_for_age(years))
        rows.append({
            "corpus": name, "doc_id": path, "modality": "spoken",
            "text": "\n".join(utts), "grade": np.nan, "age_years": age,
            "admissible": str(t), "tier": t, "ell": np.nan, "disability": np.nan,
            "econ": np.nan, "gender": np.nan, "race": np.nan, "impairment": impaired,
            "prompt": parts[-2] if len(parts) > 1 else "",
        })
    return pd.DataFrame(rows)


LOADERS = {
    "persuade": load_persuade,
    "ellipse": load_ellipse,
    "asap": load_asap,
    "gillam": lambda: load_childes("gillam"),
    "enni": lambda: load_childes("enni"),
}


def available() -> list[str]:
    """Corpora whose raw files are present locally."""
    present = {
        "persuade": (RAW / "persuade" / "persuade_essays.parquet").exists()
        or (RAW / "persuade" / "persuade_corpus_2.0_train.csv").exists(),
        "ellipse": (RAW / "ellipse" / "ELLIPSE_Final_github_train.csv").exists(),
        "asap": all((RAW / "asap" / f"{n}.parquet").exists() for n in _ASAP_FILES),
        "gillam": (RAW / "childes" / "Gillam.zip").exists(),
        "enni": (RAW / "childes" / "ENNI.zip").exists(),
    }
    return [k for k, v in present.items() if v]


def load(name: str) -> pd.DataFrame:
    return LOADERS[name]()


def drop_overlap(df: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove documents of `df` that also occur in `reference`.

    ELLIPSE shares several hundred essays with PERSUADE, so any evaluation that
    trains on one and tests on the other must remove them first.
    """
    ref = set(reference["text"].map(text_key))
    keep = ~df["text"].map(text_key).isin(ref)
    return df[keep].reset_index(drop=True), int((~keep).sum())
