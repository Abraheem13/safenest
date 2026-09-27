"""Registry of the external corpora used for the real-data experiments.

Nothing listed here is redistributed with the repository. Each entry records
where the data comes from, its licence, the reference to cite and, where the
file is fetched by `scripts/download_data.py`, the SHA-256 of the file that the
reported results were computed from.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    path: str          # relative to data/raw
    sha256: str | None  # None when the file is obtained manually


@dataclass(frozen=True)
class Corpus:
    name: str
    description: str
    modality: str      # "written" or "spoken"
    licence: str
    citation: str
    homepage: str
    files: tuple[RemoteFile, ...]
    manual: str | None = None  # instructions when a login is required


CORPORA: dict[str, Corpus] = {
    "persuade": Corpus(
        name="PERSUADE 2.0",
        description="Argumentative essays by US students in grades 6-12, with "
                    "grade, English-learner, disability, economic and "
                    "demographic fields (training release, 15,594 essays).",
        modality="written",
        licence="CC BY-NC-SA 4.0",
        citation="Crossley, S.A.; Tian, Y.; Baffour, P.; Franklin, A.; Benner, M.; "
                 "Boser, U. A large-scale corpus for assessing written argumentation: "
                 "PERSUADE 2.0. Assessing Writing 2024, 61, 100865. "
                 "https://doi.org/10.1016/j.asw.2024.100865",
        homepage="https://github.com/scrosseye/persuade_corpus_2.0",
        files=(
            RemoteFile(
                url="https://drive.usercontent.google.com/download?"
                    "id=13phHyDzIsb0MHyJr6q-B-qIa9P2tM135&export=download&confirm=t",
                path="persuade/persuade_corpus_2.0_train.csv",
                sha256="f61319edd8bf16a982711ea0399fad59c05afaec05cdf0767f16a2c05c467e23",
            ),
        ),
    ),
    "ellipse": Corpus(
        name="ELLIPSE",
        description="Essays by English-language learners in grades 8-12 "
                    "(public training split, 3,911 essays).",
        modality="written",
        licence="CC BY-NC-SA 4.0",
        citation="Crossley, S.A.; Tian, Y.; Baffour, P.; Franklin, A.; Kim, Y.; "
                 "Morris, W.; Benner, M.; Picou, A.; Boser, U. The English Language "
                 "Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) Corpus. "
                 "International Journal of Learner Corpus Research 2023, 9, 248-269. "
                 "https://doi.org/10.1075/ijlcr.22026.cro",
        homepage="https://github.com/scrosseye/ELLIPSE-Corpus",
        files=(
            RemoteFile(
                url="https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/"
                    "ELLIPSE_Final_github_train.csv",
                path="ellipse/ELLIPSE_Final_github_train.csv",
                sha256="782344e99668a3ff508d7410c0eb6e36da70f3b28f81c96e367f1ca04924b06c",
            ),
        ),
    ),
    "asap": Corpus(
        name="ASAP",
        description="Automated Student Assessment Prize essays, eight prompts "
                    "written in grades 7, 8 and 10 (12,976 essays, 4 duplicates removed).",
        modality="written",
        licence="Research use, per the original Kaggle competition terms",
        citation="The Hewlett Foundation. Automated Student Assessment Prize: Automated "
                 "Essay Scoring. Kaggle, 2012.",
        homepage="https://www.kaggle.com/competitions/asap-aes",
        files=tuple(
            RemoteFile(
                url=f"https://huggingface.co/datasets/llm-aes/{name}/resolve/main/"
                    "data/train-00000-of-00001.parquet",
                path=f"asap/{name}.parquet",
                sha256=sha,
            )
            for name, sha in (
                ("asappp-1-2-original",
                 "f1d32c572909fa7a1ff0cd2302734f8d1abfeedf7df136adada55e7b579f3a54"),
                ("asappp-3-6-original",
                 "7035b596f106822e4d52ae5389fe8185a5db5940144ae9052843488e67be0d52"),
                ("asap-7-original",
                 "717027b5222de8ec27bf0bdcfb51665ba3584082360c0b2241fd8a1b61a16416"),
                ("asap-8-original",
                 "62f79b7de8a494b83ab831dbbb54c07e785794d6be3cc7c17a3c4f403a472899"),
            )
        ),
    ),
    "gillam": Corpus(
        name="CHILDES Gillam",
        description="Oral narratives from the norming of the Test of Narrative "
                    "Language, ages 5;0-11;11, typically developing children and "
                    "children with language impairment.",
        modality="spoken",
        licence="TalkBank Ground Rules (CC BY-NC-SA 3.0)",
        citation="Gillam, R.B.; Pearson, N. Test of Narrative Language; Pro-Ed: "
                 "Austin, TX, USA, 2004. Data: doi:10.21415/T5QS3N.",
        homepage="https://talkbank.org/childes/access/Clinical-Eng/Gillam.html",
        files=(RemoteFile(url="", path="childes/Gillam.zip", sha256=None),),
        manual="Log in at talkbank.org, download the Gillam transcripts and place "
               "Gillam.zip in data/raw/childes/.",
    ),
    "enni": Corpus(
        name="CHILDES ENNI",
        description="Edmonton Narrative Norms Instrument story retellings, ages "
                    "4-9, typically developing children and children with "
                    "language impairment.",
        modality="spoken",
        licence="TalkBank Ground Rules (CC BY-NC-SA 3.0)",
        citation="Schneider, P.; Hayward, D.; Dub\u00e9, R.V. Storytelling from pictures "
                 "using the Edmonton Narrative Norms Instrument. Journal of "
                 "Speech-Language Pathology and Audiology 2006, 30, 224-238. "
                 "Data: doi:10.21415/T51G7V.",
        homepage="https://talkbank.org/childes/access/Clinical-Eng/ENNI.html",
        files=(RemoteFile(url="", path="childes/ENNI.zip", sha256=None),),
        manual="Log in at talkbank.org, download the ENNI transcripts and place "
               "ENNI.zip in data/raw/childes/.",
    ),
}


def raw_path(rel: str) -> Path:
    return RAW / rel
