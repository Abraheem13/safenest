# External data

The real-data experiments (13 to 16) use five public corpora of child-produced
language. None of them is redistributed here: each has its own licence, and
`data/raw/` and `data/processed/` are excluded from version control.

| Corpus | Language | Ages or grades | Size used | Licence | Obtain |
|---|---|---|---|---|---|
| PERSUADE 2.0 (training release) | written essays | grades 6–12 | 15,594 essays | CC BY-NC-SA 4.0 | `scripts/download_data.py persuade` |
| ELLIPSE (public training split) | written essays, English learners | grades 8–12 | 3,911 essays | CC BY-NC-SA 4.0 | `scripts/download_data.py ellipse` |
| ASAP (Automated Student Assessment Prize) | written essays | grades 7, 8, 10 | 12,972 essays | Kaggle competition terms | `scripts/download_data.py asap` |
| CHILDES Gillam | transcribed oral narratives | ages 5;0–11;11 | 668 transcripts | CC BY-NC-SA 4.0, TalkBank Ground Rules | TalkBank login |
| CHILDES ENNI | transcribed oral narratives | ages 4–9 | 361 transcripts | CC BY-NC-SA 4.0, TalkBank Ground Rules | TalkBank login |

`scripts/download_data.py` fetches the first three and checks each file against
the SHA-256 recorded in `safenest/data/registry.py`. The recorded hash is that of
the file the published results were computed from. A mismatch means the source
has changed.

The CHILDES corpora need a free TalkBank account, so they cannot be scripted:

1. Sign in at <https://talkbank.org>.
2. Download the transcripts from
   <https://talkbank.org/childes/access/Clinical-Eng/Gillam.html> and
   <https://talkbank.org/childes/access/Clinical-Eng/ENNI.html>.
3. Place `Gillam.zip` and `ENNI.zip` in `data/raw/childes/`.

`python3 scripts/download_data.py gillam enni` then checks both archives against
the recorded SHA-256. TalkBank revises transcripts from time to time, so a later
download may differ from the one the published results used. Transcripts
without a recorded child age are skipped (four in Gillam, one in ENNI).

Then build the feature tables:

```bash
python3 scripts/prepare_features.py        # parse every corpus present
python3 scripts/embed_windows.py written 50  # sentence embeddings for Experiment 13
                                              # (Experiment 16 embeds speech itself)
```

## What is derived, and how

`safenest/data/corpora.py` turns each corpus into one document table. Grades
are mapped to developmental tiers through the ages a student in that grade
has during the school year (grade *g* spans ages *g*+5 to *g*+6). Grades 7 and 10
straddle a tier boundary and are scored as correct on either adjacent tier.

A few decisions affect the numbers:

- 454 ELLIPSE essays also appear in PERSUADE. They are removed from ELLIPSE
  before any split, so no essay is ever on both sides of one.
- ASAP replaces named entities with tags such as `@PERSON1`. The tags are
  replaced by a neutral word before parsing.
- The password-protected ELLIPSE test split and the password-protected PERSUADE
  test release are not used.
- In PERSUADE every prompt was set to a single grade, so a model can learn the
  topic in place of the age. Experiment 13 measures this by holding out whole
  prompts, and its headline results are leave-one-corpus-out.

## Citations

If you use these data, cite the original sources:

- Crossley, S.A.; Tian, Y.; Baffour, P.; Franklin, A.; Benner, M.; Boser, U.
  A large-scale corpus for assessing written argumentation: PERSUADE 2.0.
  *Assessing Writing* 2024, 61, 100865. https://doi.org/10.1016/j.asw.2024.100865
- Crossley, S.A.; Tian, Y.; Baffour, P.; Franklin, A.; Kim, Y.; Morris, W.;
  Benner, M.; Picou, A.; Boser, U. The English Language Learner Insight,
  Proficiency and Skills Evaluation (ELLIPSE) Corpus. *International Journal of
  Learner Corpus Research* 2023, 9, 248–269. https://doi.org/10.1075/ijlcr.22026.cro
- The Hewlett Foundation. Automated Student Assessment Prize: Automated Essay
  Scoring. Kaggle, 2012. https://www.kaggle.com/competitions/asap-aes
- MacWhinney, B. *The CHILDES Project: Tools for Analyzing Talk*, 3rd ed.;
  Lawrence Erlbaum Associates, 2000. https://doi.org/10.4324/9781315805672
- Gillam corpus: https://doi.org/10.21415/T5QS3N. ENNI corpus:
  https://doi.org/10.21415/T51G7V.
