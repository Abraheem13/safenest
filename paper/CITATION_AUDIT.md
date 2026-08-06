# Citation audit

Two parts: **structural defects I verified by inspecting `references.bib`**, and
**content concerns I can only flag**.

**On the second part, be clear about what it is worth.** I have no ability to
retrieve or open any source. Nothing below constitutes verification of whether a
paper exists, who wrote it, or what it found. Reviewer 1 made citation integrity
the stated grounds for rejection, so a single unverified entry surviving into the
resubmission is fatal in a way that no other defect in this paper is.

The rule to apply: **if you cannot open the source and see its title, authors,
venue and year yourself, delete the citation and the claim it supports.**

---

## Part 1 — Structural defects, verified in `references.bib`

These I checked directly against the file. All are real and all are fixable
today.

### 1. Not one entry in 39 carries a DOI or URL

`grep -ci doi references.bib` returns 0. For a submission whose last review
turned on citation integrity, a reference list where no entry carries a
resolvable identifier is the worst possible signal. Every entry needs a DOI, an
arXiv ID, or a URL with an access date. Fixing this alone changes how the list
reads.

### 2. Thirteen entries use `and others` instead of a full author list

Affected: `Bai2022`, `Chen2025`, `Dalrymple2024`, `Ding2024`, `Inan2023`,
`Kosmyna2025`, `Li2024`, `OConnor2024`, `Qi2025`, `Rebedea2023`, `Sanchez2019`,
`Singh2025`, `Xu2022`.

`and others` renders as "et al." in the bibliography. That is acceptable in
running text, never in a reference list — this is exactly what the reviewer meant
by "incomplete". Every one needs its full author list.

### 3. Three entries name a benchmark as the author

```
ChildSafe2025   author = {{ChildSafe Benchmark}}
KORA2025        author = {{KORA Benchmark}}
SproutBench2025 author = {{SproutBench}}
```

A benchmark is not an author; papers introducing benchmarks have people who wrote
them. This pattern is the signature of a placeholder entry that was never filled
in, and all three support load-bearing claims in your text and occupy rows in
Table 1. Supply real author lists or remove all three, along with the numbers you
attribute to them (the 70.7% failure rate, the 1,283 prompts, the nine safety
dimensions).

### 4. `KORA2025` has no identifier at all

```
journal = {arXiv preprint}
```

That is not a citation — there is no way for a reader to find it. Either supply
the arXiv ID or cut the reference, the 70.7% figure and the Table 1 row.

### 5. NeurIPS volume numbers are internally inconsistent

```
Li2024      booktitle = {NeurIPS 37}   year = 2024
Dagreou2024 booktitle = {NeurIPS 37}   year = 2024
Behrouz2025 booktitle = {NeurIPS 38}   year = 2025
```

Two different years cannot both be NeurIPS 37, so at least one is wrong on its
own terms. NeurIPS 2023 was the 37th conference, 2024 the 38th, 2025 the 39th —
which would make all three of these off by one. Simplest fix: drop the ordinal
entirely and write `booktitle = {Advances in Neural Information Processing
Systems (NeurIPS)}` with the year, which is the conventional form anyway.

### 6. arXiv identifiers are in the `journal` field

`Bai2022`, `Inan2023`, `Dalrymple2024`, `ChildSafe2025`, `SproutBench2025`,
`Macina2023` all set `journal = {arXiv:...}`. This renders as though arXiv were a
journal. Use `@misc` with `eprint`, `archivePrefix = {arXiv}` and `primaryClass`,
or `howpublished`.

### 7. Several titles are truncated

- `Dwork2006` — "Calibrating noise to sensitivity" is missing "in Private Data Analysis".
- `Inan2023` — "Llama Guard: LLM-based safeguard" appears to be a shortened form of the full subtitle.
- `Rebedea2023` — "NeMo Guardrails" alone is not the paper's title.
- `Wood1976` — journal abbreviated to "J. Child Psych. Psychiatry"; write it out.

Check each against the source and restore the full title.

### 8. Missing volume, issue, pages or article number

`Alhumaid2025` (IEEE Access always assigns an article number), `Chen2025`
(nothing beyond the journal name), `Hsiao2024` (no volume or pages),
`OConnor2024` (nothing), `Hoehl2024` (no pages).

### 9. Two entries are defined but never cited

`Macina2023` and `UNICEF2021`. Harmless — BibTeX omits uncited entries — but
worth either citing or deleting so the file reflects the paper.

---

## Part 2 — Content concerns I can only flag, not verify

| Key | Concern |
|---|---|
| `Chen2025` | Highest risk. No volume, pages or DOI; generic author names; yet your text attributes a precise finding (cognitive fatigue as mediator, n = 580 college students). Either supply full detail or cut the sentence. |
| `Dagreou2024` | These authors' well-known bilevel optimisation framework paper is, to my recollection, NeurIPS **2022**, not 2024. Check which paper you mean. |
| `Hsiao2024` | Plausible — these authors do work on children's language. But note the corpus is generally the **"Oxford Children's Corpus"**, not the "Oxford Children's Language Corpus" as you call it throughout the text, the Data Availability statement and the figure captions. Fix the name everywhere. |
| `Gerlich2025` | *Societies* 15(1):13 — check the article number. My recollection of the 2025 Gerlich paper on cognitive offloading and critical thinking puts it at a different article number in the same volume. |
| `Qi2025` | Verify whether this is ICLR **2024** or **2025**; the arXiv posting and conference years differ. |
| `Singh2025` | A single initial and bare "et al." for a position paper. Verify it exists; if it is a workshop paper, say so. |
| `Behrouz2025` | Your entire multi-timescale framing rests on this. Confirm title, authors and venue, and cite the published version if one now exists. |
| `Kosmyna2025` | Cited as an "MIT Media Lab Technical Report" with no number or URL. You make a strong neuroscientific claim on it, so it must be locatable. |
| `MetaAI2025` | Cite the arXiv ID and author list rather than "Meta AI" as a corporate author, if an author list exists. |
| `FAS2025` | Plausible as FAS policy output, but cite as grey literature: exact title, author, URL, access date. |
| `Alhumaid2025`, `Ding2024`, `Li2024` | Cannot assess. For `Li2024` also re-check the "outperforms GPT-4 by 12 percentage points" figure, which you quote twice. |

### Likely sound, but still confirm details

`Piaget1952`, `Erikson1950`, `Vygotsky1978`, `Wood1976`, `Bai2022`, `Inan2023`,
`Rebedea2023`, `Dwork2006`, `McCarthy2010`, `Sanchez2019`, `Henrich2010`,
`Dalrymple2024`, `Xu2022`, `Kurian2024`, `Hoehl2024`, `OConnor2024`,
`FTC2025`, `FTC2025b`, `California2025`, `EUAI2024`, `ICO2021`.

Canonical works or entries whose detail is specific and well-formed. `EUAI2024`
correctly gives Regulation (EU) 2024/1689 and `FTC2025b` correctly gives 16 CFR
Part 312.

---

## Order of work

1. **Open all 37 cited references.** Not a search-result abstract — the source.
   Record a DOI or arXiv ID for each.
2. **Delete what you cannot open**, together with the sentence it supported and
   any table row or number depending on it. This will hurt. Do it anyway.
3. **Fix the structural defects in Part 1.** They are mechanical and take an
   afternoon: full author lists, DOIs, correct venue forms, restored titles.
4. **Re-check every quoted number against its source**: SocraticLM's 12 points,
   KORA's 70.7%, SproutBench's 1,283 prompts, the 0.86--0.93 F1 range, Chen's
   n = 580. Referees check quoted figures more often than they check formatting.
5. **Fix "Oxford Children's Language Corpus" → "Oxford Children's Corpus"** in
   the text, captions and Data Availability statement.

If step 2 thins Table 1 enough to weaken the gap analysis, that is a true finding
about a very young literature, not a problem to conceal. Say in the text that you
cite only what you could verify.
