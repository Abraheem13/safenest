# Citation audit

**Read this first.** I have no ability to retrieve or open any of these sources.
What follows is an assessment from background knowledge plus internal
consistency checks against the manuscript, and it is **not verification**. Every
entry marked anything other than "you must check" still needs your eyes on the
primary source. Reviewer 1 made citation integrity the stated grounds for
rejection, so a second unverified reference is fatal in a way that no other
defect in this paper is.

The rule I would apply: **if you cannot open the source and see the title, authors,
venue and year yourself, delete the citation and the claim it supports.** A
weaker paper with sound references survives review; a stronger one with invented
references does not.

---

## The eight the reviewer flagged

| # | Key | Entry as printed | Assessment |
|---|---|---|---|
| 3 | `Chen2025` | Y. Chen, L. Wang, et al. "AI dependence, cognitive fatigue, and critical thinking." *Education and Information Technologies*, 2025 | **Highest risk.** No volume, issue, pages or DOI; "et al." in a reference list; generic author names. The specific claim in your text (cognitive fatigue mediating, n=580 college students) is precise enough that it must be traceable to an exact paper. Either supply full bibliographic detail or cut the sentence. |
| 11 | `FAS2025` | Federation of American Scientists. "Ensuring child safety in the AI era: NIST AI RMF profiles for children," 2025 | Plausible as an FAS policy publication, but you need the exact title, author(s) and a URL with access date. Policy-organisation output is cited as grey literature with a URL, not as if it were a journal article. |
| 24 | `Dagreou2024` | M. Dagréou, P. Ablin, S. Vaiter, G. Peyré. "A framework for bilevel optimization." NeurIPS 37, 2024 | **Likely wrong venue/year.** These authors' well-known framework paper for bilevel optimization with stochastic and variance-reduction algorithms is NeurIPS **2022**, not NeurIPS 37 (2024). They have later bilevel work too. Check which paper you actually mean and correct the year and venue. |
| 26 | `Alhumaid2025` | A. Al-humaid, S. Alamro. "Behavioral biometrics for age estimation from touchscreen interactions." *IEEE Access*, 13, 2025 | Cannot assess. IEEE Access publishes at volume 13 in 2025, so the volume is at least consistent. Needs article number and DOI — IEEE Access articles always have both. |
| 27 | `Li2024` | J. Li, Y. Jiang, Y. Zhong, et al. "SocraticLM: Socratic personalized teaching with LLMs." NeurIPS 37, 2024 | SocraticLM at NeurIPS 2024 is plausible and I believe substantially correct. Verify the full author list and confirm the "outperforms GPT-4 by 12 percentage points" figure, since you quote it twice in the text — a misquoted effect size is as damaging as a bad reference. |
| 28 | `Ding2024` | Y. Ding, H. Hu, J. Zhou, et al. "KELE: A knowledge-enhanced framework for Socratic teaching." CIKM 2024 | Cannot assess. Confirm the venue — CIKM versus a workshop — and the exact expansion of the KELE acronym. |
| 31 | `KORA2025` | KORA Benchmark. "Evaluating child safety risks in AI systems." arXiv preprint, 2025 | **High risk.** "arXiv preprint" with no identifier is not a citation. Your text attributes a specific number to it (70.7% educational-integrity failure) and Table 1 gives it a full row. Supply the arXiv ID or remove the reference, the number, and the table row. |
| 34 | `Hsiao2024` | Y. Hsiao, N. Banerji, K. Nation. "A corpus-based developmental investigation of linguistic complexity." *Learning and Individual Differences*, 2024 | These authors do work on children's language and the Oxford Children's Corpus, so the reference is plausible. But note: the corpus is generally the **"Oxford Children's Corpus"**, not the "Oxford Children's Language Corpus" as you name it throughout. Fix the corpus name in the text, the Data Availability statement and the figure captions. Needs volume and DOI. |

## Others worth checking, not flagged by the reviewer

| Key | Concern |
|---|---|
| `Qi2025` | "Safety alignment should be more than a few tokens deep" — verify whether this is ICLR **2024** or **2025**. The arXiv posting and the conference year differ, and you cite it as ICLR 2025. |
| `Singh2025` | "G. Singh et al. Formal methods as a principled foundation for AI safety, ICML 2025" — a single initial and a bare "et al." for a position paper. Verify it exists and get the full author list; if it is a workshop paper, say so. |
| `Behrouz2025` | Nested Learning, NeurIPS 2025 — this is the paper your entire multi-timescale framing rests on. Confirm exact title, author list and venue; if it is very recent, make sure the version you cite is the published one. |
| `MetaAI2025` | LlamaFirewall — cite the arXiv identifier and authors, not "Meta AI" as a corporate author, if an author list exists. |
| `ChildSafe2025`, `SproutBench2025` | Both cited as benchmarks with arXiv IDs (2510.05484, 2508.11009). Confirm the IDs resolve and that the titles match; you attribute specific numbers to both. |
| `Kosmyna2025` | Cited as an "MIT Media Lab Technical Report". Give the report number and URL. You make a strong neuroscientific claim on it, so it must be locatable. |
| `Gerlich2025` | *Societies* 15(1):13, 2025 — MDPI journal, article-level detail looks well-formed. Verify the DOI. |
| `FTC2025`, `FTC2025b`, `California2025`, `EUAI2024`, `ICO2021` | Legal and regulatory sources. Cite with the official document number and a URL with access date. `EUAI2024` correctly gives Regulation (EU) 2024/1689. `FTC2025b` correctly gives 16 CFR part 312. |

## Likely sound (still confirm details)

`Piaget1952`, `Erikson1950`, `Vygotsky1978`, `Wood1976`, `Bai2022`, `Inan2023`,
`Rebedea2023`, `Dwork2006`, `McCarthy2010`, `Sanchez2019`, `Henrich2010`,
`Dalrymple2024`, `Xu2022`, `Kurian2024`, `Hoehl2024`, `OConnor2024`.

These are either canonical works I can identify with reasonable confidence, or
entries whose bibliographic detail is specific and well-formed. Confirm page
numbers and DOIs, but I would not expect problems.

---

## Process I would follow, in this order

1. **Open every one of the 37 references.** Not the abstract in a search result
   — the actual paper. Record the DOI or arXiv ID for each.
2. **Delete anything you cannot open.** Then delete the sentence it supported,
   and any table row or number that depended on it. This will hurt; do it anyway.
3. **Re-check every quoted figure against its source**: the 12-point SocraticLM
   margin, the 70.7% KORA failure rate, the 1,283 SproutBench prompts, the
   0.86–0.93 F1 range, the n=580 in Chen et al. Quoted numbers are checked by
   referees more often than reference formatting is.
4. **Fix the corpus name** — "Oxford Children's Corpus", not "Oxford Children's
   Language Corpus" — everywhere it appears.
5. **Add DOIs to every entry in `references.bib`.** A reference list where every
   entry carries a resolvable identifier is itself an argument that the audit
   was done.

If step 2 removes enough of Table 1 that the gap analysis weakens, that is a
true finding about the state of the literature, not a problem to be papered
over. Say in the text that the child-AI benchmark literature is very recent and
that you cite only what you could verify.
