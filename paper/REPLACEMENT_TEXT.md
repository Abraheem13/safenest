# Replacement text for the manuscript

Drafts, not finished prose — edit into your own voice. Every number is from
`results/*.json`. Tables are in `tables.tex` (regenerate with
`python3 experiments/run_all.py && python3 experiments/make_latex.py`).

---

## 1. Abstract — replace the final two sentences

The current text reads:

> Computational validation via Monte Carlo simulation confirms 98.6% classification
> accuracy at the adopted privacy level, and comparative evaluation across 7,000
> synthetic prompts demonstrates that NPL achieves 89.4% means developmental safety
> rate versus 48.1% for the best existing baseline a +41.2-percentage point
> improvement while reducing under-protection from 46.0% to 5.3%.

Replace with:

> Monte Carlo simulation using synthetic signals parameterised from published
> developmental linguistics corpora (CHILDES-db; Oxford Children's Language Corpus)
> yields 98.8% tier-classification accuracy under ε = 1.0 differential privacy,
> where the privacy unit is one child in the calibration corpus. A comparative
> policy-decision simulation over 7,000 prompts, scored against an independent
> rubric derived from chronological age, regulatory instruments and Piagetian
> stage criteria, gives NPL a mean developmental safety rate of 75.5% against
> 52.5% for the strongest age-agnostic baseline (+23.0 percentage points,
> McNemar p < 10⁻²⁶⁰) and 71.5% for a strong age-conditioned baseline
> (+4.0 points, p < 10⁻⁹), while reducing under-protection from 39.0% to 3.9%.
> Empirical validation with human subjects remains a necessary next step.

**Why each change.** "confirms" → "yields" (simulation does not confirm); the
privacy unit is now stated, because at ε = 1.0 the claim is false under the other
reading; +41.2 → +21.3 because the original ground truth was the specification
being evaluated; the age-conditioned comparison is added because omitting it
invites the reviewer to compute it themselves and find +2.2.

---

## 2. Section 3.3 — new paragraph on the privacy model

Insert immediately after Equation 8.

> The differential-privacy guarantee requires four elements to be well-defined.
> The **privacy unit** is one child in the calibration corpora. Two corpora are
> **adjacent** if one is obtained from the other by adding or removing all
> transcripts belonging to a single child. The **protected output** is the
> released vector of tier-conditional likelihood parameters — the per-tier means
> and covariances of Equation 7 — computed offline by the Gaussian mechanism with
> σ = Δ₂ √(2 ln(1.25/δ)) / ε at δ = 10⁻⁵. Because a parameter estimated over N
> children has per-child sensitivity Δ₂/N, the perturbation of the released
> estimate is negligible for corpora of the size used here, which is why the
> guarantee costs almost no accuracy (Table 12). Everything computed downstream —
> the posterior, the tier assignment, the emitted response — is a post-processing
> function of the released parameters and inherits the same guarantee.
>
> The live user's own signals are **not** protected by this ε. They are protected
> architecturally: each signal source is a separate enclave that emits only a
> clipped log-likelihood-ratio vector, raw features never cross that boundary,
> and the ratios are discarded at session end. This is data minimisation in the
> sense of the COPPA amendments, and we do not describe it as differential
> privacy.
>
> We report the alternative reading explicitly, because it is the one a reader
> may assume. If the privacy unit were instead a single interaction of the live
> user — local differential privacy — then the signal-to-noise ratio of one
> released log-likelihood-ratio vector is ε_m / (2(K−1)), independent of the
> clipping bound C, since tightening the clip attenuates signal and noise
> equally. At ε = 1.0, K = 5 and the budget allocation of Table 3 this ratio is
> 0.05, and ten interactions yield 37.2% five-way accuracy against a 20% chance
> baseline (Table 12); the cumulative loss across such a session is ε = 10 under
> basic composition. Accurate developmental inference and local differential
> privacy with respect to the inferred user are therefore not simultaneously
> achievable at this budget. We regard this as a structural property of the
> problem rather than a limitation of the estimator.

---

## 3. Section 3.5 — Equation 15 and the metacognition term

The published Equation 15 rewards only knowledge gain, under which VerifyRequest
is dominated at every tier and its usage is flat (0, 0, 4, 4, 0 across t1–t5).
Either delete the monotonicity claim or add the term. To add it, amend Equation 15:

> R(s,a) = α_learn (q′ − q) − α_reveal · 1[a reveals answer]
>          − α_frust · 1[stagnation] + α_meta · m(t_k) · 1[a = VerifyRequest]

and add:

> where m(t_k) ∈ [0, 1] is the metacognitive capacity of tier t_k, taken as
> (0, 0.15, 0.45, 0.75, 1.0) for t1–t5 following the Piagetian progression from
> concrete to formal operations: reliable reflection on one's own reasoning is
> not reliably available before concrete operations and matures through
> adolescence. Without this term the reward credits only knowledge gain, and
> requests for verification — which by construction produce little immediate
> gain — are dominated at every tier. Reward weights are α_learn = 1.0,
> α_reveal = 0.60, α_frust = 0.25, α_meta = 0.06, γ = 0.95, H = 5,
> σ_ZPD = 0.12, with q discretised into 50 points.
>
> The condition α_reveal > α_learn · max_a,k δ(a, t_k) = 0.40 is sufficient but
> not tight: the continuation value suppresses premature revelation at penalties
> below it. Across a sweep of 80 configurations spanning α_reveal ∈ [0.45, 1.5],
> α_frust ∈ [0, 0.5] and γ ∈ [0.85, 0.99], PartialExplain was never the optimal
> opening action in any configuration, while monotonicity of VerifyRequest usage
> held in 81%.

Note the honest 81% — do not round it up. A reviewer who reruns the sweep will
see it.

---

## 4. Section 5 — circularity of the comparative evaluation

Insert before the DSR results, and define DSR here (it is currently used before
it is defined).

> **Developmental Safety Rate.** Let P be a set of evaluation prompts, g(p) ∈
> {allow, scaffold, block} the ground-truth appropriate handling of prompt p, and
> f(p) the handling produced by the framework under test. Then
> DSR(P) = |P|⁻¹ Σ_p 1[f(p) = g(p)]. The complement is partitioned into
> under-protection, Σ_p 1[severity(f(p)) < severity(g(p))], and over-restriction,
> Σ_p 1[severity(f(p)) > severity(g(p))], with severity(allow) < severity(scaffold)
> < severity(block). A scaffolded response where the ground truth is "allow"
> counts as over-restriction rather than success: withholding a direct answer
> from a learner entitled to one is a real cost, and folding it into the success
> rate would conceal it.
>
> **Two ground truths.** Deriving ground-truth labels from the feature-gating
> matrix of Table 7 would make the evaluation circular, since that matrix is part
> of the specification under test; any faithful implementation would score near
> 100% by construction. We therefore report a second, independent labelling
> derived from sources external to the architecture: chronological age, the
> thresholds set by the regulatory instruments (COPPA's under-13 boundary,
> California SB 243's crisis-detection requirement, the EU AI Act's Article 5
> prohibition, the ICO Age-Appropriate Design Code's default-protection
> principle), and Piagetian criteria on abstraction demand. This labeller does
> not consult the feature-gating matrix. The two labellings agree on 77.1% of the
> 7,000 prompts (Cohen's κ = 0.645), so the choice materially affects the result:
> NPL's mean DSR is 86.3% under the circular labelling and 75.5% under the
> independent one. We treat the latter as the headline figure and report the
> former only for contrast.
>
> One qualification on the independence of the rubric labeller. During
> development the rubric identified a case in which the specification permitted
> direct answers to open-ended questions from 7-to-9-year-olds, and the
> specification was subsequently amended to scaffold direct requests below t3
> (Section 3.4). The rubric therefore informed one design decision, and to that
> extent it is no longer a fully held-out standard. We report the affected
> figure both ways: NPL's mean DSR against the rubric is 75.5% after the
> amendment and 73.8% before it, so the amendment accounts for 1.7 points of the
> 23.0-point margin over the strongest age-agnostic baseline. The conclusion is
> unchanged under either value. A genuinely held-out evaluation — prompts labelled
> by developmental psychologists who have not seen the specification — remains
> necessary and is the first item of the human-subject protocol.

---

## 5. Section 6 — three new limitation paragraphs

### Neurodivergent and atypically developing users

> The five-tier mapping infers developmental protection largely from linguistic
> sophistication, and that inference fails precisely where linguistic profile and
> protective need diverge. In simulation, a cohort whose linguistic features are
> shifted two tiers above their chronological band — the profile of a highly
> verbal 8-year-old, including many autistic children — is assigned above its
> true tier in 82.0% of sessions, with 62.3% of assignments less restrictive than
> warranted. This is the framework's most consequential failure mode, because it
> concentrates under-protection in a population plausibly at elevated risk in
> unsupervised interaction.
>
> We therefore add a discordance check to the estimator: when an external
> attestation (parental, school-issued device certificate, or verified account)
> indicates a tier that the linguistic estimate contradicts by two tiers or more,
> the attested tier governs and the assignment never exceeds it. This reduces
> over-promotion of the verbally advanced cohort from 82.0% to 3.0% of sessions.
> It is a partial remedy in two respects: it requires an attestation to be
> present, and it does not fire on one-tier mismatches, so a cohort combining a
> one-tier linguistic advance with atypical motor profiles is left with 59.2%
> under-protection. A continuous developmental representation, or a separate
> profile-discordance tier, would address the residue; we regard this as the most
> important open problem in the specification.

### Cross-cultural and multilingual deployment

> The tier-conditional likelihoods are parameterised from CHILDES-db and the
> Oxford Children's Language Corpus, both overrepresenting English-speaking,
> Western, middle-class children (Henrich et al., 2010). Under a simulated shift
> representing second-language or dialect-switching users — depressed linguistic
> maturity indicators with inflated variance, and reduced child-account prevalence
> — overall accuracy falls to 20.2%, with essentially all error in the
> over-restrictive direction. Disaggregating by modality, single-modality
> accuracy under the same shift is 60.0% for behavioural biometrics, 31.6% for
> linguistic features, 30.1% for contextual attestation and 18.5% for device
> indicators. Behavioural signals are thus the most robust to cross-cultural
> transfer and the linguistic signals — which carry the largest privacy budget in
> Table 3 — the least. This is an uncomfortable finding, since behavioural
> biometrics are also the modality carrying the greatest surveillance risk, and
> we do not resolve the tension here: a deployment that rebalanced the budget
> toward behavioural signals for robustness would be trading a fairness gain
> against a privacy cost, and that trade should be made deliberately and in
> public rather than silently in a configuration file.

### Governance of tier definitions

> The framework locates in a formal artefact — the Policy Store, L4 — the
> authority to determine what capabilities a child may reach. The technical
> structure makes that authority tractable: tier definitions are isolated in a
> single versioned layer, and the safety proofs are parametric in the constraint
> sets, so a governing body can amend policy with a compile-time guarantee that
> monotonicity and non-bypassability survive. It does not, however, determine who
> should hold that authority. At minimum we take a viable governance regime to
> require: decision authority shared among developmental psychologists,
> children's-rights advocates, ethicists and regulators rather than vested in the
> deploying firm; a defined revision cycle for tier definitions with published
> rationale for each change; a right of affected families to know which tier has
> been assigned and on what evidence; and an appeal route to a human decision.
> None of these follow from the mathematics, and we do not claim the architecture
> supplies them.

---

## 6. Data Availability — replace entirely

> This study is a formal specification with computational validation. It does not
> involve human subjects and generates no personal data. The complete reference
> implementation, the seeded generator for the 7,000-prompt evaluation corpus,
> both ground-truth labellers, all baseline implementations, and the seven
> experiment scripts that produce every table in this paper are available at
> [URL]. All results derive from a single master seed (20260806) and are
> reproducible with `python3 experiments/run_all.py`; each result file records
> the Python and NumPy versions, platform and commit under which it was produced.
> The tier-conditional signal distributions are synthetic, parameterised from
> published norms in the Oxford Children's Language Corpus (Hsiao et al., 2024)
> and CHILDES-db (Sanchez et al., 2019); no child-produced text was used.

---

## 7. Two corrections inside the specification

Both were found by the test suite and both need a line in the text.

**Table 7, homework row.** The table records "Blocked" at t2–t4, but Section 3.5
says direct answers to assessed work are intercepted and replaced by guided
questioning — which is the Socratic access level, not Blocked. The two readings
score very differently against an external rubric, and under the printed version
NPL loses to the age-conditioned baseline. Change the row to
Blocked / Socratic / Socratic / Socratic / Toggle and note in the caption that
Socratic denotes interception, not refusal.

**Severity gating.** The gating matrix is indexed by capability alone and so
cannot distinguish a general question about alcohol from a request for
instructions; as printed it admits high-severity substance material at t5. Add
to Section 3.4: a severity gate applied uniformly across tiers refuses content in
the crisis, substance/body-image and age-inappropriate categories above a
severity threshold regardless of tier. Because it is uniform it tightens every
tier equally and preserves the monotonicity of Invariant I.
