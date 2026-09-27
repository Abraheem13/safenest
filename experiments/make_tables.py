"""Emit every manuscript table as LaTeX, straight from results/*.json.

Nothing here is typed twice: each table is generated from the recorded output of
the experiment that produced it, so a published number and the number the code
produced cannot drift apart. Output follows the MDPI journal class (`mdpi.cls`):
captions above tables, `tabularx` at text width, notes below the rule.

    python3 experiments/make_tables.py      # writes tex/tab_*.tex
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS, ROOT  # noqa: E402
from safenest.signals import (  # noqa: E402
    CONTEXT_P,
    DEVICE_P,
    LINGUISTIC_MEAN,
    LINGUISTIC_STD,
    TYPING_LOG_SIGMA,
    TYPING_MEDIAN_WPM,
)
from safenest.tiers import ALL_TIERS, TIER_SPECS  # noqa: E402

TEX = ROOT / "tex"
TEX.mkdir(exist_ok=True)

AGE_AGNOSTIC = ["Topical rail", "Harm classifier", "Agent firewall", "Constitutional rule"]
AGE_AWARE = ["COPPA binary rule", "Child-safety classifier", "Age-band oracle"]
NPL_ROWS = ["NPL without Socratic substitution", "NPL (full)"]
#: Rule-based baselines carry a dagger: each models a deployed system's
#: documented behaviour and is not that system.
DAGGER = {name: name + "$^{\\dagger}$" for name in AGE_AGNOSTIC}
CORPUS = {"persuade": "PERSUADE 2.0", "asap": "ASAP", "ellipse": "ELLIPSE",
          "gillam": "CHILDES Gillam", "enni": "CHILDES ENNI"}
ESTIMATORS = ["NPL (specified)", "NPL (learned)", "Logistic (features)",
              "Gradient boosting (features)", "Logistic (TF-IDF)", "Logistic (MiniLM)"]
MODELLED = ("$^{\\dagger}$Decision rule modelled on the documented behaviour of, respectively, "
            "NeMo Guardrails~\\cite{Rebedea2023}, Llama Guard~\\cite{Inan2023}, "
            "LlamaFirewall~\\cite{MetaAI2025} and Constitutional AI~\\cite{Bai2022}; "
            "not the shipped systems.")


def load(n: str) -> dict:
    return json.loads((RESULTS / f"{n}.json").read_text())


def available(n: str) -> bool:
    p = RESULTS / f"{n}.json"
    return p.exists() and "skipped" not in json.loads(p.read_text())


def write(name: str, body: str) -> None:
    (TEX / f"{name}.tex").write_text(body.rstrip() + "\n")
    print(f"  wrote tex/{name}.tex")


def pc(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def sg(x: float, d: int = 1) -> str:
    """Signed number; a value that rounds to zero is printed unsigned."""
    t = f"{x:+.{d}f}"
    return t[1:] if float(t) == 0 else t


def tier_math(label: str) -> str:
    return f"$t_{label[1]}$"


def note(text: str) -> str:
    return "\n\\noindent{\\footnotesize{" + text + "}}"


def table(name: str, caption: str, label: str, colspec: str, head: str,
          rows: list[str], foot: str, wide: bool = False) -> None:
    width = "\\fulllength" if wide else "\\textwidth"
    body = ("\\begin{table}[H]\n" + ("\\begin{adjustwidth}{-\\extralength}{0cm}\n" if wide else "")
            + f"\\caption{{{caption}\\label{{{label}}}}}\n\\small\n"
            + f"\\begin{{tabularx}}{{{width}}}{{{colspec}}}\n"
            + "\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabularx}" + note(foot)
            + ("\n\\end{adjustwidth}" if wide else "") + "\n\\end{table}")
    write(name, body)


# ------------------------------------------------------------ conference
def tab_conference() -> None:
    comp = load("exp04_comparative")
    abl = load("exp12_ablation")["variants"]
    rub = comp["results"]["rubric"]
    mat = comp["results"]["matrix"]
    real = "---"
    if available("exp13_real_estimation"):
        n_docs = sum(sum(v.values()) for v in load("exp13_real_estimation")["documents"].values())
        real = (f"Tier estimators learned from three essay corpora ({n_docs:,} essays) and "
                "evaluated across corpora; real-subgroup and privacy analyses").replace(",", "{,}")
    rows = [
        r"Ground truth & Labels derived from the policy's own feature-gating matrix & "
        r"Independent rubric that never consults the policy; matrix labels reported for contrast \\",
        r"Headline DSR & 88.6\% vs.\ 47.0\% (best age-agnostic) & "
        f"{pc(rub['NPL (full)']['overall']['dsr'])}\\% vs.\\ "
        f"{pc(rub['Constitutional rule']['overall']['dsr'])}\\% (best age-agnostic) and "
        f"{pc(rub['Age-band oracle']['overall']['dsr'])}\\% (oracle age bands); "
        f"{pc(abl['As first specified (neither amendment)']['dsr'])}\\% before two "
        f"amendments; {pc(mat['NPL (full)']['overall']['dsr'])}\\% under matrix labels \\\\",
        r"Baselines & Four age-agnostic filters, calibrated to a target under-protection "
        r"range & Four age-agnostic rules at fixed and at tuned thresholds, three age-aware "
        r"baselines (including an oracle age-band policy under six band partitions), an "
        r"ablated NPL \\",
        r"Privacy & Per-signal Laplace noise, described as an $\varepsilon$-DP posterior "
        r"at $\varepsilon=1$ & \emph{Corrected}: local DP on the live child's signals gives "
        r"near-chance accuracy; the guarantee is a corpus-level $(\varepsilon,\delta)$ "
        r"release with stated sensitivity, also applied to parameters learned from real "
        r"text \\",
        r"Estimator theory & None & Proved finite-sample misassignment bound, tested "
        r"against real text \\",
        r"Formal properties & Four invariants (fail-safe default, complete mediation, "
        r"monotonicity, crisis pre-emption), proofs by hand & Three invariants verified "
        r"exhaustively over a region-complete abstraction and the tier-authority "
        r"automaton; fail-safe default and complete mediation become assumptions and "
        r"design rules (Section~\ref{sec:formal}); two defects found \\",
        r"Optimisation & Bilevel formulation of safety and utility & Removed: no quantity "
        r"is optimised jointly; the learned components are the tier estimators \\",
        r"Socratic engine & Expected value said to peak at $t_2$ & Value rises with tier "
        r"once the reward is stated in full (Table~\ref{tab:socratic}) \\",
        r"Policy & Seven risk categories; crisis blocked at every tier & Nine "
        r"capabilities; severity gate on substance and age-inappropriate content; crisis "
        r"disclosures referred, not refused, from $t_3$ \\",
        r"Real data & None & " + real + r" \\",
        r"New analyses & --- & Ablation, achievable ceiling, threshold sweeps, simulated "
        r"atypical profiles, impersonation adversary, calibration, overhead including "
        r"feature extraction \\",
    ]
    table("tab_conference",
          "What changed between the conference version~\\cite{Ejaz2026UKCI} and this article.",
          "tab:conference", r">{\raggedright\arraybackslash}p{2.3cm}LL",
          r"\textbf{Aspect} & \textbf{UKCI 2026 paper} & \textbf{This article} \\",
          rows,
          "DSR: developmental safety rate (Section~\\ref{sec:metrics}). The conference "
          "numbers are reproduced from~\\cite{Ejaz2026UKCI}. The random seed also changed "
          "(20260720 to 20260806), but seed-to-seed variation is "
          f"{100 * load('exp11_reliability')['seed_variance']['NPL (full)']['sd']:.2f} "
          "points, so the differences come from the changes listed.")


# ------------------------------------------------------------------ params
def tab_params() -> None:
    feats = ["MTLD", "Flesch--Kincaid grade", "Mean sentence length",
             "Parse-tree depth", "Spelling error rate"]
    rows = []
    for k, name in enumerate(feats):
        cells = " & ".join(
            f"{LINGUISTIC_MEAN[t][k]:g} ({LINGUISTIC_STD[t][k]:g})" for t in ALL_TIERS)
        rows.append(f"\\quad {name} & {cells} \\\\")
    typing = " & ".join(f"{TYPING_MEDIAN_WPM[t]:g} ({TYPING_LOG_SIGMA[t]:g})" for t in ALL_TIERS)
    dev = " & ".join(f"{DEVICE_P[t]:.2f}" for t in ALL_TIERS)
    ctx = " & ".join(f"{CONTEXT_P[t]:.2f}" for t in ALL_TIERS)
    rows = ([r"\multicolumn{6}{l}{\emph{Linguistic}, $\mathcal{N}(\mu_k,\mathrm{diag}\,\sigma_k^2)$; local share $\varepsilon_m = 0.40$} \\"]
            + rows
            + [r"\multicolumn{6}{l}{\emph{Behavioural}, $\mathrm{LN}(\log\mathrm{median},\sigma^2)$; local share $\varepsilon_m = 0.30$} \\",
               f"\\quad Typing speed (WPM) & {typing} \\\\",
               r"\multicolumn{6}{l}{\emph{Account} and \emph{attestation}, $\mathrm{Bern}(p_k)$; local shares $\varepsilon_m = 0.20$ and $0.10$} \\",
               f"\\quad Child-account flag & {dev} \\\\",
               f"\\quad Attestation present & {ctx} \\\\"])
    table("tab_params",
          "Tier-conditional signal parameters of the simulation and local privacy budget shares.",
          "tab:params", r">{\raggedright\arraybackslash}p{3.6cm}*{5}{C}",
          r"& \boldmath{$t_1$} & \boldmath{$t_2$} & \boldmath{$t_3$} & \boldmath{$t_4$} & \boldmath{$t_5$} \\",
          rows,
          "Values are author-specified simulation assumptions, not measured developmental "
          "norms; Section~\\ref{sec:real} fits the linguistic block to real text. Standard "
          "deviations in parentheses; for typing speed the parenthesised figure is the "
          "log-scale $\\sigma$. An absent attestation contributes an attenuated "
          "log-likelihood (weight 0.25), so a missing external signal cannot by itself "
          "drive the estimate. The budget shares apply only to the local-DP analysis.")


# ------------------------------------------------------------- convergence
def tab_convergence() -> None:
    d = load("exp01_convergence")
    b = load("exp02_separability")
    ms = d["milestones"]
    bound_ms = [m for m in ms if str(m) in b["misassignment_bound"]["t1"]]
    rows = []
    for t in ALL_TIERS:
        acc = d["accuracy"][t.label]
        first = d["first_milestone_above_90"][t.label]
        cells = [f"\\textbf{{{100 * acc[str(m)]:.1f}}}" if m == first
                 else f"{100 * acc[str(m)]:.1f}" for m in ms]
        spec = TIER_SPECS[t]
        rows.append(f"{tier_math(t.label)} ({spec.age_low}--{spec.age_high}) & "
                    + " & ".join(cells) + r" \\")
    brows = []
    for t in ALL_TIERS:
        cells = [f"{max(100 * (1 - b['misassignment_bound'][t.label][str(m)]), 0.0):.1f}"
                 if m in bound_ms else "--" for m in ms]
        brows.append(f"{tier_math(t.label)} & " + " & ".join(cells) + r" \\")
    first_n = b["first_n_bound_at_most_10pct"]
    head = "\\textbf{Tier} & " + " & ".join(f"\\boldmath{{$n{{=}}{m}$}}" for m in ms) + r" \\"
    table("tab_convergence",
          "Estimator accuracy (\\%) in simulation ($N = 500$ users per tier) and the "
          "guaranteed accuracy implied by Proposition~\\ref{prop:bound}.",
          "tab:convergence", "L*{" + str(len(ms)) + "}{C}", head,
          [f"\\multicolumn{{{len(ms) + 1}}}{{l}}{{\\emph{{Simulated accuracy}}}} \\\\"] + rows
          + ["\\midrule",
             f"\\multicolumn{{{len(ms) + 1}}}{{l}}{{\\emph{{Guaranteed accuracy "
             "(Proposition~\\ref{prop:bound}; 0.0 where the bound is vacuous)}} \\\\"]
          + brows,
          "Bold marks the first milestone at or above 90\\%. The simulation uses the exact "
          "likelihood parameters and applies the bypass check, which can only hold a "
          "session at a confirmed tier; the bound applies to the assignment rule of "
          "Equation~\\eqref{eq:assign}. Milestones are read from one trajectory per "
          "simulated user and are therefore paired. The bound first guarantees 90\\% at "
          + ", ".join(f"$n={first_n[t.label]}$ for {tier_math(t.label)}" for t in ALL_TIERS)
          + "; it is evaluated at $n \\in \\{1,3,5,7,10\\}$.")


# ---------------------------------------------------------------- corpora
def tab_corpora() -> None:
    d = load("exp13_real_estimation")
    docs = d["documents"]
    spoken = load("exp16_real_spoken") if available("exp16_real_spoken") else None
    rows = []
    info = {"persuade": ("grades 6--12", "CC BY-NC-SA 4.0"),
            "asap": ("grades 7, 8, 10", "competition terms"),
            "ellipse": ("grades 8--12, English learners", "CC BY-NC-SA 4.0")}
    for key in ("persuade", "asap", "ellipse"):
        c = docs[key]
        cells = [str(c.get(a, 0)) for a in ("3", "3,4", "4", "4,5", "5")]
        rows.append(f"{CORPUS[key]} & {info[key][0]} & "
                    + " & ".join(f"{int(x):,}".replace(",", "{,}") for x in cells)
                    + f" & {info[key][1]} \\\\")
    foot = ("Counts are documents with at least one 50-word window. Columns give the "
            "admissible tiers: a boundary grade admits two (grade 7: $t_3$ or $t_4$; "
            f"grade 10: $t_4$ or $t_5$). {d['ellipse_duplicates_removed']} ELLIPSE essays "
            "that also appear in PERSUADE are removed. PERSUADE sets every prompt to a single "
            "grade (Section~\\ref{sec:confound}).")
    if spoken:
        for key in ("gillam", "enni"):
            if key in spoken["corpora"]:
                c = spoken["corpora"][key]
                bt = c["by_tier"]
                rows.append(f"{CORPUS[key]} & ages {c['age_range'][0]:.0f}--{c['age_range'][1]:.0f}, "
                            f"{c['impaired']} impaired & "
                            f"\\multicolumn{{5}}{{c}}{{$t_1$ {bt.get('t1', 0)}, $t_2$ {bt.get('t2', 0)}, "
                            f"$t_3$ {bt.get('t3', 0)}}} & TalkBank rules \\\\")
    table("tab_corpora", "Corpora of child-produced language used in Section~\\ref{sec:real}.",
          "tab:corpora", r">{\raggedright\arraybackslash}p{2.3cm}Lcccccl",
          r"\textbf{Corpus} & \textbf{Writers} & \boldmath{$t_3$} & \boldmath{$t_3,t_4$} & \boldmath{$t_4$} & \boldmath{$t_4,t_5$} & \boldmath{$t_5$} & \textbf{Licence} \\",
          rows, foot, wide=True)


# ------------------------------------------------------------- real LOCO
def tab_real_loco() -> None:
    d = load("exp13_real_estimation")["leave_one_corpus_out"]
    n = "n10"
    rows = []
    for name in ESTIMATORS:
        cells = []
        for held in ("persuade", "asap", "ellipse"):
            r = d[held][name][n]
            cells += [pc(r["balanced_accuracy"]), pc(r["under"]), pc(r["over"])]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    head = (r"& \multicolumn{3}{c}{\textbf{PERSUADE held out}} & \multicolumn{3}{c}{\textbf{ASAP held out}} & \multicolumn{3}{c}{\textbf{ELLIPSE held out}} \\"
            "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" "\n"
            r"\textbf{Estimator} & \textbf{Bal.} & \textbf{Under} & \textbf{Over} & \textbf{Bal.} & \textbf{Under} & \textbf{Over} & \textbf{Bal.} & \textbf{Under} & \textbf{Over} \\")
    table("tab_real_loco",
          "Tier estimation from real writing, trained on two corpora and tested on the third, "
          "after $n = 10$ interactions (50-word windows).",
          "tab:real-loco", r"L*{9}{c}", head, rows,
          "All values are percentages. Bal.: balanced accuracy over admissible-tier groups; "
          "Under: a less protective tier than any admissible one; Over: a more protective "
          "one, including the $t_1$ floor. NPL (specified) uses the simulation parameters of "
          "Table~\\ref{tab:params}; every other estimator is fitted to the training corpora.",
          wide=True)


def tab_real_privacy() -> None:
    d = load("exp14_real_privacy")
    w = d["within_persuade"]
    rows = [f"Maximum likelihood & -- & -- & {pc(w['maximum_likelihood']['balanced_accuracy'])} & "
            f"{pc(w['maximum_likelihood']['under'])} & {pc(w['maximum_likelihood']['over'])} \\\\",
            f"Clipped, no noise & 100.0 & 0.000 & {pc(w['clipped_no_noise']['balanced_accuracy'])} & "
            f"{pc(w['clipped_no_noise']['under'])} & {pc(w['clipped_no_noise']['over'])} \\\\"]
    for e in ("10.0", "3.0", "1.0", "0.3", "0.1"):
        r = w[e]
        rows.append(f"$\\varepsilon = {float(e):g}$ & {pc(r['agreement_with_non_private']['mean'])} & "
                    f"{r['mean_error_in_sd']['mean']:.3f} & "
                    f"{pc(r['balanced_accuracy']['mean'])} $\\pm$ {pc(r['balanced_accuracy']['sd'])} & "
                    f"{pc(r['under']['mean'])} & {pc(r['over']['mean'])} \\\\")
    table("tab_real_privacy",
          "Private release of the learned estimator's parameters, five folds within "
          "PERSUADE ($n = 10$ interactions, $\\delta = 10^{-5}$).",
          "tab:real-privacy", "Lccccc",
          r"\textbf{Release} & \textbf{Agreement} & \textbf{Mean error} & \textbf{Bal. acc.} & \textbf{Under} & \textbf{Over} \\",
          rows,
          "Agreement: share of sessions assigned the same tier as by the clipped release "
          "without noise (\\%). Mean error: absolute error of the released means in "
          "standard deviations of that release. Accuracies are percentages, mean $\\pm$ SD "
          f"over {d['n_releases']} releases. Squared deviations are clipped at "
          f"{d['deviation_clip']} of each feature's public range.")


def tab_real_subgroups() -> None:
    d = load("exp15_real_subgroups")["training"]
    labels = {"ell": "English learner", "disability": "Identified disability",
              "econ": "Economically disadvantaged", "gender": "Female"}

    def cell(x: dict) -> str:
        return f"${sg(100 * x['estimate'])}$ [${sg(100 * x['ci'][0])}$, ${sg(100 * x['ci'][1])}$]"

    rows = []
    for train in ("with ELLIPSE", "without ELLIPSE"):
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{Trained {train}}}}} \\\\")
        for col, lab in labels.items():
            cells = []
            for name in ("NPL (learned)", "Gradient boosting (features)"):
                diff = d[train]["subgroups"][col][name]["difference"]
                cells += [cell(diff["under"]), cell(diff["over"])]
            rows.append(f"\\quad {lab} & " + " & ".join(cells) + r" \\")
    head = (r"& \multicolumn{2}{c}{\textbf{NPL (learned)}} & \multicolumn{2}{c}{\textbf{Gradient boosting}} \\"
            "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r"\textbf{Group} & \textbf{Under} & \textbf{Over} & \textbf{Under} & \textbf{Over} \\")
    table("tab_real_subgroups",
          "Difference in grade-standardised error rates between each group and its "
          "complement in PERSUADE (percentage points, with 95\\% interval).",
          "tab:real-subgroups", r"L*{4}{c}", head, rows,
          "Positive values mean the group is under- or over-protected more often than its "
          "complement in the same grade. Folds hold out whole prompts; comparisons use only "
          "grades in which both groups appear. Intervals are parametric bootstrap intervals "
          "over per-grade proportions (2000 resamples).", wide=True)


# --------------------------------------------------------------------- DSR
def tab_dsr() -> None:
    comp = load("exp04_comparative")
    d = comp["results"]["rubric"]
    sd = load("exp11_reliability")["seed_variance"]
    diff = comp["paired_differences"]
    groups = [("Age-agnostic", AGE_AGNOSTIC), ("Age-aware", AGE_AWARE),
              ("Nested Policy Learning", NPL_ROWS)]
    rows = []
    for gname, names in groups:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{gname}}}}} \\\\")
        for name in names:
            o = d[name]["overall"]
            full = name == "NPL (full)"
            label = DAGGER.get(name, name)
            label = f"\\quad \\textbf{{{label}}}" if full else f"\\quad {label}"
            val = f"\\textbf{{{pc(o['dsr'])}}}" if full else pc(o["dsr"])
            gap = ("ref." if full else
                   f"${sg(100 * diff[name]['difference'])}$ [${sg(100 * diff[name]['ci_low'])}$, "
                   f"${sg(100 * diff[name]['ci_high'])}$]")
            rows.append(f"{label} & {val} & [{pc(o['dsr_ci_low'])}, {pc(o['dsr_ci_high'])}] & "
                        f"{pc(o['under_protection'])} & {pc(o['over_restriction'])} & "
                        f"{100 * sd[name]['sd']:.2f} & {gap} \\\\")
    table("tab_dsr",
          "Developmental safety rate (DSR) against the independent rubric, $N = 7000$ "
          "structured records.",
          "tab:dsr", "Lcccccc",
          r"\textbf{Framework} & \textbf{DSR} & \textbf{95\% CI} & \textbf{Under} & \textbf{Over} & \textbf{SD} & \textbf{NPL minus framework} \\",
          rows,
          "All values are percentages; DSR, under-protection and over-restriction sum to 100. "
          "The confidence interval is the Wilson score interval; SD is the standard deviation "
          "of DSR across 20 independently generated corpora of the same size; the last column "
          "is the paired difference with a 95\\% bootstrap interval (2000 resamples). "
          + MODELLED, wide=True)


def tab_tuned() -> None:
    comp = load("exp04_comparative")
    rows = []
    for name, v in comp["tuned_age_agnostic"].items():
        th = ", ".join(f"{float(x):.2f}" for x in v["thresholds"].values())
        rows.append(f"{DAGGER[name]} & {th} & {pc(v['dsr'])} & {pc(v['under_protection'])} & "
                    f"{pc(v['over_restriction'])} \\\\")
    bands = comp["band_partitions"]
    for key, v in bands["by_partition"].items():
        tags = []
        if key == bands["default"]:
            tags.append("default")
        if key == bands["selected"]:
            tags.append("selected")
        tag = f" ({', '.join(tags)})" if tags else ""
        rows.append(f"Age-band oracle{tag} & {_bands(key)} & {pc(v['dsr'])} & "
                    f"{pc(v['under_protection'])} & {pc(v['over_restriction'])} \\\\")
    t = comp["tuned_npl"]
    rows.append(f"NPL (full), tuned & $\\theta_{{\\mathrm{{sev}}}}={t['severity_threshold']:.2f}$ & "
                f"{pc(t['dsr'])} & {pc(t['under_protection'])} & {pc(t['over_restriction'])} \\\\")
    table("tab_tuned",
          "Every framework at its best: thresholds and band partitions selected on a "
          "separate tuning corpus and scored on the evaluation corpus.",
          "tab:tuned", r"L>{\raggedright\arraybackslash}p{4.2cm}ccc",
          r"\textbf{Framework} & \textbf{Setting} & \textbf{DSR} & \textbf{Under} & \textbf{Over} \\",
          rows,
          "All values are percentages. Age-agnostic thresholds are searched on a grid of "
          "step 0.05 and listed in the order crisis, substance. Every contiguous three-band "
          "partition of the five tiers is shown for the oracle; the partition selected on "
          "the tuning corpus is marked. " + MODELLED)


def _bands(key: str) -> str:
    import ast

    parts = ast.literal_eval(key)
    return ", ".join("$\\{" + ",".join(f"t_{t}" for t in b) + "\\}$" for b in parts)


# ---------------------------------------------------------------- ablation
def tab_ablation() -> None:
    d = load("exp12_ablation")["variants"]
    pretty = {"Full framework": "Full framework",
              "-- crisis referral (earlier refusal)": "Crisis refused above $\\theta_{\\mathrm{sev}}$ (earlier specification)",
              "-- scaffolding below $t_3$": "Without scaffolding below $t_3$",
              "As first specified (neither amendment)": "As first specified (neither amendment)",
              "-- severity gate": "Without severity gate",
              "-- Socratic substitution": "Without Socratic substitution"}
    rows = []
    for key, label in pretty.items():
        v = d[key]
        delta = "ref." if key == "Full framework" else f"${sg(v['delta_pp'])}$"
        rows.append(f"{label} & {pc(v['dsr'])} & {delta} & {pc(v['under_protection'])} & "
                    f"{pc(v['over_restriction'])} & {pc(v['harm_dsr'])} \\\\")
    worst = max(v["mcnemar"]["p_value"] for k, v in d.items() if k != "Full framework")
    table("tab_ablation", "Component ablation against the independent rubric, $N = 7000$.",
          "tab:ablation", "Lccccc",
          r"\textbf{Variant} & \textbf{DSR} & \boldmath{$\Delta$} & \textbf{Under} & \textbf{Over} & \textbf{Harm DSR} \\",
          rows,
          "All values are percentages; $\\Delta$ is in percentage points relative to the "
          "full framework. Harm DSR is restricted to the three content-harm categories "
          "($N = 3000$). The two amendments were made after the rubric exposed the defects "
          "they correct (Section~\\ref{sec:groundtruth}). Every variant differs from the "
          f"full framework at $p < {_sci_bound(worst)}$ (McNemar).")


# -------------------------------------------------------------- populations
def tab_populations() -> None:
    d = load("exp06_populations")
    pretty = {"typical": "Typical",
              "verbally_advanced": "Verbally advanced (+2 tiers)",
              "advanced_atypical_motor": "Advanced (+1), atypical motor",
              "second_language_shared_device": "Second language ($-$1), shared device",
              "high_variance": "High variance"}
    rows = []
    for key, label in pretty.items():
        s = d["by_population"][key]["unmitigated"]
        f = d["by_population"][key]["with_discordance_flag"]
        rows.append(f"{label} & {pc(s['correct'] / s['n'])} & {pc(s['under'] / s['n'])} & "
                    f"{pc(s['over'] / s['n'])} & {pc(f['correct'] / f['n'])} & "
                    f"{pc(f['under'] / f['n'])} \\\\")
    head = (r"& \multicolumn{3}{c}{\textbf{Estimator alone}} & \multicolumn{2}{c}{\textbf{With discordance rule}} \\"
            "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" "\n"
            r"\textbf{Simulated profile} & \textbf{Acc.} & \textbf{Under} & \textbf{Over} & \textbf{Acc.} & \textbf{Under} \\")
    table("tab_populations",
          f"Tier estimation for simulated atypical profiles after $n = {d['n_interactions']}$ "
          f"interactions, $N = {d['n_trials']}$ users each.",
          "tab:pops", "Lccccc", head, rows,
          "All values are percentages. The discordance rule holds a user at an externally "
          "attested tier when the estimate differs from it by two tiers or more. Profiles are "
          "perturbations of the simulation's signal model, named for the pattern they "
          "produce; Section~\\ref{sec:subgroups} measures real subgroups.")


# ------------------------------------------------------------------ bypass
def tab_bypass() -> None:
    d = load("exp10_bypass")
    rows = [f"{a:.2f} & {pc(d['by_spoof_strength'][str(a)]['escalation_rate'])} & "
            f"{pc(d['by_spoof_strength'][str(a)]['detection_rate'])} \\\\" for a in d["alphas"]]
    pretty = {"typical": "typical", "verbally_advanced": "verbally advanced",
              "second_language_shared_device": "second-language",
              "high_variance": "high-variance"}
    ff = "; ".join(f"{pretty[k]} {pc(v)}\\%" for k, v in d["false_flag_rates"].items())
    table("tab_bypass",
          f"Impersonation of $t_5$ by a genuine $t_2$ user after $n = {d['n_interactions']}$ "
          f"interactions, $N = {d['n_trials']}$ sessions per strength.",
          "tab:bypass", "CCC",
          r"\textbf{Spoof strength} \boldmath{$\alpha$} & \textbf{Escalated above} \boldmath{$t_2$} \textbf{(\%)} & \textbf{Flagged by detector (\%)} \\",
          rows,
          "Only the linguistic channel is spoofed. False-flag rates on genuine $t_2$ children "
          f"at the adopted threshold $\\chi^2_{{5,0.99}}=15.086$: {ff}. At $\\alpha = 0$ the "
          "user is a typical $t_2$ child, so that row and the typical false-flag rate are "
          "independent samples of the same nominal 1\\% rate. Detector AUC separating "
          "a full-strength impersonator from a genuinely verbally advanced child: "
          f"{d['auc_vs_verbally_advanced']:.3f}.")


# ---------------------------------------------------------------- overhead
def tab_overhead() -> None:
    d = load("exp07_overhead")
    m = d["measurements_seconds"]
    order = [("$L_0$ token filter", "L0: Token Filter (per response, 40 tokens)", "per response", "$O(|y|)$", "yes"),
             ("$L_1$ Socratic guard", "L1: Socratic Guard (per response)", "per response", "$O(1)$", "yes"),
             ("$L_4$ policy store", "L4: Policy Store (per query)", "per query", "$O(1)$", "yes"),
             ("Socratic policy lookup", "Socratic MDP lookup (per step)", "per step", "$O(1)$", "yes"),
             ("Full engine, end to end", "Full engine (per response)", "per response", "$O(|y|)$", "yes"),
             ("Tier update ($L_2$)", "L2: Bayesian update (per interaction)", "per interaction", "$O(KM)$", "no"),
             ("Five linguistic features", "Feature extraction (per 50-word message)", "per message", "$O(|x|)$", "no"),
             ("Sentence embedding", "Sentence embedding (per 50-word message, CPU)", "per message", "$O(|x|)$", "no")]
    rows = []
    for label, key, gran, cx, sync in order:
        if key not in m:
            continue
        v = m[key]
        t = f"{v * 1e6:.2f}" if v < 1e-3 else f"{v * 1e6:,.0f}".replace(",", "{,}")
        rows.append(f"{label} & {gran} & {t} & {cx} & {sync} \\\\")
    feat = m.get("Feature extraction (per 50-word message)", 0.0)
    emb = m.get("Sentence embedding (per 50-word message, CPU)", 0.0)
    table("tab_overhead",
          "Measured cost per component on one CPU core (median of five runs).",
          "tab:overhead", "Llccc",
          r"\textbf{Component} & \textbf{Granularity} & \textbf{Time (\boldmath{$\mu$}s)} & \textbf{Complexity} & \textbf{Synchronous} \\",
          rows,
          f"The synchronous engine costs {100 * d['share_of_500ms_llm']:.4f}\\% of a 500 ms "
          "model response. Computing the inputs dominates: the five features and a sentence "
          f"embedding together take {1e3 * (feat + emb):.1f} ms per message "
          f"({100 * (feat + emb) / 0.5:.1f}\\% of 500 ms), off the response path. $|y|$ is the "
          "response length and $|x|$ the message length in tokens, $K$ the number of tiers "
          "and $M$ the number of modalities. Platform: " + d["platform"].split("-")[0]
          + ", " + d.get("processor", "") + ".")


# ---------------------------------------------------------------- appendix
def tab_socratic() -> None:
    d = load("exp03_socratic")
    acts = ["elicit", "hint", "guide", "partial_explain", "verify_request"]
    rows = [f"{tier_math(t.label)} & "
            + " & ".join(f"{d['action_distribution_step1'][t.label][a]:.1f}" for a in acts)
            + f" & {d['expected_value'][t.label]:.3f} \\\\" for t in ALL_TIERS]
    plain = ", ".join(f"{v:.1f}" for v in d["verify_usage_published_eq15"])
    table("tab_socratic",
          "Optimal Socratic action at protocol step 1: share of knowledge states (\\%) for "
          "which each action is optimal.",
          "tab:socratic", "Cccccccc",
          r"\textbf{Tier} & \textbf{Elicit} & \textbf{Hint} & \textbf{Guide} & \textbf{PartialExplain} & \textbf{VerifyRequest} & \boldmath{$E[V]$} \\",
          rows,
          "With the metacognition term removed ($\\alpha_{\\mathrm{meta}} = 0$), VerifyRequest "
          f"is optimal for {plain}\\% of states at $t_1$--$t_5$. PartialExplain is never "
          "optimal as an opening action at any tier, in this configuration or in any of the "
          f"{d['sensitivity']['n_valid']} swept alternatives.")


def tab_stats() -> None:
    comp = load("exp04_comparative")
    rows = []
    for name, v in comp["paired_differences"].items():
        mc = v["mcnemar"]
        p = "$< 10^{-15}$" if mc["p_value"] < 1e-15 else f"{mc['p_value']:.2g}"
        pairs = f"{int(mc['b01']):,} / {int(mc['b10']):,}".replace(",", "{,}")
        rows.append(f"{DAGGER.get(name, name)} & ${sg(100 * v['difference'])}$ & "
                    f"[${sg(100 * v['ci_low'])}$, ${sg(100 * v['ci_high'])}$] & "
                    f"{pairs} & {p} \\\\")
    table("tab_stats",
          "Paired comparisons of the full framework with each alternative, $N = 7000$.",
          "tab:stats", "Lcccc",
          r"\textbf{Full NPL versus} & \boldmath{$\Delta$} \textbf{(pp)} & \textbf{95\% interval} & \textbf{Discordant pairs} & \textbf{McNemar} \boldmath{$p$} \\",
          rows,
          "The interval is a paired bootstrap percentile interval (2000 resamples). "
          "Discordant pairs count records NPL gets right and the alternative wrong, and the "
          "reverse. On a synthetic corpus whose size the authors choose, $p$-values measure "
          "that size as much as the effect, so values below $10^{-15}$ are reported as a "
          "bound and the interval is the primary statistic. " + MODELLED)


def _sci_bound(x: float) -> str:
    if x <= 0:
        return "10^{-15}"
    return f"10^{{{max(math.ceil(math.log10(x)), -15)}}}"


def tab_thresholds() -> None:
    d = load("exp09_operating")
    g_rows = []
    for g in d["gamma_grid"]:
        t = d["gamma_typical"][str(g)]
        v = d["gamma_verbally_advanced"][str(g)]
        mark = "$^{\\ast}$" if g == d["adopted_gamma"] else ""
        g_rows.append(f"{g:.2f}{mark} & {pc(t['accuracy'])} & {pc(t['under_protection'])} & "
                      f"{pc(t['over_restriction'])} & {pc(v['accuracy'])} & "
                      f"{pc(v['under_protection'])} \\\\")
    s_rows = []
    for th in d["severity_grid"]:
        r = d["severity_sweep"][str(th)]
        e = d["severity_sweep_earlier_crisis_refusal"][str(th)]
        mark = "$^{\\ast}$" if th == d["adopted_severity_threshold"] else ""
        s_rows.append(f"{th:.1f}{mark} & {pc(r['dsr'])} & {pc(r['under_protection'])} & "
                      f"{pc(r['over_restriction'])} & {pc(e['dsr'])} & {pc(e['under_protection'])} \\\\")
    head = (r"& \multicolumn{3}{c}{\textbf{Typical cohort}} & \multicolumn{2}{c}{\textbf{Verbally advanced}} \\"
            "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" "\n"
            r"\boldmath{$\gamma_{\mathrm{conf}}$} & \textbf{Acc.} & \textbf{Under} & \textbf{Over} & \textbf{Acc.} & \textbf{Under} \\")
    rows = g_rows + ["\\midrule",
                     r"& \multicolumn{3}{c}{\textbf{Crisis referred}} & \multicolumn{2}{c}{\textbf{Crisis refused (earlier)}} \\",
                     r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
                     r"\boldmath{$\theta_{\mathrm{sev}}$} & \textbf{Harm DSR} & \textbf{Under} & \textbf{Over} & \textbf{Harm DSR} & \textbf{Under} \\",
                     "\\midrule"] + s_rows
    table("tab_thresholds", "Sweeps over the two free thresholds.", "tab:thresholds",
          "Cccccc", head, rows,
          "All values are percentages; $^{\\ast}$ marks the adopted setting. The confidence "
          f"sweep uses $n = {d['n_interactions']}$ interactions and $N = {d['n_trials']}$ "
          "users per cohort, so its accuracies differ slightly from Table~\\ref{tab:pops}. The "
          "severity sweep is restricted to the three content-harm categories ($N = 3000$). "
          "The rubric blocks substance content above severity 0.5 and the gate refuses it "
          "above $\\theta_{\\mathrm{sev}}$, so the two agree by construction at 0.5.")


#: Risk categories in the order of Section 5.3, with their printed names.
CATEGORY_LABEL = {"homework_assignment": "Homework", "open_ended_chat": "Open-ended chat",
                  "code_generation": "Code generation", "essay_creative_writing": "Essay writing",
                  "crisis_self_harm": "Crisis and self-harm",
                  "substance_body_image": "Substance and body image",
                  "age_inappropriate_content": "Age-inappropriate content"}


def tab_category() -> None:
    comp = load("exp04_comparative")
    hw_all = comp["per_category_t2_rubric_all_frameworks"]
    rows = []
    for cat, label in CATEGORY_LABEL.items():
        per = hw_all[cat]
        npl = per["NPL (full)"]
        cai = per["Constitutional rule"]
        orc = per["Age-band oracle"]
        rows.append(f"{label} & {pc(npl)} & {pc(cai)} & "
                    f"${sg(100 * (npl - cai))}$ & {pc(orc)} & ${sg(100 * (npl - orc))}$ \\\\")
    hw = hw_all["homework_assignment"]
    best = max((v, k) for k, v in hw.items() if k != "NPL (full)")
    table("tab_category",
          "Per-category DSR (\\%) at $t_2$ (ages 7--9) under the independent rubric, "
          "$N = 200$ per cell.",
          "tab:percat", "lCCCCC",
          r"& & \multicolumn{2}{c}{\textbf{Constitutional rule}} & \multicolumn{2}{c}{\textbf{Age-band oracle}} \\" "\n"
          r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
          r"\textbf{Risk category} & \textbf{NPL} & \textbf{DSR} & \boldmath{$\Delta$} & \textbf{DSR} & \boldmath{$\Delta$} \\",
          rows,
          "$\\Delta$ is NPL's DSR minus the framework's, in percentage points. "
          "The rubric scaffolds code generation and essay writing at $t_2$; NPL and the "
          "age-band oracle block both and the constitutional rule allows both. On homework "
          f"at $t_2$ the best alternative is the {best[1].lower()} at {pc(best[0])}\\%.")


def main() -> None:
    tab_conference(); tab_params(); tab_convergence()
    if available("exp13_real_estimation"):
        tab_corpora(); tab_real_loco()
    if available("exp14_real_privacy"):
        tab_real_privacy()
    if available("exp15_real_subgroups"):
        tab_real_subgroups()
    tab_dsr(); tab_tuned(); tab_ablation(); tab_populations(); tab_bypass()
    tab_overhead(); tab_socratic(); tab_stats(); tab_thresholds(); tab_category()


if __name__ == "__main__":
    main()
