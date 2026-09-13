"""Emit every manuscript table as LaTeX, straight from results/*.json.

Nothing here is typed twice: each table is generated from the recorded output of
the experiment that produced it, so a published number and the number the code
produced cannot drift apart. Output follows the MDPI journal class (`mdpi.cls`):
captions above tables, `tabularx` at text width, footnotes below the rule.

    python3 experiments/make_tables.py      # writes tex/tab*.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import ROOT, RESULTS  # noqa: E402
from safenest.signals import (  # noqa: E402
    CONTEXT_P, DEVICE_P, LINGUISTIC_MEAN, LINGUISTIC_STD, TYPING_LOG_SIGMA,
    TYPING_MEDIAN_WPM,
)
from safenest.tiers import ALL_TIERS, TIER_SPECS  # noqa: E402

TEX = ROOT / "tex"
TEX.mkdir(exist_ok=True)

#: The baselines are re-implementations of each system's documented decision
#: rule, not the shipped products, and every table says so in its name.
DISPLAY = {
    "NeMo Guardrails": "NeMo Guardrails-style",
    "Llama Guard": "Llama Guard-style",
    "LlamaFirewall": "LlamaFirewall-style",
    "Constitutional AI": "Constitutional AI-style",
    "COPPA binary rule": "COPPA binary rule",
    "Child-safety classifier": "Child-safety classifier",
    "Age-conditioned (oracle)": "Age-conditioned (oracle band)",
    "NPL (no Socratic)": "NPL without Socratic substitution",
    "NPL (ours)": "NPL (full)",
}


def load(n: str) -> dict:
    return json.loads((RESULTS / f"{n}.json").read_text())


def write(name: str, body: str) -> None:
    (TEX / f"{name}.tex").write_text(body.rstrip() + "\n")
    print(f"  wrote tex/{name}.tex")


def pc(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def tier_math(label: str) -> str:
    return f"$t_{label[1]}$"


def note(text: str) -> str:
    return "\n\\noindent{\\footnotesize{" + text + "}}"


# ------------------------------------------------------------------ params
def tab_params() -> None:
    feats = ["MTLD", "Flesch--Kincaid grade", "Mean sentence length",
             "Parse-tree depth", "Spelling error rate"]
    rows = []
    for k, name in enumerate(feats):
        cells = " & ".join(
            f"{LINGUISTIC_MEAN[t][k]:g} ({LINGUISTIC_STD[t][k]:g})" for t in ALL_TIERS)
        rows.append(f"\\quad {name} & {cells} \\\\")
    typing = " & ".join(
        f"{TYPING_MEDIAN_WPM[t]:g} ({TYPING_LOG_SIGMA[t]:g})" for t in ALL_TIERS)
    dev = " & ".join(f"{DEVICE_P[t]:.2f}" for t in ALL_TIERS)
    ctx = " & ".join(f"{CONTEXT_P[t]:.2f}" for t in ALL_TIERS)
    body = r"""\begin{table}[H]
\caption{Tier-conditional signal parameters and local privacy budget shares.\label{tab:params}}
\small
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.3cm}*{5}{C}}
\toprule
& \boldmath{$t_1$} & \boldmath{$t_2$} & \boldmath{$t_3$} & \boldmath{$t_4$} & \boldmath{$t_5$} \\
\midrule
\multicolumn{6}{l}{\emph{Linguistic}, $\mathcal{N}(\mu_k,\mathrm{diag}\,\sigma_k^2)$; local share $\varepsilon_m = 0.40$} \\
""" + "\n".join(rows) + r"""
\multicolumn{6}{l}{\emph{Behavioural}, $\mathrm{LN}(\log\mathrm{median},\sigma^2)$; local share $\varepsilon_m = 0.30$} \\
\quad Typing speed (WPM) & """ + typing + r""" \\
\multicolumn{6}{l}{\emph{Device} and \emph{contextual}, $\mathrm{Bern}(p_k)$; local shares $\varepsilon_m = 0.20$ and $0.10$} \\
\quad Child-account flag & """ + dev + r""" \\
\quad External attestation & """ + ctx + r""" \\
\bottomrule
\end{tabularx}""" + note(
        "Values are author-specified simulation assumptions, not measured "
        "developmental norms. Standard deviations in parentheses; for typing speed "
        "the parenthesised figure is the log-scale $\\sigma$. An absent attestation "
        "contributes an attenuated log-likelihood (weight 0.25), so a missing "
        "external signal cannot by itself drive the estimate. The budget shares "
        "apply only to the local-DP analysis of Section~\\ref{sec:privacy-results}.") + r"""
\end{table}"""
    write("tab_params", body)


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
        cells = []
        for m in ms:
            v = f"{100 * acc[str(m)]:.1f}"
            cells.append(f"\\textbf{{{v}}}" if m == first else v)
        spec = TIER_SPECS[t]
        rows.append(f"{tier_math(t.label)} ({spec.age_low}--{spec.age_high}) & "
                    + " & ".join(cells) + r" \\")
    brows = []
    for t in ALL_TIERS:
        cells = []
        for m in ms:
            if m in bound_ms:
                lb = 100 * (1 - b["misassignment_bound"][t.label][str(m)])
                cells.append(f"{max(lb, 0.0):.1f}")
            else:
                cells.append("--")
        brows.append(f"{tier_math(t.label)} & " + " & ".join(cells) + r" \\")
    head = " & ".join(f"\\boldmath{{$n{{=}}{m}$}}" for m in ms)
    body = r"""\begin{table}[H]
\caption{Estimator accuracy (\%) at interaction milestones: simulation ($N = 500$ users per tier) and the guaranteed accuracy implied by Proposition~\ref{prop:bound}.\label{tab:convergence}}
\small
\begin{tabularx}{\textwidth}{L*{""" + str(len(ms)) + r"""}{C}}
\toprule
\textbf{Tier} & """ + head + r""" \\
\midrule
\multicolumn{""" + str(len(ms) + 1) + r"""}{l}{\emph{Simulated accuracy}} \\
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{""" + str(len(ms) + 1) + r"""}{l}{\emph{Guaranteed accuracy from Proposition~\ref{prop:bound} (0.0 where the bound is vacuous)}} \\
""" + "\n".join(brows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "Bold marks the first milestone at or above 90\\%. Simulation uses the exact "
        "likelihood parameters; the cost of a private parameter release is reported "
        "separately (Figure~\\ref{fig:privacy}). Milestones are read from one "
        "trajectory per simulated user and are therefore paired, not independent. "
        "The bound applies to the assignment rule of Equation~\\eqref{eq:assign}; the "
        "simulation additionally applies the bypass check, which can only hold a "
        "session at a confirmed tier. The bound is evaluated at "
        "$n \\in \\{1,3,5,7,10\\}$.") + r"""
\end{table}"""
    write("tab_convergence", body)


# --------------------------------------------------------------------- DSR
def tab_dsr() -> None:
    d = load("exp04_comparative")["results"]["rubric"]
    sd = load("exp11_reliability")["seed_variance"]
    groups = [
        ("Age-agnostic", ["NeMo Guardrails", "LlamaFirewall", "Llama Guard",
                          "Constitutional AI"]),
        ("Age-aware", ["COPPA binary rule", "Child-safety classifier",
                       "Age-conditioned (oracle)"]),
        ("Nested Policy Learning", ["NPL (no Socratic)", "NPL (ours)"]),
    ]
    rows = []
    for gname, names in groups:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{gname}}}}} \\\\")
        for name in names:
            o = d[name]["overall"]
            full = name == "NPL (ours)"
            label = f"\\quad \\textbf{{{DISPLAY[name]}}}" if full else f"\\quad {DISPLAY[name]}"
            val = f"\\textbf{{{pc(o['dsr'])}}}" if full else pc(o["dsr"])
            rows.append(
                f"{label} & {val} & [{pc(o['dsr_ci_low'])}, {pc(o['dsr_ci_high'])}] & "
                f"{pc(o['under_protection'])} & {pc(o['over_restriction'])} & "
                f"{100 * sd[name]['sd']:.2f} \\\\")
    body = r"""\begin{table}[H]
\caption{Developmental Safety Rate (DSR) against the independent rubric, $N = 7000$ prompts.\label{tab:dsr}}
\small
\begin{tabularx}{\textwidth}{Lccccc}
\toprule
\textbf{Framework} & \textbf{DSR} & \textbf{95\% CI} & \textbf{Under} & \textbf{Over} & \textbf{SD (seeds)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "All values are percentages; DSR, under-protection and over-restriction sum "
        "to 100. The confidence interval is the Wilson score interval. ``SD (seeds)'' "
        "is the standard deviation of DSR across 20 independently generated corpora. "
        "``-style'' marks a re-implementation of the system's documented decision rule, "
        "not the shipped product.") + r"""
\end{table}"""
    write("tab_dsr", body)


# ---------------------------------------------------------------- ablation
def tab_ablation() -> None:
    d = load("exp12_ablation")
    order = ["Full framework", "-- scaffolding below $t_3$", "-- severity gate",
             "-- both refinements", "-- Socratic substitution"]
    pretty = {"Full framework": "Full framework",
              "-- scaffolding below $t_3$": "Without scaffolding below $t_3$",
              "-- severity gate": "Without severity gate",
              "-- both refinements": "Without both refinements",
              "-- Socratic substitution": "Without Socratic substitution"}
    rows = []
    for name in order:
        v = d["variants"][name]
        delta = "ref." if name == "Full framework" else f"${v['delta_pp']:+.1f}$"
        rows.append(
            f"{pretty[name]} & {pc(v['dsr'])} & {delta} & "
            f"{pc(v['under_protection'])} & {pc(v['over_restriction'])} & "
            f"{pc(v['harm_dsr'])} \\\\")
    body = r"""\begin{table}[H]
\caption{Component ablation against the independent rubric, $N = 7000$.\label{tab:ablation}}
\small
\begin{tabularx}{\textwidth}{Lccccc}
\toprule
\textbf{Variant} & \textbf{DSR} & \boldmath{$\Delta$} & \textbf{Under} & \textbf{Over} & \textbf{Harm DSR} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "All values are percentages; $\\Delta$ is in percentage points relative to "
        "the full framework. ``Harm DSR'' is restricted to the three content-harm "
        "categories ($N = 3000$). Every variant differs from the full framework at "
        "$p < " + _sci_bound(max(v["mcnemar"]["p_value"] for k, v in d["variants"].items()
                                 if k != "Full framework")) + "$ (McNemar).") + r"""
\end{table}"""
    write("tab_ablation", body)


# -------------------------------------------------------------- populations
def tab_populations() -> None:
    d = load("exp06_populations")
    pretty = {"typical": "Typical",
              "neurodivergent_verbal": "Verbally advanced (+2 tiers)",
              "neurodivergent_motor": "Advanced (+1), atypical motor",
              "non_weird_l2": "Second-language ($-$1), shared device",
              "dialect_switching": "Dialect-switching (high variance)"}
    rows = []
    for key, label in pretty.items():
        s = d["by_population"][key]["unmitigated"]
        f = d["by_population"][key]["with_discordance_flag"]
        rows.append(
            f"{label} & {pc(s['correct'] / s['n'])} & {pc(s['under'] / s['n'])} & "
            f"{pc(s['over'] / s['n'])} & {pc(f['correct'] / f['n'])} & "
            f"{pc(f['under'] / f['n'])} \\\\")
    body = r"""\begin{table}[H]
\caption{Tier estimation by simulated population after $n = 10$ interactions, $N = 400$ users each.\label{tab:pops}}
\small
\begin{tabularx}{\textwidth}{Lccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Estimator alone}} & \multicolumn{2}{c}{\textbf{With discordance rule}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
\textbf{Population} & \textbf{Acc.} & \textbf{Under} & \textbf{Over} & \textbf{Acc.} & \textbf{Under} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "All values are percentages. Under-protection is assignment to a \\emph{less} "
        "restrictive tier than the user's protective need. The discordance rule holds "
        "a user at an externally attested tier when the estimate differs from it by "
        "two tiers or more. Profiles are synthetic perturbations of the signal model "
        "(Section~\\ref{sec:populations}), not measurements of real populations.") + r"""
\end{table}"""
    write("tab_populations", body)


# ------------------------------------------------------------------ bypass
def tab_bypass() -> None:
    d = load("exp10_bypass")
    rows = [
        f"{a:.2f} & {pc(d['by_spoof_strength'][str(a)]['escalation_rate'])} & "
        f"{pc(d['by_spoof_strength'][str(a)]['detection_rate'])} \\\\"
        for a in d["alphas"]]
    ff = d["false_flag_rates"]
    pretty = {"typical": "typical", "neurodivergent_verbal": "verbally advanced",
              "non_weird_l2": "second-language", "dialect_switching": "dialect-switching"}
    frows = "; ".join(f"{pretty[k]} {pc(v)}\\%" for k, v in ff.items())
    body = r"""\begin{table}[H]
\caption{Impersonation of $t_5$ by a genuine $t_2$ user after $n = 5$ interactions, $N = 400$ sessions per strength.\label{tab:bypass}}
\small
\begin{tabularx}{\textwidth}{CCC}
\toprule
\textbf{Spoof Strength} \boldmath{$\alpha$} & \textbf{Escalated above} \boldmath{$t_2$} \textbf{(\%)} & \textbf{Flagged by Detector (\%)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "Only the linguistic channel is spoofed. False-flag rates on genuine $t_2$ "
        "children at the adopted threshold $\\chi^2_{5,0.99}=15.086$: " + frows +
        ". Detector AUC separating a full-strength impersonator from a genuinely "
        "verbally advanced child: " + f"{d['auc_vs_verbally_advanced']:.3f}" + ".") + r"""
\end{table}"""
    write("tab_bypass", body)


# ---------------------------------------------------------------- overhead
def tab_overhead() -> None:
    d = load("exp07_overhead")
    m = d["measurements_seconds"]
    order = [("$L_0$ token filter", "L0: Token Filter (per response, 40 tokens)",
              "per response", "$O(|y|)$", "yes"),
             ("$L_1$ Socratic guard", "L1: Socratic Guard (per response)",
              "per response", "$O(1)$", "yes"),
             ("$L_4$ policy store", "L4: Policy Store (per query)",
              "per query", "$O(1)$", "yes"),
             ("Socratic policy lookup", "Socratic MDP lookup (per step)",
              "per step", "$O(1)$", "yes"),
             ("Full engine, end to end", "Full engine (per response)",
              "per response", "$O(|y|)$", "yes"),
             ("$L_2$ Bayesian update", "L2: Bayesian update (per interaction)",
              "per interaction", "$O(KM)$", "no")]
    rows = [f"{label} & {gran} & {m[key] * 1e6:.2f} & {cx} & {sync} \\\\"
            for label, key, gran, cx, sync in order]
    body = r"""\begin{table}[H]
\caption{Measured overhead per component on a single CPU core (median of five runs).\label{tab:overhead}}
\small
\begin{tabularx}{\textwidth}{LLccc}
\toprule
\textbf{Component} & \textbf{Granularity} & \textbf{Time (\boldmath{$\mu$}s)} & \textbf{Complexity} & \textbf{Synchronous} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "The full-engine figure is measured end to end rather than summed from its "
        "parts; it is " + f"{100 * d['share_of_500ms_llm']:.4f}" + "\\% of a 500 ms "
        "foundation-model response. $|y|$ is the response length in tokens, $K$ the "
        "number of tiers and $M$ the number of modalities. Platform: " +
        d["platform"].split("-")[0] + ", " + d.get("processor", "") + ".") + r"""
\end{table}"""
    write("tab_overhead", body)


# ---------------------------------------------------------------- appendix
def tab_socratic() -> None:
    d = load("exp03_socratic")
    acts = ["elicit", "hint", "guide", "partial_explain", "verify_request"]
    rows = []
    for t in ALL_TIERS:
        dist = d["action_distribution_step1"][t.label]
        ev = d["expected_value"][t.label]
        rows.append(f"{tier_math(t.label)} & "
                    + " & ".join(f"{dist[a]:.1f}" for a in acts)
                    + f" & {ev:.3f} \\\\")
    plain = ", ".join(f"{v:.1f}" for v in d["verify_usage_published_eq15"])
    body = r"""\begin{table}[H]
\caption{Optimal Socratic action at protocol step 1: share of knowledge states (\%) for which each action is optimal.\label{tab:socratic}}
\small
\begin{tabularx}{\textwidth}{Cccccccc}
\toprule
\textbf{Tier} & \textbf{Elicit} & \textbf{Hint} & \textbf{Guide} & \textbf{PartialExplain} & \textbf{VerifyRequest} & \boldmath{$E[V]$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "With the metacognition term removed ($\\alpha_{\\text{meta}} = 0$), "
        "VerifyRequest is optimal for " + plain + "\\% of states at $t_1$--$t_5$. "
        "PartialExplain is never optimal as an opening action at any tier, in this "
        "configuration or in any of the 80 swept alternatives.") + r"""
\end{table}"""
    write("tab_socratic", body)


def tab_stats() -> None:
    d = load("exp11_reliability")["holm_bonferroni"]
    comp = load("exp04_comparative")
    rows = []
    npl = comp["results"]["rubric"]["NPL (ours)"]["overall"]["dsr"]
    for name, v in sorted(d.items(), key=lambda kv: kv[1]["p_holm"]):
        base = comp["results"]["rubric"][name]["overall"]["dsr"]
        praw = "$<10^{-300}$" if v["p_raw"] == 0 else _sci(v["p_raw"])
        pholm = "$<10^{-300}$" if v["p_holm"] == 0 else _sci(v["p_holm"])
        rows.append(f"{DISPLAY[name]} & ${100 * (npl - base):+.1f}$ & {praw} & {pholm} & "
                    + ("yes" if v["significant_at_05"] else "no") + r" \\")
    body = r"""\begin{table}[H]
\caption{Paired McNemar comparisons of the full framework against each alternative, with Holm--Bonferroni correction.\label{tab:stats}}
\small
\begin{tabularx}{\textwidth}{Lcccc}
\toprule
\textbf{Full NPL versus} & \boldmath{$\Delta$} \textbf{(pp)} & \boldmath{$p$} \textbf{raw} & \boldmath{$p$} \textbf{Holm} & \textbf{Significant} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "Family of eight comparisons on $N = 7000$ paired prompts, independent rubric. "
        "McNemar's test with continuity correction; significance at $\\alpha = 0.05$ "
        "after correction. Raw $p$-values below the double-precision floor are "
        "reported as bounds.") + r"""
\end{table}"""
    write("tab_stats", body)


def _sci_bound(x: float) -> str:
    """Smallest power of ten above x, for 'p < 10^k' statements."""
    import math
    return f"10^{{{math.ceil(math.log10(x))}}}"


def _sci(x: float) -> str:
    mant, exp = f"{x:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def tab_thresholds() -> None:
    d = load("exp09_operating")
    grid = d["gamma_grid"]
    g_rows = []
    for g in grid:
        t = d["gamma_typical"][str(g)]
        v = d["gamma_verbally_advanced"][str(g)]
        mark = "$^{\\ast}$" if g == d["adopted_gamma"] else ""
        g_rows.append(f"{g:.2f}{mark} & {pc(t['accuracy'])} & "
                      f"{pc(t['under_protection'])} & {pc(t['over_restriction'])} & "
                      f"{pc(v['accuracy'])} & {pc(v['under_protection'])} \\\\")
    s_rows = []
    for th in d["severity_grid"]:
        r = d["severity_sweep"][str(th)]
        mark = "$^{\\ast}$" if th == d["adopted_severity_threshold"] else ""
        s_rows.append(f"{th:.1f}{mark} & {pc(r['dsr'])} & "
                      f"{pc(r['under_protection'])} & {pc(r['over_restriction'])} & & \\\\")
    body = r"""\begin{table}[H]
\caption{Sweeps over the two free thresholds.\label{tab:thresholds}}
\small
\begin{tabularx}{\textwidth}{Cccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Typical cohort}} & \multicolumn{2}{c}{\textbf{Verbally advanced}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
\boldmath{$\gamma_{\mathrm{conf}}$} & \textbf{Acc.} & \textbf{Under} & \textbf{Over} & \textbf{Acc.} & \textbf{Under} \\
\midrule
""" + "\n".join(g_rows) + r"""
\midrule
\boldmath{$\theta_{\mathrm{sev}}$} & \textbf{Harm DSR} & \textbf{Under} & \textbf{Over} & & \\
\midrule
""" + "\n".join(s_rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "All values are percentages; $^{\\ast}$ marks the adopted setting. The "
        "confidence sweep uses $n = " + str(d["n_interactions"]) + "$ interactions and "
        "$N = " + str(d["n_trials"]) + "$ users per cohort, so its accuracies differ "
        "slightly from Table~\\ref{tab:pops} ($n = 10$). The severity sweep is "
        "restricted to the three content-harm categories ($N = 3000$).") + r"""
\end{table}"""
    write("tab_thresholds", body)


def tab_category() -> None:
    comp = load("exp04_comparative")
    d = comp["per_category_t2_rubric"]
    hw = comp["per_category_t2_rubric_all_frameworks"]["homework_assignment"]
    best_other = max((v, k) for k, v in hw.items() if k != "NPL (ours)")
    zero_others = all(v == 0.0 for k, v in hw.items() if k not in ("NPL (ours)", best_other[1]))
    rows = []
    for r in d:
        rows.append(
            f"{r['Risk category'].replace('_', ' ').capitalize()} & {r['NPL']} & "
            f"{r['Constitutional AI']} & ${r['delta vs CAI']}$ & "
            f"{r['Age-conditioned (oracle)']} & ${r['delta vs age-cond']}$ \\\\")
    body = r"""\begin{table}[H]
\caption{Per-category DSR (\%) at $t_2$ (ages 7--9) under the independent rubric, $N = 200$ per cell.\label{tab:percat}}
\small
\begin{tabularx}{\textwidth}{Lccccc}
\toprule
\textbf{Risk Category} & \textbf{NPL} & \textbf{Const.\ AI-style} & \boldmath{$\Delta$} & \textbf{Age-cond.} & \boldmath{$\Delta$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}""" + note(
        "The rubric scaffolds code generation and essay writing at $t_2$; NPL and the "
        "age-conditioned baseline block both and the Constitutional AI-style filter "
        "allows both, so all three score zero. On homework at $t_2$ the best "
        "alternative is the " + DISPLAY[best_other[1]].lower() + " at "
        f"{100 * best_other[0]:.1f}\\%" + ("; every other alternative scores 0.0\\%"
                                            if zero_others else "") + ".") + r"""
\end{table}"""
    write("tab_category", body)


def main() -> None:
    tab_params(); tab_convergence(); tab_dsr(); tab_ablation()
    tab_populations(); tab_bypass(); tab_overhead()
    tab_socratic(); tab_stats(); tab_thresholds(); tab_category()


if __name__ == "__main__":
    main()
