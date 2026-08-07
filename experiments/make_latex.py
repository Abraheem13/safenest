"""Generate the manuscript's tables as LaTeX, straight from results/*.json.

Run after `run_all.py`. Writes `tables/tables.tex`, ready to \\input.

Generating rather than transcribing keeps every published number identical to
the number the code produced. Nothing here is typed twice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS, ROOT  # noqa: E402

TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)

TIERS = ["t1", "t2", "t3", "t4", "t5"]
TIER_AGES = {"t1": "3--6", "t2": "7--9", "t3": "10--12", "t4": "13--15", "t5": "16--17"}


def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/run_all.py first")
    return json.loads(path.read_text())


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def table8(d: dict) -> str:
    ms = d["milestones"]
    head = " & ".join(f"$n={m}$" for m in ms)
    rows = []
    for t in TIERS:
        acc = d["accuracy"][t]
        first = d["first_milestone_above_90"][t]
        cells = []
        for m in ms:
            v = f"{100 * acc[str(m)]:.1f}"
            cells.append(rf"\textbf{{{v}}}" if m == first else v)
        rows.append(f"$t_{t[1]}$ ({TIER_AGES[t]}) & " + " & ".join(cells) + r" \\")
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Bayesian estimator classification accuracy (\%) at selected interaction
milestones ($N={d['n_trials']}$ trials per tier, $\varepsilon=1.0$ corpus-level
differential privacy). Bold marks the first milestone exceeding 90\%.
Regenerated from \texttt{{experiments/exp01\_convergence.py}}.}}
\label{{tab:convergence}}
\begin{{tabular}}{{l{'c' * len(ms)}}}
\toprule
Tier & {head} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table9(d: dict) -> str:
    kl = d["kl_matrix"]
    rows = []
    for i, t in enumerate(TIERS):
        cells = []
        for j in range(len(TIERS)):
            if i == j:
                cells.append("---")
            else:
                v = f"{kl[i][j]:.2f}"
                cells.append(rf"\textbf{{{v}}}" if kl[i][j] == min(
                    min(kl[k][k + 1], kl[k + 1][k]) for k in range(len(TIERS) - 1)
                ) else v)
        rows.append(f"$t_{t[1]}$ & " + " & ".join(cells) + r" \\")
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Pairwise KL divergence (nats) across the five-dimensional linguistic
feature space. The minimum adjacent-tier divergence $D_{{\min}}={d['d_min']:.2f}$
(bold, at the {esc(d['hardest_boundary'])} boundary) governs worst-case
convergence; the mean pairwise divergence is {d['mean_pairwise_kl']:.2f} nats.
Regenerated from \texttt{{experiments/exp02\_separability.py}}.}}
\label{{tab:kl}}
\begin{{tabular}}{{l{'c' * len(TIERS)}}}
\toprule
 & {' & '.join(f'$t_{t[1]}$' for t in TIERS)} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table10(d: dict) -> str:
    actions = ["elicit", "hint", "guide", "partial_explain", "verify_request"]
    names = ["Elicit", "Hint", "Guide", "P.\\,Explain", "Verify"]
    rows = []
    for t in TIERS:
        dist = d["action_distribution_step1"][t]
        cells = [f"{dist[a]:.1f}" for a in actions]
        rows.append(
            f"$t_{t[1]}$ ({TIER_AGES[t]}) & " + " & ".join(cells)
            + f" & {d['expected_value'][t]:.3f}" + r" \\"
        )
    p = d["reward_params"]
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Optimal Socratic action distribution at protocol step $p=1$, computed
by backward induction. Each cell reports the percentage of knowledge states
$q\in[0,1]$ for which the action is optimal. Reward weights:
$\alpha_{{\text{{learn}}}}={p['alpha_learn']}$,
$\alpha_{{\text{{reveal}}}}={p['alpha_reveal']}$,
$\alpha_{{\text{{frust}}}}={p['alpha_frust']}$,
$\alpha_{{\text{{meta}}}}={p['alpha_meta']}$,
$\gamma={p['discount']}$, $H={p['horizon']}$, $\sigma_{{\text{{ZPD}}}}={p['sigma_zpd']}$.
Regenerated from \texttt{{experiments/exp03\_socratic.py}}.}}
\label{{tab:socratic}}
\begin{{tabular}}{{l{'c' * len(actions)}c}}
\toprule
Tier & {' & '.join(names)} & $\mathbb{{E}}[V]$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table14(d: dict, labeller: str, label: str, caption: str) -> str:
    res = d["results"][labeller]
    rows = []
    for name, r in res.items():
        cells = [f"{100 * r['by_tier'][t]['dsr']:.1f}" for t in TIERS]
        o = r["overall"]
        mean = f"{100 * o['dsr']:.1f}"
        if name.startswith("NPL (ours)"):
            mean = rf"\textbf{{{mean}}}"
        rows.append(
            f"{esc(name)} & " + " & ".join(cells)
            + f" & {mean} & [{100 * o['dsr_ci_low']:.1f}, {100 * o['dsr_ci_high']:.1f}]"
            + f" & {100 * o['under_protection']:.1f} & {100 * o['over_restriction']:.1f}"
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
\centering
\small
\caption{{{caption}}}
\label{{tab:{label}}}
\begin{{tabular}}{{l{'c' * len(TIERS)}ccc c}}
\toprule
Framework & {' & '.join(f'$t_{t[1]}$' for t in TIERS)} & Mean & 95\% CI & Under & Over \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table12(d: dict) -> str:
    eps = d["epsilons"]
    corpus = " & ".join(f"{100 * d['accuracy_corpus_dp'][str(e)]:.1f}" for e in eps)
    local = " & ".join(f"{100 * d['accuracy_local_dp'][str(e)]:.1f}" for e in eps)
    header = " & ".join(
        (rf"\textbf{{{e:g}}}" if e == 1.0 else f"{e:g}") for e in eps
    )
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Classification accuracy (\%) as a function of the privacy budget
$\varepsilon$ after $n={d['n_interactions']}$ interactions ($N={d['n_trials']}$).
Corpus-level DP protects one child in the calibration corpora; local DP protects
one interaction of the live user. Chance accuracy is 20\%. The adopted operating
point is $\varepsilon=1.0$ (bold). Regenerated from
\texttt{{experiments/exp05\_privacy.py}}.}}
\label{{tab:privacy}}
\begin{{tabular}}{{l{'c' * len(eps)}}}
\toprule
$\varepsilon$ & {header} \\
\midrule
Corpus-level DP & {corpus} \\
Local DP at inference & {local} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table13(d: dict) -> str:
    rows = []
    for name, sec in d["measurements_seconds"].items():
        if sec < 1e-6:
            lat = f"{sec * 1e9:.0f}\\,ns"
        elif sec < 1e-3:
            lat = f"{sec * 1e6:.1f}\\,$\\mu$s"
        else:
            lat = f"{sec * 1e3:.2f}\\,ms"
        rows.append(f"{esc(name)} & {lat} \\\\")
    sync = d["synchronous_per_response_seconds"]
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Measured per-operation latency (single core, {esc(d['platform'])}).
Total synchronous per-response overhead is {sync * 1e6:.1f}\,$\mu$s,
or {100 * d['share_of_500ms_llm']:.4f}\% of a 500\,ms foundation-model response.
Regenerated from \texttt{{experiments/exp07\_overhead.py}}.}}
\label{{tab:overhead}}
\begin{{tabular}}{{lc}}
\toprule
Component & Latency \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_populations(d: dict) -> str:
    rows = []
    for r in d["population_table"]:
        rows.append(
            f"{esc(r['Population'])} & {r['Acc (%)']} & {r['Under-prot (%)']} & "
            f"{r['Over-prot (%)']} & {r['Acc w/ flag (%)']} & {r['Under w/ flag (%)']} \\\\"
        )
    v = d["verbal_case"]
    return rf"""
\begin{{table}}[t]
\centering
\small
\caption{{Tier-estimation performance by population ($n={d['n_interactions']}$
interactions). Under-protection means the child was assigned a \emph{{less}}
restrictive tier than warranted. The discordance flag holds a user at an
externally attested tier when the linguistic estimate differs by two tiers or
more: it reduces over-promotion of a verbally advanced 8-year-old from
{100 * v['assigned_above_t2_unflagged']:.1f}\% to
{100 * v['assigned_above_t2_flagged']:.1f}\% of sessions, but does not fire on
one-tier mismatches. Regenerated from \texttt{{experiments/exp06\_populations.py}}.}}
\label{{tab:populations}}
\begin{{tabular}}{{lccccc}}
\toprule
Population & Acc. & Under & Over & Acc.\ w/flag & Under w/flag \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    e1, e2 = load("exp01_convergence"), load("exp02_separability")
    e3, e4 = load("exp03_socratic"), load("exp04_comparative")
    e5, e6, e7 = load("exp05_privacy"), load("exp06_populations"), load("exp07_overhead")

    agree = e4["labeller_agreement"]
    parts = [
        "% Generated by experiments/make_latex.py -- do not edit by hand.",
        "% Regenerate with: python3 experiments/run_all.py && python3 experiments/make_latex.py",
        table8(e1),
        table9(e2),
        table10(e3),
        table12(e5),
        table13(e7),
        table_populations(e6),
        table14(
            e4, "rubric", "dsr_rubric",
            rf"Developmental Safety Rate (\%) by tier and framework, scored against "
            rf"the \emph{{independent}} rubric ground truth (age, regulatory "
            rf"instruments and Piagetian criteria; it does not read the "
            rf"feature-gating matrix). $N=7{{,}}000$ prompts. Under- and "
            rf"over-restriction are the two error types and sum with DSR to 100\%. "
            rf"Regenerated from \texttt{{experiments/exp04\_comparative.py}}.",
        ),
        table14(
            e4, "matrix", "dsr_matrix",
            rf"Developmental Safety Rate (\%) scored against the feature-gating "
            rf"matrix, i.e.\ the specification under evaluation. Reported for "
            rf"contrast only: this ground truth is circular and inflates NPL's "
            rf"margin. The two labellers agree on {100 * agree['exact_agreement']:.1f}\% "
            rf"of prompts (Cohen's $\kappa={agree['cohens_kappa']:.3f}$).",
        ),
    ]
    out = TABLES / "tables.tex"
    out.write_text("\n".join(parts))
    print(f"wrote {out.relative_to(ROOT)} ({len(parts) - 2} tables)")
    print("\\input{tables} in your manuscript, or paste the individual environments.")


if __name__ == "__main__":
    main()
