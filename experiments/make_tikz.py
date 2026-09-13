"""Emit every manuscript figure as native TikZ/pgfplots source.

Figures compiled by the document itself inherit its fonts and metrics exactly,
which no external raster can do. Each file below is written from `results/*.json`
or directly from the specification modules, so a change to the code or the
results propagates into the typeset figure on the next build.

    python3 experiments/make_tikz.py        # writes tex/fig*.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import ROOT, RESULTS  # noqa: E402
from safenest.lattice import FEATURE_GATING, Access, Capability  # noqa: E402
from safenest.tiers import ALL_TIERS, TIER_SPECS  # noqa: E402
from safenest.verification import abstraction_size  # noqa: E402

TEX = ROOT / "tex"
TEX.mkdir(exist_ok=True)

#: Restrained academic palette: black, three greys, one accent for the framework.
ACCENT = "npred"

#: Baselines are re-implementations of documented decision rules, and the
#: figures say so.
DISPLAY = {
    "NeMo Guardrails": "NeMo Guardrails-style",
    "LlamaFirewall": "LlamaFirewall-style",
    "Llama Guard": "Llama Guard-style",
    "Constitutional AI": "Constitutional AI-style",
    "COPPA binary rule": "COPPA binary rule",
    "Child-safety classifier": "Child-safety classifier",
    "Age-conditioned (oracle)": "Age-conditioned (oracle band)",
    "NPL (ours)": "NPL (full)",
}


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def write(name: str, body: str) -> None:
    (TEX / f"{name}.tex").write_text(body.rstrip() + "\n")
    print(f"  wrote tex/{name}.tex")


# ------------------------------------------------------------------ Figure 1
def fig_pipeline() -> None:
    """The measurement protocol actually run, with its three real gates.

    Every box corresponds to a step in `experiments/run_all.py` and every
    decision to a check that can fail: `validate_lattice` raises on a
    non-monotone policy edit, `test_evaluation` asserts the two labellers
    disagree enough to be independent, and the invariant suite rejects a
    specification whose decisions are not monotone in tier.
    """
    body = r"""
\begin{tikzpicture}[
  x=1cm, y=1cm,
  box/.style={draw, thin, rectangle, text width=34mm, align=center,
              inner sep=3.5pt, minimum height=8.5mm, font=\scriptsize,
              execute at begin node={\hyphenpenalty=10000\relax}},
  dec/.style={draw, thin, diamond, aspect=2.4, align=center,
              inner sep=1.2pt, font=\scriptsize},
  term/.style={draw, thin, rectangle, rounded corners=2pt, fill=black!8,
               align=center, inner sep=3.5pt, font=\scriptsize, text width=28mm,
               execute at begin node={\hyphenpenalty=10000\relax}},
  hd/.style={font=\scriptsize\bfseries},
  ar/.style={-latex, thin},
  lb/.style={font=\tiny, inner sep=1.5pt},
]
% ================= 1. corpus and ground truth =================
\node[hd] at (0,0) {1.\ Corpus and ground truth};
\node[box] (a1) at (0,-1.00) {seeded generator\\7 risk categories, 5 tiers,\\200 prompts each};
\node[box] (a2) at (0,-2.55) {7{,}000 records; label twice,\\gating matrix and rubric};
\node[dec] (a3) at (0,-4.30) {$\kappa<0.9$?};
\node[term] (a4) at (0,-5.90) {labellers not\\independent};
\draw[ar] (a1) -- (a2);
\draw[ar] (a2) -- (a3);
\draw[ar] (a3) -- node[lb,right] {no} (a4);

% ================= 2. specification under test =================
\node[hd] at (6.0,0) {2.\ Specification under test};
\node[box] (b1) at (6.0,-1.00) {monotone tier policy\\$a(c,t)$, nine capabilities};
\node[dec] (b2) at (6.0,-2.75) {monotone at\\compile time?};
\node[term] (b3) at (6.0,-4.35) {reject policy edit};
\node[box] (b4) at (6.0,-5.90) {enumerate """ + f"{abstraction_size():,}".replace(",", "{,}") + r""" region-\\complete responses $\times$ 5 tiers};
\node[dec] (b5) at (6.0,-7.65) {invariants hold?};
\node[term] (b6) at (6.0,-9.20) {specification defect};
\draw[ar] (b1) -- (b2);
\draw[ar] (b2) -- node[lb,right] {no} (b3);
\draw[ar] (b4) -- (b5);
\draw[ar] (b5) -- node[lb,right] {no} (b6);
% monotone-yes bypasses the failure terminal on the left
\draw[ar] (b2.west) -- ++(-1.05,0) node[lb,above right] {yes} |- (b4.west);
% corpus feeds the specification through the gap between the columns
\draw[ar] (a3.east) -- node[lb,above] {yes} ++(1.20,0) |- (b1.west);

% ================= 3. measurement =================
\node[hd] at (12.0,0) {3.\ Measurement};
\node[box] (c1) at (12.0,-1.00) {nine frameworks and\\five ablation variants};
\node[box] (c2) at (12.0,-2.55) {both ground truths;\\20 seeds, Wilson CIs, Holm};
\node[dec] (c3) at (12.0,-4.30) {at the achievable\\ceiling?};
\node[term] (c4) at (12.0,-5.90) {report recoverable\\headroom};
\node[term, fill=black!75, text=white] (c5) at (12.0,-7.65) {DSR verdict};
\draw[ar] (c1) -- (c2);
\draw[ar] (c2) -- (c3);
\draw[ar] (c3) -- node[lb,right] {no} (c4);
\draw[ar] (c4) -- (c5);
\draw[ar] (c3.east) -- node[lb,above] {yes} ++(0.55,0) |- (c5.east);
% a validated specification feeds measurement through the right-hand gap
\draw[ar] (b5.east) -- node[lb,above] {yes} ++(1.35,0) |- (c1.west);
\end{tikzpicture}
"""
    write("fig1_pipeline", body)


# ------------------------------------------------------------------ Figure 2
def fig_deployment() -> None:
    """Deployment and trust boundary, in the style of a systems diagram."""
    body = r"""
\begin{tikzpicture}[
  box/.style={draw, thin, rectangle, align=center, inner sep=5pt,
              minimum height=9mm, font=\small},
  ar/.style={-latex, thin},
  obs/.style={-latex, thin, dashed},
  lab/.style={font=\scriptsize, inner sep=2pt},
]
\node[box, text width=20mm] (c)  at (0.7,1.55) {child};
\node[box, text width=20mm] (s)  at (0.7,0.35) {signals $x$};
\node[box, text width=20mm] (dv) at (0.7,-0.95) {device and\\account flags};

\node[box, text width=27mm] (est) at (4.7,0.35) {Bayesian tier\\estimator};
\node[box, text width=36mm] (pol) at (9.4,0.35) {Nested Policy Engine\\$L_0,\ldots,L_4$};
\node[box, text width=36mm] (llm) at (9.4,2.15) {foundation model $\theta$\\(unmodified)};
\node[box, text width=36mm] (soc) at (9.4,-1.45) {Socratic engine $\pi^{*}$};
\node[box, text width=22mm] (out) at (14.1,0.35) {response $y$};

\draw[ar] (c) -- (s);
\draw[ar] (s) -- node[lab, above] {LLRs} (est);
\draw[ar] (dv.east) -- ++(0.75,0) |- (est.west);
\draw[ar] (est) -- node[lab, above] {tier $\hat{\tau}$} (pol);
\draw[ar] (llm) -- node[lab, right] {candidate $y$} (pol);
\draw[ar] (pol) -- node[lab, right] {\textsc{modify}} (soc);
\draw[ar] (pol) -- node[lab, above] {\textsc{accept}} (out);
\draw[ar] (soc.east) -- ++(1.95,0) node[lab, above, pos=0.5] {scaffold} |- (out.south);

% trust boundary around the signal enclave
\draw[dashed, thin, black!60] (-0.55,0.95) rectangle (2.05,-1.75);
\node[lab, anchor=north west, text=black!70] at (-0.55,-1.75) {signal enclave (raw features stay inside)};
\end{tikzpicture}
"""
    write("fig2_deployment", body)


# ------------------------------------------------------------------ Figure 3
def fig_layers() -> None:
    """Five layers and the tier chain."""
    rows = []
    for i, tier in enumerate(reversed(ALL_TIERS)):
        idx = int(tier) - 1
        spec = TIER_SPECS[tier]
        n_open = sum(1 for c in Capability
                     if FEATURE_GATING[c][tier] is not Access.BLOCKED)
        shade = [70, 55, 40, 25, 12][idx]
        txt = "white" if shade >= 45 else "black"
        rows.append(
            f"\\node[tierbox, fill=black!{shade}, text={txt}] (t{idx+1}) at "
            f"(12.0,{-0.85 * i - 0.9:.2f}) "
            f"{{$t_{idx+1}$\\quad ages {spec.age_low}--{spec.age_high}"
            f"\\quad {n_open}/9 open}};"
        )
    chain = "\n".join(rows)
    layers = [
        ("L_0", "Token Filter", "per token", "$10^{3}$\\,Hz", "synchronous"),
        ("L_1", "Socratic Guard", "per response", "$1$\\,Hz", "synchronous"),
        ("L_2", "Tier Refiner", "per session", "$1.7{\\times}10^{-3}$\\,Hz", "asynchronous"),
        ("L_3", "Memory Layer", "cross-session", "$1.2{\\times}10^{-5}$\\,Hz", "asynchronous"),
        ("L_4", "Policy Store", "per revision", "$3.2{\\times}10^{-8}$\\,Hz", "synchronous"),
    ]
    lrows = []
    for i, (tag, name, gran, freq, sync) in enumerate(layers):
        y = -0.85 * i - 0.9
        style = "" if sync == "synchronous" else ", text=black!55"
        lrows.append(
            f"\\node[lbl] at (0,{y:.2f}) {{${tag}$}};\n"
            f"\\node[lname{style}] at (1.9,{y:.2f}) {{{name}}};\n"
            f"\\node[lcell{style}] at (4.1,{y:.2f}) {{{gran}}};\n"
            f"\\node[lcell{style}] at (5.9,{y:.2f}) {{{freq}}};\n"
            f"\\node[font=\\scriptsize{style}, anchor=west] at (7.0,{y:.2f}) {{{sync}}};"
        )
    body = r"""
\begin{tikzpicture}[
  lbl/.style={draw, thin, rectangle, minimum width=7mm, minimum height=6mm,
              fill=black!85, text=white, font=\small},
  lname/.style={draw, thin, rectangle, minimum width=26mm, minimum height=6mm,
                font=\small},
  lcell/.style={draw, thin, rectangle, minimum width=20mm, minimum height=6mm,
                font=\scriptsize},
  tierbox/.style={draw, thin, rectangle, minimum width=52mm, minimum height=6mm,
                  font=\small},
]
\node[font=\small\bfseries, anchor=west] at (-0.45,0) {(a) Nested Policy Engine};
\node[font=\small\bfseries, anchor=west] at (9.35,0) {(b) Constraint lattice};
""" + "\n".join(lrows) + "\n" + chain + r"""
\node[font=\scriptsize, anchor=west, text width=80mm] at (-0.45,-5.5)
  {Decisions compose by conjunction: the most severe wins, so no layer can relax
   another. Only the three synchronous layers gate a response.};
\node[font=\scriptsize, anchor=west, text width=58mm] at (9.35,-5.5)
  {Unblocked sets are nested; $L_4$ rejects any policy edit that
   breaks monotonicity of $a(c,t)$ in $t$.};
\end{tikzpicture}
"""
    write("fig3_layers", body)


# ------------------------------------------------------------------ Figure 4
def fig_gating() -> None:
    """Feature-gating matrix, drawn from FEATURE_GATING itself."""
    marks = {Access.BLOCKED: ("black!72", "white", "B"),
             Access.SOCRATIC: ("black!42", "white", "S"),
             Access.LIMITED: ("black!18", "black", "L"),
             Access.AVAILABLE: ("white", "black", "A")}
    caps = list(Capability)
    cells = []
    for j, cap in enumerate(caps):
        y = -0.62 * j
        cells.append(
            f"\\node[font=\\scriptsize, anchor=east] at (-0.60,{y:.2f}) "
            f"{{{cap.value.replace('_', ' ')}}};")
        for i, tier in enumerate(ALL_TIERS):
            fill, tc, ch = marks[FEATURE_GATING[cap][tier]]
            cells.append(
                f"\\node[cell, fill={fill}, text={tc}] at ({0.95 * i:.2f},{y:.2f}) {{{ch}}};")
    heads = "\n".join(
        f"\\node[font=\\scriptsize] at ({0.95 * i:.2f},0.70) "
        f"{{$t_{i+1}$}};\n"
        f"\\node[font=\\scriptsize] at ({0.95 * i:.2f},0.36) "
        f"{{\\tiny {TIER_SPECS[t].age_low}--{TIER_SPECS[t].age_high}}};"
        for i, t in enumerate(ALL_TIERS))
    body = r"""
\begin{tikzpicture}[
  cell/.style={draw, thin, rectangle, minimum width=8.5mm, minimum height=5.4mm,
               font=\scriptsize},
]
""" + heads + "\n" + "\n".join(cells) + r"""
\node[font=\scriptsize, anchor=west] at (-4.10,-6.05)
  {\textbf{B} blocked (refused)\quad \textbf{S} Socratic (scaffolded)\quad
   \textbf{L} limited (monitored)\quad \textbf{A} available};
\end{tikzpicture}
"""
    write("fig4_gating", body)


# ------------------------------------------------------------------ Figure 5
def fig_dsr() -> None:
    """Grouped DSR by tier, with the Wilson intervals actually drawn."""
    d = load("exp04_comparative")["results"]["rubric"]
    order = ["NeMo Guardrails", "LlamaFirewall", "Llama Guard", "Constitutional AI",
             "COPPA binary rule", "Child-safety classifier",
             "Age-conditioned (oracle)", "NPL (ours)"]
    shades = ["black!8", "black!20", "black!32", "black!44",
              "black!56", "black!68", "black!82", ACCENT]
    plots = []
    for name, sh in zip(order, shades):
        bt = d[name]["by_tier"]
        coords = []
        for i in range(1, 6):
            cell = bt[f"t{i}"]
            v = 100 * cell["dsr"]
            # Wilson intervals are near-symmetric at these n; half-width is
            # plotted so the bar carries its own uncertainty.
            half = 100 * (cell["dsr_ci_high"] - cell["dsr_ci_low"]) / 2
            coords.append(f"({i},{v:.1f}) +- (0,{half:.2f})")
        plots.append(
            "\\addplot[ybar, fill=" + sh + ", draw=black, line width=0.3pt,\n"
            "  error bars/.cd, y dir=both, y explicit,\n"
            "  error bar style={line width=0.3pt, black!55},\n"
            "  error mark options={line width=0.3pt, mark size=0.6pt, black!55}]\n"
            "  coordinates {" + " ".join(coords) + "};\n"
            "\\addlegendentry{" + DISPLAY[name] + "}")
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  width=\linewidth, height=64mm,
  ybar=0.35pt, bar width=3.4pt,
  enlarge x limits=0.11,
  ymin=0, ymax=100,
  xtick={1,2,3,4,5},
  xticklabels={$t_1$,$t_2$,$t_3$,$t_4$,$t_5$},
  xlabel={Developmental tier}, ylabel={Developmental Safety Rate (\%)},
  legend style={at={(0.5,1.03)}, anchor=south, legend columns=4,
                draw=none, font=\scriptsize, column sep=0.7ex,
                /tikz/every even column/.append style={column sep=0.5ex}},
  legend image code/.code={\draw[#1] (0cm,-0.08cm) rectangle (0.26cm,0.14cm);},
  tick label style={font=\scriptsize},
  label style={font=\small},
  ymajorgrids, grid style={black!12, line width=0.3pt},
  axis lines*=left,
]
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    write("fig5_dsr", body)


# ------------------------------------------------------------------ Figure 6
def fig_ceiling() -> None:
    d = load("exp08_ceiling")
    npl, ceil = 100 * d["npl_dsr"], 100 * d["ceiling_cv"]
    cells = d["cell_losses"][:6]
    short = {"code_generation": "code generation",
             "essay_creative_writing": "essay writing",
             "crisis_self_harm": "crisis", "open_ended_chat": "open chat",
             "homework_assignment": "homework",
             "substance_body_image": "substance",
             "age_inappropriate_content": "age-inappropriate"}
    labels = ",".join(
        f"$t_{c['tier'][1]}$ {short.get(c['category'], c['category'])}" for c in cells)
    ceil_c = " ".join(f"({c['ceiling']:.1f},{i})" for i, c in enumerate(cells))
    npl_c = " ".join(f"({c['npl']:.1f},{i})" for i, c in enumerate(cells))
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  name=left, width=0.36\linewidth, height=46mm,
  xbar, bar width=6pt, xmin=0, xmax=118, enlarge y limits=0.25,
  ytick={0,1,2},
  yticklabels={{recoverable},{NPL},{ceiling}},
  xlabel={Developmental Safety Rate (\%)},
  nodes near coords, nodes near coords style={font=\scriptsize},
  point meta=explicit symbolic,
  tick label style={font=\scriptsize}, label style={font=\small},
  xmajorgrids, grid style={black!12, line width=0.3pt}, axis lines*=left,
  title style={font=\small, align=left}, title={(a) Reachable headroom},
]
\addplot[fill=black!18, draw=black, line width=0.3pt] coordinates {
""" + f"  (%.1f,0) [%.1f]\n  (%.1f,1) [%.1f]\n  (%.1f,2) [%.1f]\n" % (
        ceil - npl, ceil - npl, npl, npl, ceil, ceil) + r"""};
\end{axis}
\begin{axis}[
  name=right, at={(left.east)}, anchor=west, xshift=36mm,
  width=0.42\linewidth, height=46mm,
  xbar, bar width=4pt, xmin=0, xmax=105, enlarge y limits=0.12,
  ytick={0,1,2,3,4,5}, yticklabels={""" + labels + r"""},
  y dir=reverse,
  xlabel={DSR within cell (\%)},
  tick label style={font=\scriptsize}, label style={font=\small},
  xmajorgrids, grid style={black!12, line width=0.3pt}, axis lines*=left,
  legend style={at={(1.03,1.0)}, anchor=north west, draw=none,
                font=\scriptsize},
  title style={font=\small, align=left},
  title={(b) Where the recoverable error sits},
]
\addplot[fill=black!12, draw=black, line width=0.3pt] coordinates {""" + ceil_c + r"""};
\addlegendentry{ceiling}
\addplot[fill=""" + ACCENT + r""", draw=black, line width=0.3pt] coordinates {""" + npl_c + r"""};
\addlegendentry{NPL}
\end{axis}
\end{tikzpicture}
"""
    write("fig6_ceiling", body)


# ------------------------------------------------------------------ Figure 7
def fig_privacy() -> None:
    d = load("exp05_privacy")
    eps = d["epsilons"]
    corpus = " ".join(
        f"({e},{100 * d['accuracy_corpus_dp'][str(e)]:.1f}) +- "
        f"(0,{100 * d['accuracy_corpus_dp_sd_over_releases'][str(e)]:.1f})" for e in eps)
    local = " ".join(f"({e},{100 * d['accuracy_local_dp'][str(e)]:.1f})" for e in eps)
    ref = 100 * d["accuracy_nonprivate_release"]
    at1 = 100 * d["accuracy_corpus_dp"]["1.0"]
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  width=0.86\linewidth, height=56mm,
  xmode=log, log basis x=10,
  xmin=0.08, xmax=130, ymin=0, ymax=105,
  xlabel={Privacy budget $\varepsilon$ (log scale)},
  ylabel={Tier accuracy at $n=10$ (\%)},
  tick label style={font=\scriptsize}, label style={font=\small},
  legend style={at={(0.5,-0.30)}, anchor=north, draw=none, font=\scriptsize,
                legend columns=3, /tikz/every even column/.append style={column sep=1.2ex}},
  xmajorgrids, ymajorgrids, grid style={black!12, line width=0.3pt},
  axis lines*=left,
]
\addplot[black!45, dashed, line width=0.6pt]
  coordinates {(0.08,""" + f"{ref:.1f}" + r""") (130,""" + f"{ref:.1f}" + r""")};
\addlegendentry{non-private release (""" + f"{ref:.1f}" + r"""\%)}
\addplot[mark=*, mark size=1.5pt, black, line width=0.8pt,
  error bars/.cd, y dir=both, y explicit,
  error bar style={line width=0.4pt, black!70}]
  coordinates {""" + corpus + r"""};
\addlegendentry{corpus-level DP (mean $\pm$ SD, 20 releases)}
\addplot[mark=square*, mark size=1.5pt, """ + ACCENT + r""", dashed, line width=0.8pt]
  coordinates {""" + local + r"""};
\addlegendentry{local DP on the live child}
\addplot[black!45, dotted, line width=0.7pt, forget plot]
  coordinates {(0.08,20) (130,20)};
\node[font=\scriptsize, anchor=south east, text=black!55] at (axis cs:120,21)
  {chance (20\%)};
\node[font=\scriptsize, anchor=north west] at (axis cs:1.25,70)
  {adopted $\varepsilon=1$: """ + f"{at1:.1f}" + r"""\%};
\draw[-latex, thin] (axis cs:1.3,70.5) -- (axis cs:1.02,""" + f"{at1 - 3:.1f}" + r""");
\end{axis}
\end{tikzpicture}
"""
    write("fig7_privacy", body)


# ------------------------------------------------------------------ Figure 8
def fig_bypass() -> None:
    d = load("exp10_bypass")

    def curve(points: list[dict]) -> str:
        pts = sorted({(round(p["fpr"], 4), round(p["tpr"], 4)) for p in points})
        pts = [(0.0, 0.0)] + pts + [(1.0, 1.0)]
        return " ".join(f"({x:.4f},{y:.4f})" for x, y in pts)

    typ = curve(d["roc"]["1.0"])
    verb = curve(d["roc_vs_verbally_advanced"])
    op = d["operating_points_alpha1"]
    op_t = f"({op['typical']['fpr']:.4f},{op['typical']['tpr']:.4f})"
    op_v = f"({op['neurodivergent_verbal']['fpr']:.4f},{op['neurodivergent_verbal']['tpr']:.4f})"
    esc = " ".join(
        f"({a},{100 * d['by_spoof_strength'][str(a)]['escalation_rate']:.1f})"
        for a in d["alphas"])
    det = " ".join(
        f"({a},{100 * d['by_spoof_strength'][str(a)]['detection_rate']:.1f})"
        for a in d["alphas"])
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  name=l, width=0.47\linewidth, height=52mm,
  xmin=0, xmax=1, ymin=0, ymax=1,
  xlabel={False-flag rate, genuine $t_2$ children},
  ylabel={Impersonators flagged ($\alpha=1$)},
  tick label style={font=\scriptsize}, label style={font=\small},
  legend style={at={(0.5,-0.30)}, anchor=north, draw=none, font=\scriptsize,
                legend columns=1},
  xmajorgrids, ymajorgrids, grid style={black!12, line width=0.3pt},
  axis lines*=left, title style={font=\small},
  title={(a) Detector ROC},
]
\addplot[black, line width=0.9pt] coordinates {""" + typ + r"""};
\addlegendentry{vs typical child (AUC """ + f"{d['auc_vs_typical']['1.0']:.2f}" + r""")}
\addplot[""" + ACCENT + r""", dashed, line width=0.9pt] coordinates {""" + verb + r"""};
\addlegendentry{vs verbally advanced child (AUC """ + f"{d['auc_vs_verbally_advanced']:.2f}" + r""")}
\addplot[black!45, dotted, line width=0.8pt, forget plot] coordinates {(0,0) (1,1)};
\addplot[only marks, mark=*, mark size=2pt, black, forget plot] coordinates {""" + op_t + r"""};
\addplot[only marks, mark=*, mark size=2pt, """ + ACCENT + r""", forget plot] coordinates {""" + op_v + r"""};
\node[font=\scriptsize, anchor=north west] at (axis cs:""" + f"{op['neurodivergent_verbal']['fpr'] + 0.03:.3f},{op['neurodivergent_verbal']['tpr'] - 0.02:.3f}" + r""")
  {adopted threshold};
\end{axis}
\begin{axis}[
  name=r, at={(l.east)}, anchor=west, xshift=16mm,
  width=0.47\linewidth, height=52mm,
  xmin=-0.05, xmax=1.05, ymin=0, ymax=105,
  xlabel={Spoof strength $\alpha$}, ylabel={Sessions (\%)},
  tick label style={font=\scriptsize}, label style={font=\small},
  legend style={at={(0.5,-0.30)}, anchor=north, draw=none, font=\scriptsize,
                legend columns=1},
  xmajorgrids, ymajorgrids, grid style={black!12, line width=0.3pt},
  axis lines*=left, title style={font=\small},
  title={(b) Escalation outruns detection},
]
\addplot[mark=*, mark size=1.5pt, """ + ACCENT + r""", line width=0.9pt]
  coordinates {""" + esc + r"""};
\addlegendentry{escalated above $t_2$}
\addplot[mark=square*, mark size=1.5pt, black, dashed, line width=0.9pt]
  coordinates {""" + det + r"""};
\addlegendentry{flagged by detector}
\end{axis}
\end{tikzpicture}
"""
    write("fig8_bypass", body)


# ------------------------------------------------------------------ Figure 9
def fig_calibration() -> None:
    d = load("exp11_reliability")
    bins = d["calibration_bins"]
    pts = " ".join(
        f"({100 * b['mean_confidence']:.1f},{100 * b['observed_accuracy']:.1f})"
        for b in bins)
    gam = load("exp09_operating")
    grid = gam["gamma_grid"]
    typ_u = " ".join(
        f"({g},{100 * gam['gamma_typical'][str(g)]['over_restriction']:.1f})"
        for g in grid)
    atyp = " ".join(
        f"({g},{100 * gam['gamma_verbally_advanced'][str(g)]['under_protection']:.1f})"
        for g in grid)
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  name=l, width=0.45\linewidth, height=46mm,
  xmin=40, xmax=105, ymin=30, ymax=105,
  xlabel={Mean posterior confidence (\%)},
  ylabel={Observed accuracy (\%)},
  tick label style={font=\scriptsize}, label style={font=\small},
  xmajorgrids, ymajorgrids, grid style={black!12, line width=0.3pt},
  axis lines*=left, title style={font=\small, align=left},
  title={(a) Posterior calibration},
  legend style={at={(0.97,0.05)}, anchor=south east, draw=none, font=\scriptsize},
]
\addplot[black!45, dotted, line width=0.8pt] coordinates {(40,40) (105,105)};
\addlegendentry{perfect}
\addplot[mark=*, mark size=1.6pt, black, line width=0.9pt]
  coordinates {""" + pts + r"""};
\addlegendentry{observed}
\node[font=\scriptsize, anchor=west] at (axis cs:44,88)
  {ECE """ + f"{d['expected_calibration_error']:.4f}" + r"""};
\end{axis}
\begin{axis}[
  name=r, at={(l.east)}, anchor=west, xshift=14mm,
  width=0.45\linewidth, height=46mm,
  xmin=0.15, xmax=1.0, ymin=0, ymax=75,
  xlabel={Confidence threshold $\gamma_{\mathrm{conf}}$},
  ylabel={Error rate (\%)},
  tick label style={font=\scriptsize}, label style={font=\small},
  legend style={at={(0.03,0.62)}, anchor=north west, draw=none, font=\scriptsize},
  xmajorgrids, ymajorgrids, grid style={black!12, line width=0.3pt},
  axis lines*=left, title style={font=\small, align=left},
  title={(b) The threshold is not a lever},
]
\addplot[mark=square*, mark size=1.5pt, """ + ACCENT + r""", line width=0.9pt]
  coordinates {""" + atyp + r"""};
\addlegendentry{under-protection, verbally advanced}
\addplot[mark=*, mark size=1.5pt, black, dashed, line width=0.9pt]
  coordinates {""" + typ_u + r"""};
\addlegendentry{over-restriction, typical}
\draw[black!45, dotted, line width=0.7pt]
  (axis cs:0.55,0) -- (axis cs:0.55,75);
\node[font=\scriptsize, anchor=west] at (axis cs:0.57,68) {adopted};
\end{axis}
\end{tikzpicture}
"""
    write("fig9_calibration", body)


# ----------------------------------------------------------------- Figure 10
def fig_confusion() -> None:
    d = load("exp01_convergence")
    # n = 1 is where the estimator errs; by n = 10 the matrix is the identity
    # and carries no information about the direction of error.
    conf = d["confusion_at_n1"]
    cells = []
    for i in range(5):
        for j in range(5):
            v = conf[i][j]
            shade = int(round(88 * v))
            tc = "white" if shade > 45 else "black"
            label = f"{v:.3f}" if v >= 0.0005 else "--"
            if i != j and v >= 0.0005:
                # above the diagonal = assigned a less restrictive tier
                tc = ACCENT if j > i else "black"
            cells.append(
                f"\\node[cc, fill=black!{shade}, text={tc}] "
                f"at ({0.95 * j:.2f},{-0.62 * i:.2f}) {{{label}}};")
    heads = "\n".join(
        f"\\node[font=\\scriptsize] at ({0.95 * j:.2f},0.55) {{$t_{j+1}$}};"
        for j in range(5))
    ylab = "\n".join(
        f"\\node[font=\\scriptsize, anchor=east] at (-0.58,{-0.62 * i:.2f}) "
        f"{{$t_{i+1}$}};" for i in range(5))
    body = r"""
\begin{tikzpicture}[
  cc/.style={draw, thin, rectangle, minimum width=8.5mm, minimum height=5.4mm,
             font=\scriptsize},
]
""" + heads + "\n" + ylab + "\n" + "\n".join(cells) + r"""
\node[font=\scriptsize] at (1.9,1.05) {assigned tier};
\node[font=\scriptsize, rotate=90] at (-1.15,-1.24) {true tier};
\end{tikzpicture}
"""
    write("fig10_confusion", body)


def main() -> None:
    fig_pipeline(); fig_deployment(); fig_layers(); fig_gating()
    fig_dsr(); fig_ceiling(); fig_privacy(); fig_bypass()
    fig_calibration(); fig_confusion()


if __name__ == "__main__":
    main()
