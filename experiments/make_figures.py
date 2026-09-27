"""Build every manuscript figure as a standalone vector PDF and a 600-dpi PNG.

Each figure is typeset once, at the exact width it occupies in the MDPI layout
(the 13.86 cm text column or the 18.47 cm full width), in the journal's font
(Palatino via `mathpazo`), so the manuscript includes it at 100% and every label
prints at its design size: 8 pt for ticks, legends and diagram text, 9 pt for
axis labels. No figure is rescaled after typesetting. Data figures read
`results/*.json`; diagrams read the specification modules.

    python3 experiments/make_figures.py        # writes figures/*.pdf and *.png

Colours follow the Okabe-Ito palette, which stays distinguishable under the
common colour-vision deficiencies and in greyscale print.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS, ROOT  # noqa: E402
from safenest.lattice import FEATURE_GATING, Access, Capability  # noqa: E402
from safenest.tiers import ALL_TIERS, TIER_SPECS  # noqa: E402
from safenest.verification import abstraction_size  # noqa: E402

FIG = ROOT / "figures"
TEXT_CM = 13.86   # \textwidth of mdpi.cls (394.36 pt)
FULL_CM = 18.47   # \textwidth + \extralength, for full-width figures
DPI = 600

PREAMBLE = r"""\documentclass[10pt,border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathpazo}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes.geometric,positioning,arrows.meta,calc}
\definecolor{okorange}{HTML}{E69F00}
\definecolor{oksky}{HTML}{56B4E9}
\definecolor{okgreen}{HTML}{009E73}
\definecolor{okblue}{HTML}{0072B2}
\definecolor{okverm}{HTML}{D55E00}
\definecolor{okpurple}{HTML}{CC79A7}
\colorlet{accent}{okverm}
\pgfplotsset{
  safenest/.style={
    tick label style={font=\footnotesize},
    label style={font=\small},
    title style={font=\small, align=left},
    legend style={font=\footnotesize, draw=none, fill=none, cells={anchor=west}},
    grid style={black!12, line width=0.3pt},
    axis line style={black!70, line width=0.4pt},
    tick style={black!70, line width=0.4pt},
    axis lines*=left,
  },
}
\newlength{\figwidth}
\tikzset{every node/.append style={execute at begin node={\hyphenpenalty=10000\exhyphenpenalty=10000\relax}}}
"""


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def available(name: str) -> bool:
    p = RESULTS / f"{name}.json"
    return p.exists() and "skipped" not in json.loads(p.read_text())


BUILT: list[tuple[str, float, float]] = []


def build(name: str, body: str, width_cm: float) -> None:
    """Typeset `body` standalone and check it fits its layout width."""
    FIG.mkdir(exist_ok=True)
    tex = FIG / f"{name}.tex"
    tex.write_text(PREAMBLE + f"\\setlength{{\\figwidth}}{{{width_cm}cm}}\n"
                   "\\begin{document}\n" + body.strip() + "\n\\end{document}\n")
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                          tex.name], cwd=FIG, capture_output=True, text=True)
    if run.returncode != 0:
        log = (FIG / f"{name}.log").read_text(errors="replace")
        err = [ln for ln in log.splitlines() if ln.startswith("!")][:5]
        raise RuntimeError(f"{name}: LaTeX failed: {err}")
    info = subprocess.run(["pdfinfo", f"{name}.pdf"], cwd=FIG, capture_output=True,
                          text=True).stdout
    w_pt, h_pt = [float(x) for x in
                  next(ln for ln in info.splitlines() if ln.startswith("Page size"))
                  .split(":")[1].split("pts")[0].split(" x ")]
    width = w_pt / 72 * 2.54
    if width > width_cm + 0.05:
        raise RuntimeError(f"{name}: {width:.2f} cm exceeds its {width_cm} cm layout width")
    subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-singlefile", f"{name}.pdf", name],
                   cwd=FIG, check=True)
    for ext in (".aux", ".log"):
        (FIG / f"{name}{ext}").unlink(missing_ok=True)
    BUILT.append((name, width, h_pt / 72 * 2.54))
    print(f"  {name:22s} {width:5.2f} x {h_pt / 72 * 2.54:5.2f} cm (layout {width_cm} cm)")


# ================================================================ diagrams
def fig_deployment() -> None:
    body = r"""
\begin{tikzpicture}[
  box/.style={draw, thin, rectangle, align=center, inner sep=4pt,
              minimum height=9mm, font=\footnotesize},
  ar/.style={-{Latex[length=2mm]}, thin},
  lab/.style={font=\footnotesize, inner sep=2pt, align=left},
]
\node[box, text width=19mm] (c)  at (0.7,1.55) {child};
\node[box, text width=19mm] (s)  at (0.7,0.35) {signals $x$};
\node[box, text width=19mm] (dv) at (0.7,-0.95) {account and\\attestation flags};
\node[box, text width=26mm] (est) at (4.6,0.35) {tier estimator\\(learned or specified)};
\node[box, text width=33mm] (pol) at (9.1,0.35) {Nested Policy Engine\\$L_0,\ldots,L_4$};
\node[box, text width=33mm] (llm) at (9.1,2.15) {foundation model $\theta$\\(unmodified)};
\node[box, text width=30mm] (soc) at (10.0,-1.45) {Socratic engine $\pi^{*}$\\or referral protocol};
\node[box, text width=31mm] (rej) at (4.9,-1.45) {no model content;\\crisis: hand-off to a\\guardian or helpline};
\node[box, text width=20mm] (out) at (13.6,0.35) {response $y$};
\draw[ar] (c) -- (s);
\draw[ar] (s) -- node[lab, above] {LLRs} (est);
\draw[ar] (dv.east) -- ++(0.75,0) |- (est.west);
\draw[ar] (est) -- node[lab, above] {tier $\hat{\tau}$} (pol);
\draw[ar] (llm) -- node[lab, right] {candidate $y$} (pol);
\draw[ar] (pol.south -| soc.north) -- node[lab, right] {\textsc{modify}} (soc.north);
\draw[ar] ($(pol.south)+(-1.2,0)$) |- node[lab, above, pos=0.78] {\textsc{reject}} (rej.east);
\draw[ar] (pol) -- node[lab, above] {\textsc{accept}} (out);
\draw[ar] (soc.east) -| node[lab, right, pos=0.72] {scaffold\\or refer} (out.south);
\draw[dashed, thin, black!60] (-0.55,0.97) rectangle (1.95,-1.78);
\node[lab, anchor=north west, text=black!70] at (-0.55,-1.8) {signal enclave};
\end{tikzpicture}"""
    build("fig_deployment", body, FULL_CM)


#: Printed names of the nine capabilities, as the text uses them.
CAPABILITY_LABEL = {"homework_answer": "homework answer", "open_ended_chat": "open-ended chat",
                    "code_generation": "code generation", "essay_writing": "essay writing",
                    "freeform_text_output": "free-form text output",
                    "crisis_content": "crisis content",
                    "substance_bodyimage": "substance and body image",
                    "age_inappropriate": "age-inappropriate content",
                    "legal_risk_content": "legal-risk content"}


def fig_gating() -> None:
    marks = {Access.BLOCKED: ("black!72", "white", "B"),
             Access.SOCRATIC: ("black!42", "white", "S"),
             Access.LIMITED: ("black!16", "black", "L"),
             Access.AVAILABLE: ("white", "black", "A")}
    cells = []
    for j, cap in enumerate(Capability):
        y = -0.62 * j
        cells.append(f"\\node[font=\\footnotesize, anchor=east] at (-0.62,{y:.2f}) "
                     f"{{{CAPABILITY_LABEL[cap.value]}}};")
        for i, tier in enumerate(ALL_TIERS):
            fill, tc, ch = marks[FEATURE_GATING[cap][tier]]
            cells.append(f"\\node[cell, fill={fill}, text={tc}] at ({1.12 * i:.2f},{y:.2f}) {{{ch}}};")
    heads = "\n".join(
        f"\\node[font=\\footnotesize] at ({1.12 * i:.2f},0.98) {{$t_{i + 1}$}};\n"
        f"\\node[font=\\footnotesize] at ({1.12 * i:.2f},0.56) "
        f"{{{TIER_SPECS[t].age_low}--{TIER_SPECS[t].age_high}}};"
        for i, t in enumerate(ALL_TIERS))
    body = (r"""
\begin{tikzpicture}[cell/.style={draw, thin, rectangle, minimum width=9mm,
                    minimum height=5.4mm, font=\footnotesize}]
""" + heads + "\n" + "\n".join(cells) + r"""
\node[font=\footnotesize, anchor=north] at (2.24,-5.55)
  {\textbf{B} blocked\quad \textbf{S} Socratic (scaffolded)\quad
   \textbf{L} limited (monitored)\quad \textbf{A} available};
\end{tikzpicture}""")
    build("fig_gating", body, TEXT_CM)


def fig_layers() -> None:
    chain = []
    for i, tier in enumerate(reversed(ALL_TIERS)):
        idx = int(tier) - 1
        spec = TIER_SPECS[tier]
        n_open = sum(1 for c in Capability if FEATURE_GATING[c][tier] is not Access.BLOCKED)
        shade = [70, 55, 40, 25, 12][idx]
        txt = "white" if shade >= 45 else "black"
        chain.append(f"\\node[tierbox, fill=black!{shade}, text={txt}] at (12.75,{-0.85 * i - 0.9:.2f}) "
                     f"{{$t_{idx + 1}$\\quad ages {spec.age_low}--{spec.age_high}\\quad {n_open}/9 unblocked}};")
    layers = [("L_0", "Token filter", "per token", "$10^{3}$\\,Hz", "synchronous"),
              ("L_1", "Socratic guard", "per response", "$1$\\,Hz", "synchronous"),
              ("L_2", "Tier refiner", "per session", "$1.7{\\times}10^{-3}$\\,Hz", "asynchronous"),
              ("L_3", "Memory layer", "cross-session", "$1.2{\\times}10^{-5}$\\,Hz", "asynchronous"),
              ("L_4", "Policy store", "per revision", "$3.2{\\times}10^{-8}$\\,Hz", "synchronous")]
    rows = []
    for i, (tag, name, gran, freq, sync) in enumerate(layers):
        y = -0.85 * i - 0.9
        st = "" if sync == "synchronous" else ", text=black!55"
        rows.append(f"\\node[lbl] at (0,{y:.2f}) {{${tag}$}};\n"
                    f"\\node[lname{st}] at (1.85,{y:.2f}) {{{name}}};\n"
                    f"\\node[lcell{st}] at (4.25,{y:.2f}) {{{gran}}};\n"
                    f"\\node[lcell{st}] at (6.4,{y:.2f}) {{{freq}}};\n"
                    f"\\node[font=\\footnotesize{st}, anchor=west] at (7.55,{y:.2f}) {{{sync}}};")
    body = (r"""
\begin{tikzpicture}[
  lbl/.style={draw, thin, rectangle, minimum width=7mm, minimum height=6mm,
              fill=black!85, text=white, font=\footnotesize},
  lname/.style={draw, thin, rectangle, minimum width=24mm, minimum height=6mm, font=\footnotesize},
  lcell/.style={draw, thin, rectangle, minimum width=20.5mm, minimum height=6mm, font=\footnotesize},
  tierbox/.style={draw, thin, rectangle, minimum width=55mm, minimum height=6mm, font=\footnotesize},
]
\node[font=\small\bfseries, anchor=west] at (-0.45,0) {(a) Nested Policy Engine};
\node[font=\small\bfseries, anchor=west] at (10.0,0) {(b) Monotone policy};
""" + "\n".join(rows) + "\n" + "\n".join(chain) + r"""
\node[font=\footnotesize, anchor=north west, text width=88mm] at (-0.45,-5.2)
  {Decisions compose by conjunction, so no layer can relax another. Only the
   three synchronous layers gate a response.};
\node[font=\footnotesize, anchor=north west, text width=56mm] at (10.0,-5.2)
  {Unblocked sets are nested; $L_4$ rejects any policy edit that breaks
   monotonicity of $a(c,t)$ in $t$.};
\end{tikzpicture}""")
    build("fig_layers", body, FULL_CM)


def fig_protocol() -> None:
    """The measurement protocol: four columns, each diamond a check that can fail."""
    n_abs = f"{abstraction_size():,}".replace(",", "{,}")
    corpora = "three essay corpora"
    if available("exp13_real_estimation"):
        docs = load("exp13_real_estimation")["documents"]
        n_docs = sum(sum(v.values()) for v in docs.values())
        corpora += ", " + f"{n_docs:,}".replace(",", "{,}") + " essays"
    if available("exp16_real_spoken"):
        spoken = load("exp16_real_spoken")["corpora"]
        n_kids = sum(c["children"] for c in spoken.values())
        corpora += "; " + f"{n_kids:,}".replace(",", "{,}") + " children's narratives"
    body = r"""
\begin{tikzpicture}[
  box/.style={draw, thin, rectangle, text width=31mm, align=center, inner sep=3pt,
              minimum height=8.5mm, font=\footnotesize},
  dec/.style={draw, thin, diamond, aspect=2.2, align=center, inner sep=1pt,
              font=\footnotesize},
  term/.style={draw, thin, rectangle, rounded corners=2pt, fill=black!8, align=center,
               inner sep=3pt, font=\footnotesize, text width=27mm},
  hd/.style={font=\footnotesize\bfseries, align=center, text width=36mm},
  ar/.style={-{Latex[length=1.8mm]}, thin},
  lb/.style={font=\footnotesize, inner sep=1.5pt},
]
\node[hd] at (0,0.1) {1.\ Policy corpus and\\ground truth};
\node[box] (a1) at (0,-1.0) {seeded generator: 7 risk categories $\times$ 5 tiers};
\node[box] (a2) at (0,-2.45) {7{,}000 records, labelled by gating matrix and by rubric};
\node[dec] (a3) at (0,-4.05) {rubric reads\\the policy?};
\node[term] (a4) at (0,-5.6) {not independent};
\draw[ar] (a1) -- (a2); \draw[ar] (a2) -- (a3);
\draw[ar] (a3) -- node[lb,right] {yes} (a4);

\node[hd] at (4.55,0.1) {2.\ Specification\\under test};
\node[box] (b1) at (4.55,-1.0) {monotone tier policy $a(c,t)$, nine capabilities};
\node[dec] (b2) at (4.55,-2.6) {monotone?};
\node[term] (b3) at (4.55,-4.05) {reject policy edit};
\node[box] (b4) at (4.55,-5.6) {enumerate """ + n_abs + r""" region-complete responses $\times$ 5 tiers};
\node[dec] (b5) at (4.55,-7.2) {invariants\\hold?};
\node[term] (b6) at (4.55,-8.75) {specification defect};
\draw[ar] (b1) -- (b2); \draw[ar] (b2) -- node[lb,right] {no} (b3);
\draw[ar] (b4) -- (b5); \draw[ar] (b5) -- node[lb,right] {no} (b6);
\draw[ar] (b2.west) -- ++(-0.55,0) node[lb,above] {yes} |- (b4.west);
\draw[ar] (a3.east) -- node[lb,above] {no} ++(0.62,0) |- (b1.west);

\node[hd] at (9.1,0.1) {3.\ Real children's\\language};
\node[box] (c1) at (9.1,-1.0) {""" + corpora + r"""};
\node[box] (c2) at (9.1,-2.45) {five features per 50-word window; duplicates removed};
\node[box] (c3) at (9.1,-3.9) {learned estimators, tested on unseen corpora or children};
\node[dec] (c4) at (9.1,-5.5) {prompt\\held out?};
\node[term] (c5) at (9.1,-7.05) {topic confound: report separately};
\draw[ar] (c1) -- (c2); \draw[ar] (c2) -- (c3); \draw[ar] (c3) -- (c4);
\draw[ar] (c4) -- node[lb,right] {no} (c5);

\node[hd] at (13.65,0.1) {4.\ Measurement};
\node[box] (d1) at (13.65,-1.0) {frameworks, ablations, tuned baselines};
\node[box] (d2) at (13.65,-2.45) {Wilson and bootstrap intervals, 20 seeds};
\node[dec] (d3) at (13.65,-4.05) {at the\\ceiling?};
\node[term] (d4) at (13.65,-5.6) {report recoverable headroom};
\node[term, fill=black!78, text=white] (d5) at (13.65,-7.2) {verdict, with error direction};
\draw[ar] (d1) -- (d2); \draw[ar] (d2) -- (d3);
\draw[ar] (d3) -- node[lb,right] {no} (d4); \draw[ar] (d4) -- (d5);
\draw[ar] (d3.east) -- node[lb,above] {yes} ++(0.35,0) |- (d5.east);
\draw[ar] (b5.east) -- node[lb,above] {yes} ++(0.62,0) |- (c1.west);
\draw[ar] (c4.east) -- node[lb,above] {yes} ++(0.55,0) |- (d1.west);
\end{tikzpicture}"""
    build("fig_protocol", body, FULL_CM)


# ============================================================ data figures
def fig_confusion() -> None:
    conf = load("exp01_convergence")["confusion_at_n1"]
    cells = []
    for i in range(5):
        for j in range(5):
            v = conf[i][j]
            shade = int(round(88 * v))
            tc = "white" if shade > 45 else "black"
            label = f"{v:.3f}" if v >= 0.0005 else "--"
            if i != j and v >= 0.0005:
                tc = "accent" if j > i else "black"
            cells.append(f"\\node[cc, fill=black!{shade}, text={tc}] at ({1.1 * j:.2f},{-0.66 * i:.2f}) {{{label}}};")
    heads = "\n".join(f"\\node[font=\\footnotesize] at ({1.1 * j:.2f},0.6) {{$t_{j + 1}$}};" for j in range(5))
    ylab = "\n".join(f"\\node[font=\\footnotesize, anchor=east] at (-0.62,{-0.66 * i:.2f}) {{$t_{i + 1}$}};"
                     for i in range(5))
    body = (r"""
\begin{tikzpicture}[cc/.style={draw, thin, rectangle, minimum width=10mm,
                    minimum height=5.8mm, font=\footnotesize}]
""" + heads + "\n" + ylab + "\n" + "\n".join(cells) + r"""
\node[font=\small] at (2.2,1.1) {assigned tier};
\node[font=\small, rotate=90] at (-1.25,-1.32) {true tier};
\end{tikzpicture}""")
    build("fig_confusion", body, TEXT_CM)


def _stacked_panel(res: dict, title: str, name: str, first: bool, at: str) -> str:
    order = ["NPL (specified)", "NPL (learned)", "Logistic (features)",
             "Gradient boosting (features)", "Logistic (TF-IDF)", "Logistic (MiniLM)"]
    short = {"NPL (specified)": "NPL, specified", "NPL (learned)": "NPL, learned",
             "Logistic (features)": "Logistic, features",
             "Gradient boosting (features)": "Boosting, features",
             "Logistic (TF-IDF)": "Logistic, TF-IDF", "Logistic (MiniLM)": "Logistic, MiniLM"}
    parts = {"under": [], "correct": [], "over": []}
    for k, m in enumerate(order):
        r = res[m]["n10"]
        parts["under"].append(f"({100 * r['under']:.1f},{k})")
        parts["correct"].append(f"({100 * r['accuracy']:.1f},{k})")
        parts["over"].append(f"({100 * r['over']:.1f},{k})")
    yl = ",".join(f"{{{short[m]}}}" for m in order) if first else ""
    return (r"""
\begin{axis}[safenest, name=""" + name + ", " + at + r"""
  width=0.305\figwidth, height=52mm, scale only axis=false,
  xbar stacked, bar width=7pt, xmin=0, xmax=100, y dir=reverse,
  ytick={0,1,2,3,4,5}, yticklabels={""" + yl + r"""},
  enlarge y limits=0.12, xmajorgrids, xtick={0,25,50,75,100},
  xlabel={Share of children (\%)}, title={""" + title + r"""},
]
\addplot[fill=accent, draw=black!60, line width=0.3pt] coordinates {""" + " ".join(parts["under"]) + r"""};
\addplot[fill=black!18, draw=black!60, line width=0.3pt] coordinates {""" + " ".join(parts["correct"]) + r"""};
\addplot[fill=oksky, draw=black!60, line width=0.3pt] coordinates {""" + " ".join(parts["over"]) + r"""};
\end{axis}""")


def fig_real_loco() -> None:
    d = load("exp13_real_estimation")["leave_one_corpus_out"]
    body = (r"\begin{tikzpicture}"
            + _stacked_panel(d["persuade"], "(a) PERSUADE held out", "p1", True, "")
            + _stacked_panel(d["asap"], "(b) ASAP held out", "p2", False,
                             "at={($(p1.east)+(9mm,0)$)}, anchor=west,")
            + _stacked_panel(d["ellipse"], "(c) ELLIPSE held out", "p3", False,
                             "at={($(p2.east)+(9mm,0)$)}, anchor=west,")
            + r"""
\node[anchor=north, font=\footnotesize] at ($(p2.south)+(0,-9mm)$) {%
  \tikz\fill[accent, draw=black!60] (0,0) rectangle (3mm,2mm);~under-protected\quad
  \tikz\fill[black!18, draw=black!60] (0,0) rectangle (3mm,2mm);~correct tier\quad
  \tikz\fill[oksky, draw=black!60] (0,0) rectangle (3mm,2mm);~over-protected};
\end{tikzpicture}""")
    build("fig_real_loco", body, FULL_CM)


def fig_real_subgroups() -> None:
    d = load("exp15_real_subgroups")["training"]
    groups = [("ell", "English learner"), ("disability", "Identified disability"),
              ("econ", "Econ.\\ disadvantaged"), ("gender", "Female")]
    models = [("NPL (learned)", "NPL"), ("Gradient boosting (features)", "Boosting")]
    labels, rows = [], []
    for g, glab in groups:
        for m, mlab in models:
            labels.append(f"{{{glab}, {mlab}}}")
            rows.append((g, m))

    def series(what: str, train: str, offset: float) -> str:
        pts = []
        for k, (g, m) in enumerate(rows):
            diff = d[train]["subgroups"][g][m]["difference"][what]
            est, lo, hi = (100 * diff["estimate"], 100 * diff["ci"][0], 100 * diff["ci"][1])
            pts.append(f"({est:.2f},{k + offset:.2f}) -= ({est - lo:.2f},0) += ({hi - est:.2f},0)")
        return " ".join(pts)

    lo = min(100 * d[t]["subgroups"][g][m]["difference"][w]["ci"][0]
             for t in d for g, _ in groups for m, _ in models for w in ("under", "over"))
    hi = max(100 * d[t]["subgroups"][g][m]["difference"][w]["ci"][1]
             for t in d for g, _ in groups for m, _ in models for w in ("under", "over"))
    xmin, xmax = 5 * (int(lo // 5) - 1), 5 * (int(hi // 5) + 1)

    def panel(what: str, title: str, name: str, first: bool, at: str) -> str:
        yl = ",".join(labels) if first else ""
        return (r"""
\begin{axis}[safenest, name=""" + name + ", " + at + r"""
  width=0.40\figwidth, height=62mm, y dir=reverse,
  xmin=""" + str(xmin) + ", xmax=" + str(xmax) + r""",
  ytick={0,...,7}, yticklabels={""" + yl + r"""}, enlarge y limits=0.07,
  ytick style={draw=none},
  xmajorgrids, xlabel={Difference (percentage points)}, title={""" + title + r"""},
  legend style={at={(0.5,-0.30)}, anchor=north, legend columns=2,
                /tikz/every even column/.append style={column sep=2ex}},
]
\draw[black!55, line width=0.5pt] (axis cs:0,-0.6) -- (axis cs:0,7.6);
\addplot[only marks, mark=*, mark size=1.6pt, black,
  error bars/.cd, x dir=both, x explicit, error bar style={line width=0.5pt}]
  coordinates {""" + series(what, "with ELLIPSE", -0.17) + r"""};
\addplot[only marks, mark=square*, mark size=1.5pt, accent,
  error bars/.cd, x dir=both, x explicit, error bar style={line width=0.5pt}]
  coordinates {""" + series(what, "without ELLIPSE", 0.17) + r"""};
""" + (r"\legend{trained with ELLIPSE, trained without ELLIPSE}" if not first else "") + r"""
\end{axis}""")

    body = (r"\begin{tikzpicture}"
            + panel("under", "(a) Under-protection", "u", True, "")
            + panel("over", "(b) Over-protection", "o", False,
                    "at={($(u.east)+(6mm,0)$)}, anchor=west,")
            + r"\end{tikzpicture}")
    build("fig_real_subgroups", body, FULL_CM)


def fig_real_spoken() -> None:
    """Children's speech: pooled outcomes, and the language-impairment gap."""
    d = load("exp16_real_spoken")
    order = ["NPL (specified)", "NPL (learned)", "Logistic (features)",
             "Gradient boosting (features)", "Logistic (TF-IDF)", "Logistic (MiniLM)"]
    imp = d["impairment"]

    def series(what: str, offset: float) -> str:
        pts = []
        for k, m in enumerate(order):
            x = imp[m]["difference"][what]
            est, lo, hi = 100 * x["estimate"], 100 * x["ci"][0], 100 * x["ci"][1]
            pts.append(f"({est:.2f},{k + offset:.2f}) -= ({est - lo:.2f},0) += ({hi - est:.2f},0)")
        return " ".join(pts)

    lo = min(100 * imp[m]["difference"][w]["ci"][0] for m in order for w in ("under", "over"))
    hi = max(100 * imp[m]["difference"][w]["ci"][1] for m in order for w in ("under", "over"))
    xmin, xmax = 10 * (int(lo // 10)), 10 * (int(hi // 10) + 1)
    body = (r"\begin{tikzpicture}"
            + _stacked_panel(d["within_corpus"]["pooled"],
                             "(a) Gillam and ENNI pooled, five folds", "s1", True, "")
            + r"""
\begin{axis}[safenest, name=s2, at={($(s1.east)+(9mm,0)$)}, anchor=west,
  width=0.40\figwidth, height=52mm, y dir=reverse,
  xmin=""" + str(xmin) + ", xmax=" + str(xmax) + r""",
  ytick={0,1,2,3,4,5}, yticklabels={}, enlarge y limits=0.12, ytick style={draw=none},
  xmajorgrids, xlabel={Impaired minus typical (percentage points)},
  title={(b) Language impairment, same tiers},
]
\draw[black!55, line width=0.5pt] (axis cs:0,-0.6) -- (axis cs:0,5.6);
\addplot[only marks, mark=*, mark size=1.6pt, accent,
  error bars/.cd, x dir=both, x explicit, error bar style={line width=0.5pt}]
  coordinates {""" + series("under", -0.15) + r"""};
\addplot[only marks, mark=square*, mark size=1.5pt, oksky,
  error bars/.cd, x dir=both, x explicit, error bar style={line width=0.5pt}]
  coordinates {""" + series("over", 0.15) + r"""};
\end{axis}
\node[anchor=north, font=\footnotesize] at ($(s1.south)!0.5!(s2.south)+(0,-9mm)$) {%
  \tikz\fill[accent, draw=black!60] (0,0) rectangle (3mm,2mm);~under-protected\quad
  \tikz\fill[black!18, draw=black!60] (0,0) rectangle (3mm,2mm);~correct tier\quad
  \tikz\fill[oksky, draw=black!60] (0,0) rectangle (3mm,2mm);~over-protected\qquad
  \tikz\fill[accent] (0,0) circle (0.9mm);~under-protection gap\quad
  \tikz\fill[oksky] (0,0) rectangle (1.8mm,1.8mm);~over-protection gap};
\end{tikzpicture}""")
    build("fig_real_spoken", body, FULL_CM)


def fig_privacy() -> None:
    d = load("exp05_privacy")
    eps = d["epsilons"]
    corpus = " ".join(f"({e},{100 * d['accuracy_corpus_dp'][str(e)]:.1f}) +- "
                      f"(0,{100 * d['accuracy_corpus_dp_sd_over_releases'][str(e)]:.1f})" for e in eps)
    local = " ".join(f"({e},{100 * d['accuracy_local_dp'][str(e)]:.1f})" for e in eps)
    ref = 100 * d["accuracy_nonprivate_release"]
    left = r"""
\begin{axis}[safenest, name=l, width=0.40\figwidth, height=52mm,
  xmode=log, log basis x=10, xmin=0.08, xmax=130, ymin=0, ymax=105,
  xlabel={Privacy budget $\varepsilon$}, ylabel={Tier accuracy at $n=10$ (\%)},
  xmajorgrids, ymajorgrids, title={(a) Simulation},
  legend style={at={(0.5,-0.32)}, anchor=north, legend columns=1},
]
\addplot[black!45, dashed, line width=0.6pt] coordinates {(0.08,""" + f"{ref:.1f}" + r""") (130,""" + f"{ref:.1f}" + r""")};
\addlegendentry{non-private release}
\addplot[mark=*, mark size=1.5pt, okblue, line width=0.8pt,
  error bars/.cd, y dir=both, y explicit, error bar style={line width=0.4pt}]
  coordinates {""" + corpus + r"""};
\addlegendentry{corpus-level DP (mean $\pm$ SD, 20 releases)}
\addplot[mark=square*, mark size=1.5pt, accent, dashed, line width=0.8pt]
  coordinates {""" + local + r"""};
\addlegendentry{local DP on the live child}
\addplot[black!45, dotted, line width=0.7pt, forget plot] coordinates {(0.08,20) (130,20)};
\node[font=\footnotesize, anchor=south east, text=black!60] at (axis cs:120,21) {chance};
\end{axis}"""
    right = ""
    if available("exp14_real_privacy"):
        w = load("exp14_real_privacy")["within_persuade"]
        es = [0.1, 0.3, 1.0, 3.0, 10.0]
        agree = " ".join(f"({e},{100 * w[str(e)]['agreement_with_non_private']['mean']:.1f})" for e in es)
        bal = " ".join(f"({e},{100 * w[str(e)]['balanced_accuracy']['mean']:.1f}) +- "
                       f"(0,{100 * w[str(e)]['balanced_accuracy']['sd']:.1f})" for e in es)
        bal_ref = 100 * w["clipped_no_noise"]["balanced_accuracy"]
        right = r"""
\begin{axis}[safenest, name=r, at={($(l.east)+(18mm,0)$)}, anchor=west,
  width=0.40\figwidth, height=52mm,
  xmode=log, log basis x=10, xmin=0.08, xmax=13, ymin=0, ymax=105,
  xlabel={Privacy budget $\varepsilon$}, ylabel={Percentage of sessions},
  xmajorgrids, ymajorgrids, title={(b) Parameters learned from PERSUADE},
  legend style={at={(0.5,-0.32)}, anchor=north, legend columns=1},
]
\addplot[mark=*, mark size=1.5pt, okblue, line width=0.8pt] coordinates {""" + agree + r"""};
\addlegendentry{same tier as the non-private release}
\addplot[mark=triangle*, mark size=1.8pt, okgreen, line width=0.8pt,
  error bars/.cd, y dir=both, y explicit, error bar style={line width=0.4pt}]
  coordinates {""" + bal + r"""};
\addlegendentry{balanced accuracy (mean $\pm$ SD)}
\addplot[okgreen, dashed, line width=0.6pt] coordinates {(0.08,""" + f"{bal_ref:.1f}" + r""") (13,""" + f"{bal_ref:.1f}" + r""")};
\addlegendentry{balanced accuracy, non-private}
\end{axis}"""
    build("fig_privacy", r"\begin{tikzpicture}" + left + right + r"\end{tikzpicture}", FULL_CM)


def fig_dsr() -> None:
    d = load("exp04_comparative")["results"]["rubric"]
    order = ["Topical rail", "Harm classifier", "Agent firewall", "Constitutional rule",
             "COPPA binary rule", "Child-safety classifier", "Age-band oracle", "NPL (full)"]
    fills = ["black!6", "black!18", "black!30", "black!42", "black!54", "black!66",
             "black!82", "accent"]
    plots = []
    for name, fill in zip(order, fills):
        coords = []
        for i in range(1, 6):
            c = d[name]["by_tier"][f"t{i}"]
            v = 100 * c["dsr"]
            coords.append(f"({i},{v:.1f}) -= (0,{v - 100 * c['dsr_ci_low']:.2f}) "
                          f"+= (0,{100 * c['dsr_ci_high'] - v:.2f})")
        plots.append("\\addplot[ybar, fill=" + fill + ", draw=black!70, line width=0.3pt,\n"
                     "  error bars/.cd, y dir=both, y explicit,\n"
                     "  error bar style={line width=0.3pt, black!60}]\n"
                     "  coordinates {" + " ".join(coords) + "};\n"
                     "\\addlegendentry{" + name + "}")
    body = r"""
\begin{tikzpicture}
\begin{axis}[safenest, width=\figwidth, height=62mm,
  ybar=0.4pt, bar width=4.2pt, enlarge x limits=0.10, ymin=0, ymax=106,
  ytick={0,20,40,60,80,100},
  xtick={1,2,3,4,5}, xticklabels={$t_1$ (3--6),$t_2$ (7--9),$t_3$ (10--12),$t_4$ (13--15),$t_5$ (16--17)},
  xlabel={Developmental tier}, ylabel={Developmental safety rate (\%)},
  ymajorgrids,
  legend style={at={(0.5,1.03)}, anchor=south, legend columns=4,
                /tikz/every even column/.append style={column sep=1.2ex}},
  legend image code/.code={\draw[#1] (0cm,-0.08cm) rectangle (0.26cm,0.14cm);},
]
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}"""
    build("fig_dsr", body, FULL_CM)


def fig_ceiling() -> None:
    d = load("exp08_ceiling")
    npl, ceil = 100 * d["npl_dsr"], 100 * d["ceiling_cv"]
    # Only cells where some rule on the same features does better than NPL.
    cells = [c for c in d["cell_losses"] if c["prompts_lost"] > 0][:6]
    short = {"code_generation": "code", "essay_creative_writing": "essay",
             "crisis_self_harm": "crisis", "open_ended_chat": "open chat",
             "homework_assignment": "homework", "substance_body_image": "substance",
             "age_inappropriate_content": "age-inappropriate"}
    labels = ",".join(f"{{$t_{c['tier'][1]}$ {short.get(c['category'], c['category'])}}}" for c in cells)
    ceil_c = " ".join(f"({c['ceiling']:.1f},{i})" for i, c in enumerate(cells))
    npl_c = " ".join(f"({c['npl']:.1f},{i})" for i, c in enumerate(cells))
    body = r"""
\begin{tikzpicture}
\begin{axis}[safenest, name=l, width=0.34\figwidth, height=44mm,
  xbar, bar width=7pt, xmin=0, xmax=118, enlarge y limits=0.28,
  ytick={0,1,2}, yticklabels={{recoverable},{NPL},{ceiling}},
  xlabel={Developmental safety rate (\%)}, xmajorgrids,
  nodes near coords, nodes near coords style={font=\footnotesize},
  title={(a) Reachable headroom},
]
\addplot[fill=black!18, draw=black!70, line width=0.3pt] coordinates {""" + (
        f"({ceil - npl:.1f},0) ({npl:.1f},1) ({ceil:.1f},2)") + r"""};
\end{axis}
\begin{axis}[safenest, at={($(l.east)+(30mm,0)$)}, anchor=west,
  width=0.40\figwidth, height=44mm,
  xbar, bar width=4.5pt, xmin=0, xmax=112, enlarge y limits=0.12, y dir=reverse,
  xtick={0,20,40,60,80,100},
  nodes near coords, point meta=x,
  every node near coord/.append style={font=\scriptsize, /pgf/number format/fixed,
                                       /pgf/number format/precision=0},
  ytick={""" + ",".join(str(i) for i in range(len(cells))) + r"""},
  yticklabels={""" + labels + r"""}, ytick style={draw=none},
  xlabel={DSR within cell (\%)}, xmajorgrids,
  legend style={at={(1.02,1.0)}, anchor=north west},
  legend image code/.code={\draw[#1] (0cm,-0.08cm) rectangle (0.3cm,0.14cm);},
  title={(b) Where the recoverable error sits},
]
\addplot[fill=black!12, draw=black!70, line width=0.3pt] coordinates {""" + ceil_c + r"""};
\addlegendentry{ceiling}
\addplot[fill=accent, draw=black!70, line width=0.3pt] coordinates {""" + npl_c + r"""};
\addlegendentry{NPL}
\end{axis}
\end{tikzpicture}"""
    build("fig_ceiling", body, FULL_CM)


def fig_bypass() -> None:
    d = load("exp10_bypass")

    def curve(points: list[dict]) -> str:
        pts = sorted({(round(p["fpr"], 4), round(p["tpr"], 4)) for p in points})
        return " ".join(f"({x:.4f},{y:.4f})" for x, y in [(0.0, 0.0)] + pts + [(1.0, 1.0)])

    op = d["operating_points_alpha1"]
    esc = " ".join(f"({a},{100 * d['by_spoof_strength'][str(a)]['escalation_rate']:.1f})" for a in d["alphas"])
    det = " ".join(f"({a},{100 * d['by_spoof_strength'][str(a)]['detection_rate']:.1f})" for a in d["alphas"])
    body = r"""
\begin{tikzpicture}
\begin{axis}[safenest, name=l, width=0.40\figwidth, height=52mm,
  xmin=0, xmax=1, ymin=0, ymax=1,
  xlabel={False-flag rate, genuine $t_2$ children}, ylabel={Impersonators flagged},
  xmajorgrids, ymajorgrids, title={(a) Detector ROC at $\alpha=1$},
  legend style={at={(0.5,-0.32)}, anchor=north, legend columns=1},
]
\addplot[black, line width=0.9pt] coordinates {""" + curve(d["roc"]["1.0"]) + r"""};
\addlegendentry{vs typical child (AUC """ + f"{d['auc_vs_typical']['1.0']:.2f}" + r""")}
\addplot[accent, dashed, line width=0.9pt] coordinates {""" + curve(d["roc_vs_verbally_advanced"]) + r"""};
\addlegendentry{vs verbally advanced child (AUC """ + f"{d['auc_vs_verbally_advanced']:.2f}" + r""")}
\addplot[black!45, dotted, line width=0.8pt, forget plot] coordinates {(0,0) (1,1)};
\addplot[only marks, mark=*, mark size=2pt, black, forget plot] coordinates {(""" + (
        f"{op['typical']['fpr']:.4f},{op['typical']['tpr']:.4f}") + r""")};
\addplot[only marks, mark=*, mark size=2pt, accent, forget plot] coordinates {(""" + (
        f"{op['verbally_advanced']['fpr']:.4f},{op['verbally_advanced']['tpr']:.4f}") + r""")};
\end{axis}
\begin{axis}[safenest, at={($(l.east)+(18mm,0)$)}, anchor=west,
  width=0.40\figwidth, height=52mm,
  xmin=-0.05, xmax=1.05, ymin=0, ymax=105,
  xlabel={Spoof strength $\alpha$}, ylabel={Sessions (\%)},
  xmajorgrids, ymajorgrids, title={(b) Escalation outruns detection},
  legend style={at={(0.5,-0.32)}, anchor=north, legend columns=1},
]
\addplot[mark=*, mark size=1.5pt, accent, line width=0.9pt] coordinates {""" + esc + r"""};
\addlegendentry{escalated above $t_2$}
\addplot[mark=square*, mark size=1.5pt, black, dashed, line width=0.9pt] coordinates {""" + det + r"""};
\addlegendentry{flagged by the detector}
\end{axis}
\end{tikzpicture}"""
    build("fig_bypass", body, FULL_CM)


def fig_calibration() -> None:
    d = load("exp11_reliability")
    pts = " ".join(f"({100 * b['mean_confidence']:.1f},{100 * b['observed_accuracy']:.1f})"
                   for b in d["calibration_bins"])
    g = load("exp09_operating")
    typ = " ".join(f"({x},{100 * g['gamma_typical'][str(x)]['over_restriction']:.1f})" for x in g["gamma_grid"])
    atyp = " ".join(f"({x},{100 * g['gamma_verbally_advanced'][str(x)]['under_protection']:.1f})"
                    for x in g["gamma_grid"])
    body = r"""
\begin{tikzpicture}
\begin{axis}[safenest, name=l, width=0.40\figwidth, height=48mm,
  xmin=40, xmax=105, ymin=30, ymax=105,
  xlabel={Mean posterior confidence (\%)}, ylabel={Observed accuracy (\%)},
  xmajorgrids, ymajorgrids, title={(a) Posterior calibration},
  legend style={at={(0.97,0.05)}, anchor=south east},
]
\addplot[black!45, dotted, line width=0.8pt] coordinates {(40,40) (105,105)};
\addlegendentry{perfect}
\addplot[mark=*, mark size=1.6pt, okblue, line width=0.9pt] coordinates {""" + pts + r"""};
\addlegendentry{observed}
\node[font=\footnotesize, anchor=west] at (axis cs:44,90) {ECE """ + f"{d['expected_calibration_error']:.4f}" + r"""};
\end{axis}
\begin{axis}[safenest, at={($(l.east)+(18mm,0)$)}, anchor=west,
  width=0.40\figwidth, height=48mm, xmin=0.15, xmax=1.0, ymin=0, ymax=75,
  xlabel={Confidence threshold $\gamma_{\mathrm{conf}}$}, ylabel={Error rate (\%)},
  xmajorgrids, ymajorgrids, title={(b) The threshold is not a lever},
  legend style={at={(0.03,0.62)}, anchor=north west},
]
\addplot[mark=square*, mark size=1.5pt, accent, line width=0.9pt] coordinates {""" + atyp + r"""};
\addlegendentry{under-protection, verbally advanced}
\addplot[mark=*, mark size=1.5pt, black, dashed, line width=0.9pt] coordinates {""" + typ + r"""};
\addlegendentry{over-restriction, typical}
\draw[black!45, dotted, line width=0.7pt] (axis cs:0.55,0) -- (axis cs:0.55,75);
\node[font=\footnotesize, anchor=west] at (axis cs:0.57,68) {adopted};
\end{axis}
\end{tikzpicture}"""
    build("fig_calibration", body, FULL_CM)


def main() -> int:
    if not shutil.which("pdflatex") or not shutil.which("pdftoppm"):
        print("  make_figures needs pdflatex and pdftoppm (poppler)")
        return 1
    fig_deployment(); fig_gating(); fig_layers(); fig_protocol(); fig_confusion()
    if available("exp13_real_estimation"):
        fig_real_loco()
    if available("exp15_real_subgroups"):
        fig_real_subgroups()
    if available("exp16_real_spoken") and "impairment" in load("exp16_real_spoken"):
        fig_real_spoken()
    fig_privacy(); fig_dsr(); fig_ceiling(); fig_bypass(); fig_calibration()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
