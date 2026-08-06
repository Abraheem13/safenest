"""Regenerate the four results figures from results/*.json.

Writes into figures/ using the filenames the manuscript already \\includegraphics:
    fig7_confusion_heatmap.png
    fig8_feature_radar.png
    fig9_privacy_curve.png
    fig10_dsr_grouped.png

The three schematic figures (fig1, fig2, fig4, fig5, fig6) are unchanged and are
not produced here -- they illustrate the architecture, not the results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import FIGURES, RESULTS  # noqa: E402
from safenest.signals import LINGUISTIC_FEATURES, SignalModel  # noqa: E402
from safenest.tiers import ALL_TIERS  # noqa: E402

TIER_LABELS = [f"$t_{i}$" for i in range(1, 6)]
DPI = 300


def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/run_all.py first")
    return json.loads(path.read_text())


def fig7_confusion() -> None:
    d = load("exp01_convergence")
    conf = np.array(d["confusion_at_n10"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), TIER_LABELS)
    ax.set_yticks(range(5), TIER_LABELS)
    ax.set_xlabel("Assigned tier")
    ax.set_ylabel("True tier")
    for i in range(5):
        for j in range(5):
            v = conf[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Proportion of trials")
    ax.set_title("Tier classification at $n=10$ interactions", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_confusion_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig8_radar() -> None:
    model = SignalModel()
    means = np.array([model.linguistic_params(t)[0] for t in ALL_TIERS])
    # Normalise each feature to [0, 1] across tiers so axes are comparable.
    lo, hi = means.min(axis=0), means.max(axis=0)
    norm = (means - lo) / np.where(hi - lo == 0, 1, hi - lo)
    labels = [f.replace("_", " ") for f in LINGUISTIC_FEATURES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw={"polar": True})
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    for i, t in enumerate(ALL_TIERS):
        vals = np.concatenate([norm[i], norm[i][:1]])
        ax.plot(angles, vals, color=colours[i], linewidth=1.6, label=TIER_LABELS[i])
        ax.fill(angles, vals, color=colours[i], alpha=0.08)
    ax.set_xticks(angles[:-1], labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Normalised linguistic feature profiles by tier", fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_feature_radar.png", dpi=DPI)
    plt.close(fig)


def fig9_privacy() -> None:
    d = load("exp05_privacy")
    eps = np.array(d["epsilons"], dtype=float)
    corpus = np.array([100 * d["accuracy_corpus_dp"][str(e)] for e in eps])
    local = np.array([100 * d["accuracy_local_dp"][str(e)] for e in eps])

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.semilogx(eps, corpus, "o-", color="#1f5f8b", label="Corpus-level DP")
    ax.semilogx(eps, local, "s--", color="#b5432f", label="Local DP at inference")
    ax.axhline(20, color="grey", linestyle=":", linewidth=1)
    ax.text(eps[0], 21.5, "chance (20%)", fontsize=7, color="grey")
    i = list(eps).index(1.0)
    ax.plot(1.0, corpus[i], marker="*", markersize=15, color="#1f5f8b", zorder=5)
    ax.annotate(f"adopted\n$\\varepsilon=1.0$, {corpus[i]:.1f}%",
                xy=(1.0, corpus[i]), xytext=(1.6, 78), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (log scale)")
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_privacy_curve.png", dpi=DPI)
    plt.close(fig)


def fig10_dsr() -> None:
    d = load("exp04_comparative")
    res = d["results"]["rubric"]
    order = [
        "NeMo Guardrails", "Llama Guard", "LlamaFirewall", "Constitutional AI",
        "COPPA binary rule", "Child-safety classifier", "Age-conditioned (oracle)",
        "NPL (ours)",
    ]
    greys = plt.cm.Greys(np.linspace(0.30, 0.72, len(order) - 1))
    colours = list(greys) + ["#b5432f"]

    x = np.arange(5)
    width = 0.10
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for k, name in enumerate(order):
        by_tier = res[name]["by_tier"]
        vals = [100 * by_tier[f"t{i}"]["dsr"] for i in range(1, 6)]
        err = [
            [100 * (by_tier[f"t{i}"]["dsr"] - by_tier[f"t{i}"]["dsr_ci_low"]) for i in range(1, 6)],
            [100 * (by_tier[f"t{i}"]["dsr_ci_high"] - by_tier[f"t{i}"]["dsr"]) for i in range(1, 6)],
        ]
        ax.bar(x + (k - len(order) / 2) * width, vals, width, yerr=err,
               capsize=1.5, error_kw={"lw": 0.5}, label=name, color=colours[k],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x, [f"$t_{i}$" for i in range(1, 6)])
    ax.set_xlabel("Developmental tier")
    ax.set_ylabel("Developmental Safety Rate (%)")
    ax.set_ylim(0, 118)
    ax.legend(fontsize=7, ncol=4, loc="upper center", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_dsr_grouped.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    fig7_confusion()
    fig8_radar()
    fig9_privacy()
    fig10_dsr()
    for name in ("fig7_confusion_heatmap", "fig8_feature_radar",
                 "fig9_privacy_curve", "fig10_dsr_grouped"):
        print(f"  wrote figures/{name}.png")


if __name__ == "__main__":
    main()
