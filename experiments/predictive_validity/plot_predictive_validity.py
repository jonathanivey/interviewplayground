"""
Faceted scatter plot of simulated vs. human metric values across all
studies/agents: one small panel per metric, in raw units (no standardization),
with a fixed 4-column grid and a dedicated legend column. This is the figure
used in the paper.

Reads: comparisons/human_vs_simulation_comparison__all_studies.json
Writes: figures/predictive_validity_facets.{png,pdf}

Usage:
    python plot_predictive_validity.py
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "comparisons" / "human_vs_simulation_comparison__all_studies.json"
OUT_DIR = BASE_DIR / "figures"

# Metric -> category, in the order each metric is computed by its evaluator
# (src/interviewplayground/interviewreportcard/{category}.py).
METRIC_CATEGORIES = {
    "relevant_response_volume_avg": "participant_responses",
    "interview_guide_coverage": "participant_responses",
    "novel_responses_avg": "participant_responses",
    "overall_experience": "participant_experience",
    "comfort_level": "participant_experience",
    "avg_turns": "conversation_length",
    "avg_response_length": "conversation_length",
    "coherence": "interviewer_behavior",
    "adaptiveness": "interviewer_behavior",
    "leading_questions_avg": "interviewer_behavior",
    "unclear_questions_avg": "interviewer_behavior",
    "support_rapport_avg": "interviewer_behavior",
}

CATEGORY_ORDER = [
    "participant_responses",
    "interviewer_behavior",
    "participant_experience",
    "conversation_length",
]
CATEGORY_LABELS = {
    "conversation_length": "Conversation Length",
    "interviewer_behavior": "Interviewer Behavior",
    "participant_experience": "Participant Experience",
    "participant_responses": "Participant Responses",
}
# One fixed hue per category (categorical slots 1/2/7/3 of the validated
# default palette).
CATEGORY_BASE_HUE = {
    "conversation_length": "#2a78d6",   # blue
    "interviewer_behavior": "#eb6834",  # orange
    "participant_experience": "#4a3aa7",  # violet
    "participant_responses": "#1baf7a",      # aqua
}

METRIC_LABELS = {
    "avg_turns": "Conversation Turns",
    "avg_response_length": "Average Response\nLength",
    "relevant_response_volume_avg": "Relevant Response\nVolume",
    "interview_guide_coverage": "Interview Guide\nCoverage",
    "novel_responses_avg": "Novel Responses",
    "coherence": "Coherence",
    "adaptiveness": "Adaptiveness",
    "leading_questions_avg": "Leading Questions",
    "support_rapport_avg": "Support & Rapport\nStatements",
    "unclear_questions_avg": "Unclear Questions",
    "comfort_level": "Comfort Level",
    "overall_experience": "Overall Experience",
}

PRESET_LABELS = {
    "obesity_weight_management": "Obesity & Weight\nManagement",
    "asian_american_politics": "Asian American\nPolitics",
    "genai_knowledge_work": "GenAI Knowledge\nWork",
}
PRESET_MARKERS = {
    "obesity_weight_management": "o",
    "asian_american_politics": "s",
    "genai_knowledge_work": "^",
}

# Shared tick logic so every facet panel shows the same number of ticks
# (FACET_N_TICKS), on the same scale (x/y ranges match within a panel), with
# the same label formatting — regardless of that metric's units/scale.
FACET_N_TICKS = 3

# These metrics get fixed whole-number ticks instead of FACET_N_TICKS evenly
# spaced "nice" ticks — their data range is narrow enough that the generic
# algorithm lands on half-point ticks (2.5, 3.5), which reads as more
# precision than a 1-5 rating scale actually has.
FACET_INTEGER_TICKS = {
    "coherence": [1, 2, 3, 4],
    "adaptiveness": [1, 2, 3, 4],
    "comfort_level": [1, 2, 3, 4],
}

# Grid layout: metric panels fill a DATA_COLS-wide block row-major, grouped by
# category in CATEGORY_ORDER, plus one more column to the right holding the
# Study/Category legends (making the grid DATA_COLS + 1 wide).
DATA_COLS = 4


def _nice_num(x: float) -> float:
    """Round x up to the nearest "nice" 1/2/5/10 * 10^k value (classic
    Talbot/Heckbert axis-tick algorithm) — used for the tick step so that
    n ticks are always enough to span the full data range."""
    exp = math.floor(math.log10(x))
    f = x / 10**exp
    nf = 1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10
    return nf * 10**exp


def _nice_ticks_raw(lo: float, hi: float, n: int) -> np.ndarray:
    """n evenly-spaced "nice" tick values (uniform step, round numbers) that
    together span at least [lo, hi]."""
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    step = _nice_num((hi - lo) / (n - 1))
    start = math.floor(lo / step) * step
    ticks = [start + i * step for i in range(n)]
    for _ in range(n):
        if ticks[-1] >= hi - 1e-9:
            break
        start += step
        ticks = [start + i * step for i in range(n)]
    return np.array(ticks)


def _nice_ticks(lo: float, hi: float, n: int) -> np.ndarray:
    """n evenly-spaced "nice" tick values, taken as every other point of a
    finer (2n-1)-point nice grid — building a finer grid first and then
    subsampling keeps the axis tightly fit to [lo, hi]; computing n ticks
    directly would pick a much coarser step for small n (e.g. 3) and could
    overshoot the data range substantially."""
    return _nice_ticks_raw(lo, hi, 2 * n - 1)[::2]


def _format_tick_value(value: float, _pos=None) -> str:
    """Values >= 1000 shown in "K" units, whole numbers unadorned, fractional
    values trimmed to <= 2 decimals."""
    if abs(value) >= 1000:
        thousands = value / 1000
        if abs(thousands - round(thousands)) < 1e-9:
            return f"{int(round(thousands))}K"
        return f"{thousands:.1f}".rstrip("0").rstrip(".") + "K"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _metrics_by_category() -> dict[str, list[str]]:
    return {
        category: [m for m, c in METRIC_CATEGORIES.items() if c == category]
        for category in CATEGORY_ORDER
    }


def plot_facets(data: dict, out_path: pathlib.Path) -> None:
    """One small scatter per metric (raw units, no z-score), filling a
    DATA_COLS-wide grid row-major (grouped by category), plus one more
    column on the right holding the Study/Category legends."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    metrics_by_cat = _metrics_by_category()
    flat_metrics = [
        (m, cat) for cat in CATEGORY_ORDER for m in metrics_by_cat[cat]
    ]
    nrows = math.ceil(len(flat_metrics) / DATA_COLS)
    ncols = DATA_COLS + 1  # + 1 legend column

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.25 * nrows))
    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        top=0.90,
        bottom=0.09,
        hspace=0.45,
        wspace=0.35,
    )

    # Fixed axes-fraction y for every title's vertical CENTER (not ax.set_title's
    # baseline-anchored pad, which bottom-aligns titles instead of centering them -
    # this way one-line and two-line titles share the same center height).
    TITLE_Y = 1.14

    rng = np.random.default_rng(42)
    metric_iter = iter(flat_metrics)
    legend_axes = []
    blank_axes = []

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if c == DATA_COLS:
                legend_axes.append(ax)
                continue
            metric_cat = next(metric_iter, None)
            if metric_cat is None:
                blank_axes.append(ax)
                continue
            metric, category = metric_cat
            color = CATEGORY_BASE_HUE[category]
            entry = data["metrics"][metric]
            pts = entry["points"]
            human = np.array([p["human"] for p in pts])
            sim = np.array([p["simulation"] for p in pts])

            lo, hi = min(human.min(), sim.min()), max(human.max(), sim.max())
            span = hi - lo or 1.0
            pad = span * 0.14
            if metric in FACET_INTEGER_TICKS:
                ticks = np.array(FACET_INTEGER_TICKS[metric], dtype=float)
            else:
                ticks = _nice_ticks(lo, hi, FACET_N_TICKS)
            tick_step = ticks[1] - ticks[0]
            lo2 = min(lo - pad, ticks[0] - tick_step * 0.1)
            hi2 = max(hi + pad, ticks[-1] + tick_step * 0.1)

            ax.plot([lo2, hi2], [lo2, hi2], color="#c3c2b7", linewidth=1,
                     linestyle=(0, (5, 3)), zorder=1)

            jitter = span * 0.018
            dx = rng.uniform(-jitter, jitter, size=len(pts))
            dy = rng.uniform(-jitter, jitter, size=len(pts))
            for p, jx, jy in zip(pts, dx, dy):
                ax.scatter(
                    p["simulation"] + jx, p["human"] + jy,
                    marker=PRESET_MARKERS[p["preset"]], color=color,
                    s=40, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=2,
                )

            ax.set_xlim(lo2, hi2)
            ax.set_ylim(lo2, hi2)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(_format_tick_value))
            ax.yaxis.set_major_formatter(FuncFormatter(_format_tick_value))
            ax.text(0.5, TITLE_Y, METRIC_LABELS[metric], transform=ax.transAxes,
                    ha="center", va="center", fontsize=15, fontweight="bold", color="#0b0b0b")
            if metric in ("overall_experience", "comfort_level"):
                r_x, r_y, r_ha, r_va = 0.94, 0.06, "right", "bottom"
            else:
                r_x, r_y, r_ha, r_va = 0.06, 0.94, "left", "top"
            ax.text(r_x, r_y, f"r = {entry['pearson_r']:.2f}", transform=ax.transAxes,
                    fontsize=14, color="#52514e", ha=r_ha, va=r_va)
            ax.tick_params(colors="#52514e", labelsize=12, pad=4)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#c3c2b7")

    # Hide axes that hold no panel content: the legend column's unused cells,
    # plus any leftover data cells if the metric count doesn't fill the grid.
    for ax in legend_axes + blank_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Drop the Study (shape) and Category (color) legends into the first two
    # cells of the legend column.
    study_ax, category_ax = legend_axes[0], legend_axes[1]

    study_handles = [
        Line2D([0], [0], marker=PRESET_MARKERS[preset], linestyle="none",
               markerfacecolor="#4b4b4b", markeredgecolor="none", markersize=7)
        for preset in PRESET_LABELS
    ]
    study_labels = [PRESET_LABELS[preset] for preset in PRESET_LABELS]
    study_ax.legend(
        study_handles, study_labels, title="Study Topic", loc="center",
        frameon=False, fontsize=14, title_fontsize=16,
        handletextpad=0.5, labelspacing=0.8, borderaxespad=0,
    )

    category_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=CATEGORY_BASE_HUE[cat], markeredgecolor="white",
               markeredgewidth=0.4, markersize=7)
        for cat in CATEGORY_ORDER
    ]
    category_labels = [CATEGORY_LABELS[cat] for cat in CATEGORY_ORDER]
    category_ax.legend(
        category_handles, category_labels, title="Metric Category", loc="center",
        frameon=False, fontsize=14, title_fontsize=16,
        handletextpad=0.5, labelspacing=0.8, borderaxespad=0,
    )

    fig.supxlabel("Simulation Studies", fontsize=18, color="#0b0b0b", y=0.02)
    fig.supylabel("Human Studies", fontsize=18, color="#0b0b0b", x=0.02)

    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = json.loads(DATA_PATH.read_text())
    plot_facets(data, OUT_DIR / "predictive_validity_facets")
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
