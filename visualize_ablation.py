"""
GARDEN Ablation Study — Visualization Script
=============================================
Generates publication-ready figures from ablation_results/ablation_study_summary.json.

Fix log:
  - Fig 1 (Task Success Rate): now reads 'success_rate' (compute_success-based)
    instead of hard-coded task_success_rate which was always 1.0.
  - Fig 3 (Correction Efficiency): bars were swapped — drifts plotted as
    corrections and vice versa. Fixed by reading total_drifts and
    total_corrections into correctly-named variables before plotting.
  - Fig 4 (Run Success/Failure): now distinguishes successful_tasks from
    total rather than showing all-green bars.
  - Fig 5 (Heatmap): Goal Adherence row now uses avg_alignment (0-1) directly;
    normalization applied correctly so colours are meaningful.
"""

import json
import os
import sys
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ── Style constants ──────────────────────────────────────────────────────── #

COLORS = {
    "baseline":       "#E74C3C",
    "embedding_only": "#1ABC9C",
    "judge_only":     "#3498DB",
    "hybrid":         "#2ECC71",
}
MODE_LABELS = {
    "baseline":       "Baseline",
    "embedding_only": "Embedding Only",
    "judge_only":     "Judge Only",
    "hybrid":         "Hybrid (GARDEN)",
}
MODES = ["baseline", "embedding_only", "judge_only", "hybrid"]


def load_summary(summary_path: str) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def save_fig(fig, output_dir: str, filename: str, dpi: int = 300) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ── Figure 1: Task Success Rate (compute_success-based) ─────────────────── #

def plot_task_success_rate(summary: dict, output_dir: str) -> None:
    by_mode = summary["summary_statistics"]["by_mode"]
    rates   = [by_mode.get(m, {}).get("success_rate", 0.0) * 100 for m in MODES]
    labels  = [MODE_LABELS[m] for m in MODES]
    bar_colors = [COLORS[m] for m in MODES]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, rates, color=bar_colors, edgecolor="black", linewidth=1.2)

    for bar, val in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_title("GARDEN Ablation Study: Task Success Rate by Mode", fontsize=14, fontweight="bold")
    ax.set_xlabel("Agent Mode", fontsize=12)
    ax.set_ylabel("Task Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, output_dir, "01_task_success_rate.png")


# ── Figure 2: Goal Adherence Score ──────────────────────────────────────── #

def plot_goal_adherence(summary: dict, output_dir: str) -> None:
    by_mode = summary["summary_statistics"]["by_mode"]
    scores  = [by_mode.get(m, {}).get("avg_alignment", 0.0) for m in MODES]
    labels  = [MODE_LABELS[m] for m in MODES]
    bar_colors = [COLORS[m] for m in MODES]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, scores, color=bar_colors, edgecolor="black", linewidth=1.2)

    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_title("GARDEN Ablation Study: Goal Adherence (Avg Alignment) by Mode",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Agent Mode", fontsize=12)
    ax.set_ylabel("Average Alignment Score (0–1)", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.65, color="red", linestyle="--", linewidth=1.5, label="Success threshold (0.65)")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, output_dir, "02_goal_adherence.png")


# ── Figure 3: Drift Detection & Correction Efficiency ───────────────────── #

def plot_correction_efficiency(summary: dict, output_dir: str) -> None:
    """
    Grouped bar chart: drifts detected vs corrections applied per mode.
    Previously these were swapped; now explicitly read from the correct keys.
    """
    by_mode     = summary["summary_statistics"]["by_mode"]
    drifts      = [by_mode.get(m, {}).get("total_drifts",      0) for m in MODES]
    corrections = [by_mode.get(m, {}).get("total_corrections", 0) for m in MODES]
    labels      = [MODE_LABELS[m] for m in MODES]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars_d = ax.bar(x - width / 2, drifts,      width, label="Drifts Detected",    color="#E74C3C", edgecolor="black")
    bars_c = ax.bar(x + width / 2, corrections, width, label="Corrections Applied", color="#1ABC9C", edgecolor="black")

    for bar, val in zip(bars_d, drifts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontweight="bold", fontsize=10)
    for bar, val in zip(bars_c, corrections):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_title("GARDEN Ablation Study: Drift Detection & Correction Efficiency",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Agent Mode", fontsize=12)
    ax.set_ylabel("Count (across all 10 tasks)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, output_dir, "03_correction_efficiency.png")


# ── Figure 4: Run Success / Failure ─────────────────────────────────────── #

def plot_run_success(summary: dict, output_dir: str) -> None:
    """
    Stacked bar chart showing successful vs failed tasks per mode.
    Uses compute_success counts so the chart is informative.
    """
    by_mode     = summary["summary_statistics"]["by_mode"]
    total_tasks = summary["study_metadata"]["total_tasks"]

    successes = [by_mode.get(m, {}).get("successful_tasks", 0) for m in MODES]
    failures  = [total_tasks - s for s in successes]
    labels    = [MODE_LABELS[m] for m in MODES]

    x     = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_s = ax.bar(x, successes, width, label="Successful Tasks", color="#2ECC71", edgecolor="black")
    bars_f = ax.bar(x, failures,  width, bottom=successes, label="Failed Tasks", color="#E74C3C", edgecolor="black", alpha=0.8)

    for bar, val in zip(bars_s, successes):
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                str(val), ha="center", va="center", fontweight="bold", fontsize=12, color="white")
    for bar, base, val in zip(bars_f, successes, failures):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, base + val / 2,
                    str(val), ha="center", va="center", fontweight="bold", fontsize=12, color="white")

    ax.set_title("GARDEN Ablation Study: Task Outcomes by Mode (Quality-Gated)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Agent Mode", fontsize=12)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, total_tasks + 2)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, output_dir, "04_run_success_rate.png")


# ── Figure 5: Normalised Performance Heatmap ────────────────────────────── #

def plot_heatmap(summary: dict, output_dir: str) -> None:
    """
    3-row heatmap: success rate, avg alignment (0-1), correction efficiency.
    All rows normalised to [0, 1] so colour scale is meaningful.
    Correction efficiency = corrections / (drifts + 1) to avoid div-by-zero.
    """
    by_mode = summary["summary_statistics"]["by_mode"]

    success_rates = [by_mode.get(m, {}).get("success_rate",   0.0) for m in MODES]
    alignments    = [by_mode.get(m, {}).get("avg_alignment",  0.0) for m in MODES]
    drifts        = [by_mode.get(m, {}).get("total_drifts",   0)   for m in MODES]
    corrections   = [by_mode.get(m, {}).get("total_corrections", 0) for m in MODES]

    # Correction efficiency: fraction of drifts that were corrected
    corr_eff = [c / (d + 1e-9) for c, d in zip(corrections, drifts)]
    # Clip to [0, 1] in case corrections > drifts (e.g. proactive corrections)
    corr_eff = [min(1.0, v) for v in corr_eff]

    data = np.array([success_rates, alignments, corr_eff])

    row_labels = ["Task Success Rate", "Avg Alignment (0–1)", "Correction Efficiency"]
    col_labels = [MODE_LABELS[m] for m in MODES]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", fontweight="bold", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Score (0–1)", rotation=270, labelpad=15, fontsize=10)

    ax.set_title("GARDEN Ablation Study: Normalized Performance Heatmap",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir, "05_performance_heatmap.png")


# ── Main ─────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Generate GARDEN ablation study visualizations"
    )
    parser.add_argument(
        "--results-dir", default="ablation_results",
        help="Directory containing ablation_study_summary.json",
    )
    parser.add_argument(
        "--output-dir", default="visualizations",
        help="Directory to save PNG figures",
    )
    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    summary_path = os.path.join(args.results_dir, "ablation_study_summary.json")
    if not os.path.exists(summary_path):
        print(f"❌ Summary file not found: {summary_path}")
        print("   Run: python main.py --mode ablation  first.")
        sys.exit(1)

    print(f"\n📊 Loading results from: {summary_path}")
    summary = load_summary(summary_path)

    print(f"\n🖼️  Generating figures → {args.output_dir}/")
    plot_task_success_rate(summary,    args.output_dir)
    plot_goal_adherence(summary,       args.output_dir)
    plot_correction_efficiency(summary, args.output_dir)
    plot_run_success(summary,          args.output_dir)
    plot_heatmap(summary,              args.output_dir)

    print(f"\n✅ All 5 figures saved to '{args.output_dir}/' at 300 DPI.")
    print("   Include them in your paper's results section.")


if __name__ == "__main__":
    main()