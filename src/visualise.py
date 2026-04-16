"""
MCBench visualisation suite.

Three plots designed to showcase the benchmark's discriminative power:

  plot1_capability_landscape(leaderboard_df)
      d′ × M-ratio scatter for all solo models.  Quadrant labels expose
      the geometry that standard benchmarks (d′ only) cannot see.

  plot2_esma_trajectory(esma_results)
      M-ratio across ESMA epochs with dual y-axis for d′, NES variance band,
      and "do-no-harm" proof that accuracy is preserved.

  plot3_full_picture(leaderboard_df, esma_results)
      Combined: solo models + ESMA path + MetaMind arrow, all on d′ × M-ratio.
      The complete story in a single figure.

After a Kaggle run, pull the output CSV and pass it here:

    kaggle kernels output ivogeorg/mcbench-metacognition -p /tmp/output
    lb = pd.read_csv('/tmp/output/mcbench_leaderboard.csv', index_col='rank')
    plot1_capability_landscape(lb, save='plot1.png')
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

# ── Style constants ──────────────────────────────────────────────────────────

_BG        = "#0d1117"   # GitHub dark background
_GRID      = "#21262d"
_TEXT      = "#e6edf3"
_MUTED     = "#8b949e"
_ACCENT    = "#f0883e"   # orange: MetaMind / highlights
_ESMA_COL  = "#3fb950"   # green: ESMA trajectory

# Model family colours (matched to known proxy models)
_FAMILY_COLORS = {
    "gemini":   "#4285f4",   # Google blue
    "llama":    "#e07b39",   # Meta orange
    "mistral":  "#9c6ade",   # Mistral purple
    "qwen":     "#06aed5",   # Qwen cyan
    "deepseek": "#f78c6c",   # DeepSeek coral
    "gpt":      "#74aa9c",   # OpenAI teal
    "claude":   "#d4a76a",   # Anthropic sand
    "default":  "#8b949e",   # fallback grey
}

_QUADRANT_LABELS = [
    # (x_frac, y_frac, text, ha)  — in axes coordinates
    (0.74, 0.88, "Capable\n& Self-Aware",    "center"),
    (0.74, 0.12, "Capable but\nOverconfident", "center"),
    (0.14, 0.88, "Uncertain but\nCalibrated",  "center"),
    (0.14, 0.12, "Low Capability\nLow Awareness", "center"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _family_color(model_name: str) -> str:
    n = model_name.lower()
    for key, col in _FAMILY_COLORS.items():
        if key in n:
            return col
    return _FAMILY_COLORS["default"]


def _dark_axes(ax, title: str = ""):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.6, zorder=0)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=_TEXT, pad=8)


def _annotate_model(ax, x, y, name, offset=(5, 5), fontsize=7):
    """Annotate a scatter point with model name and a subtle drop shadow."""
    short = name.replace("-instruct", "").replace("-it", "")
    ax.annotate(
        short, (x, y),
        xytext=offset, textcoords="offset points",
        fontsize=fontsize, color=_TEXT, alpha=0.85,
        path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG)],
    )


def _add_quadrant_guides(ax, d_ref=1.0, m_ref=1.0):
    """Reference cross-hairs and quadrant background shading."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.axvline(d_ref, color=_MUTED, linewidth=0.8, linestyle="--", zorder=1, alpha=0.6)
    ax.axhline(m_ref, color=_MUTED, linewidth=0.8, linestyle="--", zorder=1, alpha=0.6)

    # Subtle quadrant shading
    ax.fill_betweenx([m_ref, ylim[1]], xlim[0], d_ref, color="#1c2a1e", alpha=0.3, zorder=0)
    ax.fill_betweenx([ylim[0], m_ref], d_ref, xlim[1], color="#2a1c1c", alpha=0.3, zorder=0)
    ax.fill_betweenx([m_ref, ylim[1]], d_ref, xlim[1], color="#1c2a24", alpha=0.4, zorder=0)

    for xf, yf, text, ha in _QUADRANT_LABELS:
        x = xlim[0] + xf * (xlim[1] - xlim[0])
        y = ylim[0] + yf * (ylim[1] - ylim[0])
        ax.text(x, y, text, ha=ha, va="center", fontsize=7,
                color=_MUTED, alpha=0.7, style="italic",
                path_effects=[pe.withStroke(linewidth=1, foreground=_BG)])


# ── Plot 1: Capability landscape ─────────────────────────────────────────────

def plot1_capability_landscape(
    leaderboard: pd.DataFrame,
    save: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    d′ × M-ratio scatter for all solo models.

    Columns required in leaderboard: model, d_prime, m_ratio, srs, mci.
    """
    # Exclude MetaMind from this solo-models plot
    df = leaderboard[~leaderboard["model"].str.contains("MetaMind", case=False)].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(_BG)
    _dark_axes(ax, "MCBench: Capability Landscape — d′ vs M-ratio")

    # Scale point size by SRS (normalise to 60–200 pt²)
    srs_vals = df["srs"].fillna(0.333).values
    sizes = 60 + 140 * (srs_vals - srs_vals.min()) / max(np.ptp(srs_vals), 0.01)

    for _, row in df.iterrows():
        col = _family_color(row["model"])
        ax.scatter(
            row["d_prime"], row["m_ratio"],
            s=sizes[df.index.get_loc(_)],
            color=col, edgecolors=_BG, linewidths=0.8,
            alpha=0.92, zorder=5,
        )
        offset = (6, 4) if row["d_prime"] < df["d_prime"].median() else (-6, 4)
        _annotate_model(ax, row["d_prime"], row["m_ratio"], row["model"], offset=offset)

    # Axis limits with padding
    dp_pad = df["d_prime"].pipe(np.ptp) * 0.15
    mr_pad = max(df["m_ratio"].pipe(np.ptp) * 0.2, 0.3)
    ax.set_xlim(df["d_prime"].min() - dp_pad, df["d_prime"].max() + dp_pad)
    ax.set_ylim(min(df["m_ratio"].min() - mr_pad, 0),
                df["m_ratio"].max() + mr_pad)

    _add_quadrant_guides(ax, d_ref=df["d_prime"].median(), m_ref=1.0)

    ax.set_xlabel("d′  (Type 1 sensitivity — factual accuracy)", fontsize=9)
    ax.set_ylabel("M-ratio  (meta-d′/d′ — metacognitive efficiency)", fontsize=9)

    # Size legend for SRS
    for srs_val, label in [(0.34, "SRS≈0.34\n(chance)"), (0.55, "SRS=0.55"), (0.70, "SRS=0.70")]:
        s = 60 + 140 * (srs_val - srs_vals.min()) / max(np.ptp(srs_vals), 0.01)
        ax.scatter([], [], s=s, color=_MUTED, label=label, alpha=0.8)

    # Family colour legend
    families_present = {_family_color(r["model"]): r["model"].split("-")[0].title()
                        for _, r in df.iterrows()}
    patches = [mpatches.Patch(color=c, label=n) for c, n in families_present.items()]

    l1 = ax.legend(handles=patches, loc="upper left", fontsize=7,
                   facecolor=_GRID, edgecolor=_MUTED, labelcolor=_TEXT,
                   title="Model family", title_fontsize=7)
    ax.add_artist(l1)
    ax.legend(loc="lower right", fontsize=7, facecolor=_GRID,
              edgecolor=_MUTED, labelcolor=_TEXT,
              title="Point size = SRS", title_fontsize=7)

    plt.tight_layout(pad=1.2)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"Saved: {save}")
    if show:
        plt.show()
    return fig


# ── Plot 2: ESMA trajectory ───────────────────────────────────────────────────

def plot2_esma_trajectory(
    esma_results: dict,
    save: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    M-ratio across ESMA epochs with NES variance band and d′ stability track.

    esma_results must have keys: m_ratio_trajectory, all_rewards, baseline_m_ratio.
    If d_prime_trajectory is present, it is plotted on the right y-axis.
    """
    traj = esma_results["m_ratio_trajectory"]          # [epoch_0, epoch_1, ...]
    rewards = esma_results.get("all_rewards", [])      # [[r1,r2,...], ...] per epoch
    baseline = esma_results["baseline_m_ratio"]
    d_traj = esma_results.get("d_prime_trajectory")    # optional

    epochs = list(range(len(traj)))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_BG)
    _dark_axes(ax1, "ESMA: Metacognitive Alignment of Gemma 4 E2B — M-ratio Trajectory")

    # ── NES population band ──────────────────────────────────────────────────
    if rewards:
        pop_epochs = list(range(1, len(rewards) + 1))
        pop_max = [np.max(r) for r in rewards]
        pop_min = [np.min(r) for r in rewards]
        pop_mean = [np.mean(r) for r in rewards]
        ax1.fill_between(pop_epochs, pop_min, pop_max,
                         color=_ESMA_COL, alpha=0.12, label="NES population range")
        ax1.plot(pop_epochs, pop_mean, color=_ESMA_COL, linewidth=0.8,
                 linestyle=":", alpha=0.6, label="Population mean")

    # ── M-ratio trajectory ───────────────────────────────────────────────────
    ax1.plot(epochs, traj, "o-", color=_ESMA_COL, linewidth=2.2,
             markersize=6, zorder=6, label="M-ratio (parent model)")
    ax1.axhline(baseline, color=_MUTED, linewidth=1, linestyle="--",
                alpha=0.7, label=f"Baseline = {baseline:.3f}")
    ax1.axhline(1.0, color=_ACCENT, linewidth=0.8, linestyle="--",
                alpha=0.5, label="Ideal M-ratio = 1.0")

    # Annotate best epoch
    best_idx = int(np.argmax(traj))
    best_val = traj[best_idx]
    ax1.annotate(
        f"Best: {best_val:.3f}\n(epoch {best_idx})",
        (best_idx, best_val),
        xytext=(10, 8), textcoords="offset points",
        fontsize=7.5, color=_ESMA_COL, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=_ESMA_COL, lw=0.8),
        path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG)],
    )

    ax1.set_xlabel("Epoch  (0 = pre-ESMA baseline)", fontsize=9)
    ax1.set_ylabel("M-ratio  (meta-d′/d′)", fontsize=9, color=_ESMA_COL)
    ax1.tick_params(axis="y", colors=_ESMA_COL)

    # ── d′ stability (right axis) ────────────────────────────────────────────
    if d_traj:
        ax2 = ax1.twinx()
        _dark_axes(ax2)
        ax2.plot(range(len(d_traj)), d_traj, "s--",
                 color="#a8c5da", linewidth=1.2, markersize=4,
                 alpha=0.7, label="d′ (factual accuracy)")
        ax2.set_ylabel("d′  (Type 1 sensitivity)", fontsize=9, color="#a8c5da")
        ax2.tick_params(axis="y", colors="#a8c5da")
        ax2.legend(loc="upper right", fontsize=7, facecolor=_GRID,
                   edgecolor=_MUTED, labelcolor=_TEXT)

    ax1.legend(loc="lower right" if d_traj is None else "lower left",
               fontsize=7, facecolor=_GRID, edgecolor=_MUTED, labelcolor=_TEXT)

    # ΔMCI annotation
    improvement = best_val - baseline
    ax1.text(0.02, 0.96,
             f"Δ M-ratio = {improvement:+.4f}  ({improvement/max(abs(baseline),0.001)*100:+.1f}%)",
             transform=ax1.transAxes, fontsize=8.5, color=_ESMA_COL,
             va="top", fontweight="bold",
             path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG)])

    plt.tight_layout(pad=1.2)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"Saved: {save}")
    if show:
        plt.show()
    return fig


# ── Plot 3: Full picture ──────────────────────────────────────────────────────

def plot3_full_picture(
    leaderboard: pd.DataFrame,
    esma_results: dict | None = None,
    save: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Combined: solo models + ESMA trajectory waypoints + MetaMind arrow.

    Shows the complete story in one figure:
    - Where models sit in d′ × M-ratio space (capability vs awareness)
    - ESMA moves a model *up* the M-ratio axis without shifting d′
    - MetaMind moves its base model *up* the M-ratio axis (architecture effect)

    leaderboard : must include MetaMind row if it was run
    esma_results : output dict from run_esma(); pass None to omit
    """
    solo = leaderboard[~leaderboard["model"].str.contains("MetaMind", case=False)].copy()
    meta_rows = leaderboard[leaderboard["model"].str.contains("MetaMind", case=False)]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(_BG)
    _dark_axes(ax, "MCBench: Factual Capability vs Metacognitive Efficiency")

    srs_vals = solo["srs"].fillna(0.333).values
    srs_range = max(np.ptp(srs_vals), 0.01)

    # ── Solo models ──────────────────────────────────────────────────────────
    for i, (_, row) in enumerate(solo.iterrows()):
        col = _family_color(row["model"])
        s = 70 + 130 * (srs_vals[i] - srs_vals.min()) / srs_range
        ax.scatter(row["d_prime"], row["m_ratio"],
                   s=s, color=col, edgecolors=_BG, linewidths=0.8,
                   alpha=0.88, zorder=5)
        _annotate_model(ax, row["d_prime"], row["m_ratio"], row["model"])

    # ── ESMA waypoints ───────────────────────────────────────────────────────
    if esma_results is not None:
        traj = esma_results["m_ratio_trajectory"]
        d_traj = esma_results.get("d_prime_trajectory")
        esma_model_name = esma_results.get("model_name", "gemma-4-e2b-it")

        # Match ESMA start to the solo leaderboard entry for that model
        esma_solo_row = solo[solo["model"].str.contains(
            esma_model_name.split("/")[-1].replace("-it", ""), case=False
        )]
        base_d = esma_solo_row.iloc[0]["d_prime"] if not esma_solo_row.empty else 0.5
        base_m = traj[0]  # epoch 0 = pre-ESMA baseline

        # Choose waypoints: epoch 0, median epoch, best epoch
        best_ep = int(np.argmax(traj))
        mid_ep  = max(1, len(traj) // 2)
        waypoints = {
            "ESMA\nBaseline": (base_d, base_m, 0),
            f"ESMA\nEpoch {mid_ep}": (
                (d_traj[mid_ep] if d_traj else base_d), traj[mid_ep], mid_ep
            ),
            f"ESMA\nBest (ep{best_ep})": (
                (d_traj[best_ep] if d_traj else base_d), traj[best_ep], best_ep
            ),
        }

        wp_items = list(waypoints.items())
        for k, (label, (dx, mx, ep)) in enumerate(wp_items):
            ax.scatter(dx, mx, s=140, color=_ESMA_COL,
                       marker="D", edgecolors=_BG, linewidths=1, zorder=8)
            _annotate_model(ax, dx, mx, label, offset=(6, -10), fontsize=7)

        # Trajectory arrows between waypoints
        for k in range(len(wp_items) - 1):
            _, (x0, y0, _) = wp_items[k]
            _, (x1, y1, _) = wp_items[k + 1]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=_ESMA_COL,
                                        lw=1.4, connectionstyle="arc3,rad=0.15"))

    # ── MetaMind arrow ───────────────────────────────────────────────────────
    for _, meta_row in meta_rows.iterrows():
        base_name = meta_row.get("base_model", "")
        base_solo = solo[solo["model"] == base_name]
        if base_solo.empty:
            continue

        bx, by = base_solo.iloc[0]["d_prime"], base_solo.iloc[0]["m_ratio"]
        mx, my = meta_row["d_prime"], meta_row["m_ratio"]

        ax.scatter(mx, my, s=220, color=_ACCENT,
                   marker="*", edgecolors=_BG, linewidths=0.8, zorder=9)
        ax.annotate(
            "MetaMind\n(3-agent)", (mx, my),
            xytext=(8, 6), textcoords="offset points",
            fontsize=7.5, color=_ACCENT, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG)],
        )
        # Arrow from base model to MetaMind
        ax.annotate("", xy=(mx, my), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="-|>", color=_ACCENT,
                                    lw=1.8, linestyle="dashed",
                                    connectionstyle="arc3,rad=-0.2"))
        delta = my - by
        ax.text((bx + mx) / 2 + 0.05, (by + my) / 2 + 0.05,
                f"ΔMCI={delta:+.3f}", fontsize=7, color=_ACCENT,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG)])

    # ── Axis limits, guides, labels ──────────────────────────────────────────
    all_d = pd.concat([solo["d_prime"],
                       pd.Series([meta_row["d_prime"] for _, meta_row in meta_rows.iterrows()]
                                 if not meta_rows.empty else [])
                       ])
    all_m = pd.concat([solo["m_ratio"],
                       pd.Series([meta_row["m_ratio"] for _, meta_row in meta_rows.iterrows()]
                                 if not meta_rows.empty else [])
                       ])
    if esma_results:
        all_m = pd.concat([all_m, pd.Series(esma_results["m_ratio_trajectory"])])

    dp_pad = np.ptp(all_d) * 0.18
    mr_pad = max(np.ptp(all_m) * 0.22, 0.4)
    ax.set_xlim(all_d.min() - dp_pad, all_d.max() + dp_pad)
    ax.set_ylim(min(all_m.min() - mr_pad, 0), all_m.max() + mr_pad)

    _add_quadrant_guides(ax, d_ref=solo["d_prime"].median(), m_ref=1.0)

    ax.set_xlabel("d′  (Type 1 sensitivity — factual accuracy)", fontsize=9)
    ax.set_ylabel("M-ratio  (meta-d′/d′ — metacognitive efficiency)", fontsize=9)

    # Legend
    legend_items = [
        mpatches.Patch(color=_MUTED,    label="Solo models (size ∝ SRS)"),
        mpatches.Patch(color=_ESMA_COL, label="ESMA checkpoints (Gemma 4 E2B)"),
        mpatches.Patch(color=_ACCENT,   label="MetaMind-3Agent ★"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8,
              facecolor=_GRID, edgecolor=_MUTED, labelcolor=_TEXT)

    plt.tight_layout(pad=1.2)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"Saved: {save}")
    if show:
        plt.show()
    return fig


# ── Convenience: generate all three from Kaggle output files ─────────────────

def generate_all(
    leaderboard_csv: str | Path,
    esma_json: str | Path | None = None,
    output_dir: str | Path = ".",
    show: bool = False,
) -> None:
    """
    Load results from Kaggle output files and generate all three plots.

    Usage after `kaggle kernels output`:
        from src.visualise import generate_all
        generate_all(
            leaderboard_csv='/tmp/output/mcbench_leaderboard.csv',
            esma_json='/tmp/output/esma_results.json',   # optional
            output_dir='media/',
        )
    """
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lb = pd.read_csv(leaderboard_csv, index_col="rank")

    esma = None
    if esma_json and Path(esma_json).exists():
        with open(esma_json) as f:
            esma = json.load(f)

    plot1_capability_landscape(lb, save=out / "plot1_landscape.png", show=show)
    if esma:
        plot2_esma_trajectory(esma, save=out / "plot2_esma.png", show=show)
    plot3_full_picture(lb, esma, save=out / "plot3_full.png", show=show)

    print(f"\nAll plots saved to {out}/")
