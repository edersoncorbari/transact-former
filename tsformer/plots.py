"""
tsFormer — plots.py

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

Visualisation utilities for all figures in the paper.

Requires: matplotlib (optional — functions degrade gracefully if absent)

Functions
─────────
plot_roc_curves          Figure: ROC curves for late fusion vs tsFormer
plot_pr_curves           Figure: Precision-Recall curves
plot_model_size_scaling  Figure: AUC gain vs # parameters  (Table 3 / Figure 6)
plot_context_scaling     Figure: AUC vs context length per model size  (Figure 6)
plot_data_volume_scaling Figure: AUC vs data volume  (Figure 7)
plot_oot_stability       Figure: Out-of-time AUC stability  (Figure 8)
plot_embedding_space     Figure: UMAP/PCA projection of user embeddings

Usage
─────
    from tsformer.plots import plot_roc_curves
    import json

    roc = json.load(open("results/roc_curves.json"))
    fig = plot_roc_curves(roc["late_fusion"], roc["tsformer"])
    fig.savefig("roc.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Optional

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend (safe for servers)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting.  "
            "Install it with:  pip install matplotlib"
        )


# ── Style constants (matches Nubank-ish palette) ──────────────────────────────
_COLORS = {
    "tsformer": "#8A3FFC",  # purple
    "late_fusion": "#0F62FE",  # blue
    "baseline": "#6F6F6F",  # grey
    "accent": "#FF7EB6",  # pink
    "small": "#0F62FE",
    "medium": "#42BE65",
    "large": "#8A3FFC",
}
_FIGSIZE = (7, 5)
_FONTSIZE = 11


def _setup_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=_FONTSIZE + 1, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=_FONTSIZE - 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


# ──────────────────────────────────────────────────────────────────────────────
# ROC / PR curves
# ──────────────────────────────────────────────────────────────────────────────


def plot_roc_curves(
    late_fusion_roc: dict,
    tsformer_roc: dict,
    title: str = "ROC Curve — Late Fusion vs tsFormer",
    output_path: Optional[str] = None,
):
    """
    Plot ROC curves for both models.

    Args
    ────
    late_fusion_roc / tsformer_roc  : dicts with keys fpr, tpr, auc (output of evaluate.compute_roc_curve)
    output_path                     : if given, save figure to this path

    Returns matplotlib Figure.
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    ax.plot(
        late_fusion_roc["fpr"],
        late_fusion_roc["tpr"],
        color=_COLORS["late_fusion"],
        lw=2,
        label=f"Late Fusion  (AUC = {late_fusion_roc['auc']:.4f})",
    )
    ax.plot(
        tsformer_roc["fpr"],
        tsformer_roc["tpr"],
        color=_COLORS["tsformer"],
        lw=2,
        label=f"tsFormer     (AUC = {tsformer_roc['auc']:.4f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color=_COLORS["baseline"],
        lw=1.2,
        linestyle="--",
        label="Random baseline",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _setup_ax(ax, title, "False Positive Rate", "True Positive Rate")
    ax.legend(fontsize=_FONTSIZE - 1, loc="lower right")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    return fig


def plot_pr_curves(
    late_fusion_pr: dict,
    tsformer_pr: dict,
    title: str = "Precision-Recall — Late Fusion vs tsFormer",
    output_path: Optional[str] = None,
):
    """Plot PR curves. Args same structure as plot_roc_curves."""
    _require_mpl()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    ax.plot(
        late_fusion_pr["recall"],
        late_fusion_pr["precision"],
        color=_COLORS["late_fusion"],
        lw=2,
        label="Late Fusion",
    )
    ax.plot(
        tsformer_pr["recall"],
        tsformer_pr["precision"],
        color=_COLORS["tsformer"],
        lw=2,
        label="tsFormer",
    )

    _setup_ax(ax, title, "Recall", "Precision")
    ax.legend(fontsize=_FONTSIZE - 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Model size scaling  (Table 3 / Figure 6 left)
# ──────────────────────────────────────────────────────────────────────────────


def plot_model_size_scaling(
    results: list[dict],
    title: str = "tsFormer AUC vs Model Size  (Table 3)",
    output_path: Optional[str] = None,
):
    """
    Bar chart of AUC gain vs model parameter count.

    results: list of dicts with keys n_params, auc, label
             (output of scaling_analysis.experiment_model_size)
    """
    _require_mpl()
    baseline = results[0]["auc"]
    labels = [r["label"].strip() for r in results]
    gains = [r["auc"] - baseline for r in results]
    n_params = [r["n_params"] / 1e6 for r in results]  # in millions

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: AUC gain bar chart
    ax = axes[0]
    colors = [_COLORS["tsformer"] if g >= 0 else _COLORS["baseline"] for g in gains]
    bars = ax.bar(labels, gains, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=_COLORS["baseline"], lw=1, linestyle="--")
    for bar, g in zip(bars, gains):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0002,
            f"{g:+.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _setup_ax(ax, "AUC Gain vs Smallest Model", "Model Variant", "Δ AUC")
    ax.tick_params(axis="x", rotation=20)

    # Right: AUC vs # parameters scatter
    ax2 = axes[1]
    aucs = [r["auc"] for r in results]
    ax2.scatter(n_params, aucs, s=120, color=_COLORS["tsformer"], zorder=3)
    ax2.plot(n_params, aucs, color=_COLORS["tsformer"], lw=1.5, alpha=0.6)
    for x, y, lbl in zip(n_params, aucs, labels):
        ax2.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    _setup_ax(ax2, "AUC vs # Parameters", "Parameters (M)", "Test AUC")

    fig.suptitle(title, fontsize=_FONTSIZE + 2, fontweight="bold", y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Context length scaling  (Figure 6)
# ──────────────────────────────────────────────────────────────────────────────


def plot_context_scaling(
    results_by_size: dict[str, list[dict]],
    title: str = "tsFormer AUC vs Context Length  (Figure 6)",
    output_path: Optional[str] = None,
):
    """
    Line chart: AUC vs context length, one line per model size.

    results_by_size: output of scaling_analysis.experiment_context_length
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    color_cycle = list(_COLORS.values())
    for i, (size_name, records) in enumerate(results_by_size.items()):
        ctx_lens = [r["context_length"] for r in records]
        aucs = [r["auc"] for r in records]
        color = color_cycle[i % len(color_cycle)]
        ax.plot(
            ctx_lens,
            aucs,
            marker="o",
            lw=2,
            color=color,
            label=size_name.strip(),
            markersize=7,
        )

    ax.xaxis.set_major_locator(
        mticker.FixedLocator(
            sorted(
                {r["context_length"] for recs in results_by_size.values() for r in recs}
            )
        )
    )
    _setup_ax(ax, title, "Context Length (tokens)", "Test AUC")
    ax.legend(fontsize=_FONTSIZE - 1)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Data volume scaling  (Figure 7)
# ──────────────────────────────────────────────────────────────────────────────


def plot_data_volume_scaling(
    results_by_size: dict[str, list[dict]],
    title: str = "tsFormer AUC vs Training Data Volume  (Figure 7)",
    output_path: Optional[str] = None,
):
    """
    Line chart: AUC vs # training rows, one line per model size.

    results_by_size: output of scaling_analysis.experiment_data_volume
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    color_cycle = list(_COLORS.values())
    for i, (size_name, records) in enumerate(results_by_size.items()):
        ns = [r["n_used"] for r in records]
        aucs = [r["auc"] for r in records]
        color = color_cycle[i % len(color_cycle)]
        ax.plot(
            ns,
            aucs,
            marker="o",
            lw=2,
            color=color,
            label=size_name.strip(),
            markersize=7,
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _setup_ax(ax, title, "Training Rows", "Test AUC")
    ax.legend(fontsize=_FONTSIZE - 1)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Out-of-time stability  (Figure 8)
# ──────────────────────────────────────────────────────────────────────────────


def plot_oot_stability(
    weekly_results: dict,
    title: str = "Out-of-Time AUC Stability  (Figure 8)",
    output_path: Optional[str] = None,
):
    """
    Line chart of relative AUC per week after the training period.

    weekly_results: output of evaluate.evaluate_out_of_time_stability
                    (dict with keys week, relative_auc, n)
    """
    _require_mpl()
    weeks = weekly_results["week"]
    rel = weekly_results["relative_auc"]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(weeks, rel, marker="o", lw=2, color=_COLORS["tsformer"], markersize=7)
    ax.axhline(
        0,
        color=_COLORS["baseline"],
        lw=1.2,
        linestyle="--",
        label="Baseline (relative = 0)",
    )
    ax.fill_between(weeks, rel, 0, alpha=0.12, color=_COLORS["tsformer"])

    _setup_ax(ax, title, "Weeks After Training Period", "Relative AUC vs Baseline")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.3f}"))
    ax.legend(fontsize=_FONTSIZE - 1)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Embedding space projection  (optional — requires umap-learn or sklearn)
# ──────────────────────────────────────────────────────────────────────────────


def plot_embedding_space(
    embeddings: list[list[float]],
    labels: list[int],
    # member_ids: Optional[list[str]] = None,
    method: str = "pca",  # "pca" | "umap"
    title: str = "User Embedding Space",
    output_path: Optional[str] = None,
    n_samples: int = 1000,
):
    """
    2-D projection of user embeddings coloured by label (0 / 1).

    embeddings : list of float lists (output of EmbeddingExtractor.embed_batch)
    labels     : binary labels (0 = negative, 1 = positive)
    method     : "pca" (always available) or "umap" (requires umap-learn)
    """
    _require_mpl()
    import numpy as np

    emb_arr = np.array(embeddings[:n_samples])
    lbl_arr = np.array(labels[:n_samples])

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(emb_arr)
            method_label = "UMAP"
        except ImportError:
            print("  umap-learn not installed — falling back to PCA")
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(emb_arr)
        method_label = "PCA"

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for lbl, color, name in [
        (0, _COLORS["baseline"], "Negative"),
        (1, _COLORS["tsformer"], "Positive"),
    ]:
        mask = lbl_arr == lbl
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=f"{name} (n={mask.sum():,})",
            s=18,
            alpha=0.6,
            edgecolors="none",
        )

    _setup_ax(
        ax,
        f"{title}  [{method_label}]",
        f"{method_label} Component 1",
        f"{method_label} Component 2",
    )
    ax.legend(fontsize=_FONTSIZE - 1)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: save all result figures from a run
# ──────────────────────────────────────────────────────────────────────────────


def save_all_figures(results_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Load results from a completed run and save all figures.

    results_dir : directory containing roc_curves.json, full_results.json
    output_dir  : where to write .png files  (default: results_dir/figures/)
    """
    import json
    from pathlib import Path

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ROC curves
    roc_path = results_dir / "roc_curves.json"
    if roc_path.exists():
        roc = json.loads(roc_path.read_text())
        plot_roc_curves(
            roc["late_fusion"],
            roc["tsformer"],
            output_path=str(output_dir / "roc_curves.png"),
        )
        plot_pr_curves(
            roc.get("pr_late_fusion", {}),
            roc.get("pr_tsformer", {}),
            output_path=str(output_dir / "pr_curves.png"),
        )

    # Scaling results (if present)
    sc_path = results_dir / "scaling_results.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        if "model_size" in sc:
            plot_model_size_scaling(
                sc["model_size"],
                output_path=str(output_dir / "model_size_scaling.png"),
            )
        if "context_length" in sc:
            plot_context_scaling(
                sc["context_length"],
                output_path=str(output_dir / "context_scaling.png"),
            )
        if "data_volume" in sc:
            plot_data_volume_scaling(
                sc["data_volume"],
                output_path=str(output_dir / "data_volume_scaling.png"),
            )

    print(f"\n  All figures saved → {output_dir}/")
