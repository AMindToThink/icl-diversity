"""Test C × a_∞ as diversity score on Tevet evaluation data.

The hypothesis: in Regime 2 (few samples, no repeated modes), the asymptotic
surprise floor a_∞ carries the diversity signal — diverse responses stay
surprising after conditioning, paraphrases become predictable.

Computes C × a_∞ and compares to C × E (current paper score) using:
- Spearman ρ (all datasets)
- ROC AUC (binary datasets: McDiv_nuggets, ConTest)
- ROC curves

All metrics derived from existing sidecar data — no model re-runs needed.

Usage:
    uv run python scripts/analyze_c_ainf.py --run-tag qwen25_completion_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = PROJECT_ROOT / "results" / "tevet"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ak_curves import exponential_ak, fit_exponential

# Import OCA from Tevet's diversity-eval codebase
sys.path.insert(0, str(PROJECT_ROOT / "diversity-eval"))
from utils import optimal_classification_accuracy


# ---------------------------------------------------------------------------
# Data loading (reuse patterns from analyze_per_byte.py)
# ---------------------------------------------------------------------------

def load_sidecar(sidecar_path: Path) -> dict[str, dict]:
    with open(sidecar_path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("__")}


def load_csv_rows(csv_path: Path) -> dict[str, dict]:
    """Load CSV rows keyed by sidecar_key (falls back to sample_id for old CSVs)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Use sidecar_key if present (handles duplicate sample_ids), else sample_id
    key_col = "sidecar_key" if rows and "sidecar_key" in rows[0] else "sample_id"
    return {row[key_col]: row for row in rows}


def compute_per_byte_curve(entry: dict) -> list[float] | None:
    """Mean of (bits_k / bytes_k) across permutations — correct mean-of-ratios."""
    perm_curves = entry.get("per_permutation_a_k_curves")
    perm_bytes = entry.get("per_permutation_byte_counts")
    if perm_curves is None or perm_bytes is None:
        return entry.get("a_k_curve_per_byte")
    n_positions = len(perm_curves[0])
    per_byte_curve: list[float] = []
    for k_idx in range(n_positions):
        ratios = []
        for p in range(len(perm_curves)):
            bits = perm_curves[p][k_idx]
            bc = perm_bytes[p][k_idx]
            if bc > 0:
                ratios.append(bits / bc)
        per_byte_curve.append(sum(ratios) / len(ratios) if ratios else 0.0)
    return per_byte_curve


# ---------------------------------------------------------------------------
# ROC / AUC (no sklearn dependency)
# ---------------------------------------------------------------------------

def compute_roc_curve(
    scores: list[float], labels: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (fpr, tpr) arrays for plotting."""
    order = np.argsort(-np.array(scores))
    sorted_labels = np.array(labels)[order]
    n_pos = int(sum(lab == 1.0 for lab in labels))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative labels for ROC")
    tp, fp = 0, 0
    fpr_list, tpr_list = [0.0], [0.0]
    for lab in sorted_labels:
        if lab == 1.0:
            tp += 1
        else:
            fp += 1
        fpr_list.append(fp / n_neg)
        tpr_list.append(tp / n_pos)
    return np.array(fpr_list), np.array(tpr_list)


def compute_auc(scores: list[float], labels: list[float]) -> float:
    """AUC via Mann-Whitney U. Higher score → label=1 is the positive direction."""
    pos = [s for s, lab in zip(scores, labels) if lab == 1.0]
    neg = [s for s, lab in zip(scores, labels) if lab == 0.0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    return u / (len(pos) * len(neg))


def safe_rho(scores: list[float], labels: list[float]) -> float:
    """Spearman ρ with NaN filtering."""
    pairs = [(s, lab) for s, lab in zip(scores, labels)
             if not np.isnan(s) and not np.isnan(lab)]
    if len(pairs) < 5:
        return float("nan")
    xs, ys = zip(*pairs)
    rho, _ = spearmanr(xs, ys)
    return float(rho)


# ---------------------------------------------------------------------------
# Per-sample metric computation
# ---------------------------------------------------------------------------

def compute_sample_metrics(
    entry: dict,
    label: float,
    tag: str,
    csv_row: dict,
    skip_fit: bool = False,
) -> dict[str, float] | None:
    """Compute all metric variants for one sample from sidecar + CSV data."""
    a_k_bits = entry.get("a_k_curve")
    if a_k_bits is None or len(a_k_bits) < 2:
        return None

    a_k_pb = compute_per_byte_curve(entry)
    if a_k_pb is None or len(a_k_pb) < 2:
        return None

    # C from sidecar metrics
    metrics = entry.get("metrics", {})
    C = metrics.get("coherence_C")
    if C is None:
        e_col = f"metric_icl_C_{tag}"
        C = float(csv_row.get(e_col, "nan"))
    if np.isnan(C):
        return None

    # a_1: unconditional surprise
    uncond_bits = entry.get("unconditional_total_bits", [])
    uncond_pb = entry.get("unconditional_surprises", [])
    a_1_bits = float(np.mean(uncond_bits)) if uncond_bits else a_k_bits[0]
    a_1_pb = float(np.mean(uncond_pb)) if uncond_pb else a_k_pb[0]

    # a_n: last observed point
    a_n_bits = a_k_bits[-1]
    a_n_pb = a_k_pb[-1]

    # a_{n-1}: second-to-last point (avoids k=5 uptick if present)
    a_n1_bits = a_k_bits[-2] if len(a_k_bits) >= 2 else a_n_bits
    a_n1_pb = a_k_pb[-2] if len(a_k_pb) >= 2 else a_n_pb

    # E_discrete
    E_disc_bits = sum(v - a_n_bits for v in a_k_bits)
    E_disc_pb = sum(v - a_n_pb for v in a_k_pb)

    # Exponential fits (optional — slow on large datasets)
    nan = float("nan")
    if skip_fit:
        a_inf_fit_bits = a_inf_fit_pb = E_fit_bits = E_fit_pb = nan
    else:
        k = np.arange(1, len(a_k_bits) + 1, dtype=float)
        fit_bits, ok_bits = fit_exponential(k, np.array(a_k_bits))
        fit_pb, ok_pb = fit_exponential(k, np.array(a_k_pb))
        a_inf_fit_bits = fit_bits["a_inf"] if ok_bits else nan
        a_inf_fit_pb = fit_pb["a_inf"] if ok_pb else nan
        E_fit_bits = fit_bits["E_fit"] if ok_bits else nan
        E_fit_pb = fit_pb["E_fit"] if ok_pb else nan

    return {
        "label": label,
        "C": C,
        # a_∞ variants
        "a_inf_fit_bits": a_inf_fit_bits,
        "a_inf_fit_pb": a_inf_fit_pb,
        "a_n_bits": a_n_bits,
        "a_n_pb": a_n_pb,
        "a_n1_bits": a_n1_bits,
        "a_n1_pb": a_n1_pb,
        # a_1
        "a_1_bits": a_1_bits,
        "a_1_pb": a_1_pb,
        # E variants
        "E_fit_bits": E_fit_bits,
        "E_fit_pb": E_fit_pb,
        "E_disc_bits": E_disc_bits,
        "E_disc_pb": E_disc_pb,
        # Composite: C × a_∞
        "C_a_inf_fit_bits": C * a_inf_fit_bits,
        "C_a_inf_fit_pb": C * a_inf_fit_pb,
        "C_a_n_bits": C * a_n_bits,
        "C_a_n_pb": C * a_n_pb,
        "C_a_n1_bits": C * a_n1_bits,
        "C_a_n1_pb": C * a_n1_pb,
        # Composite: D = C × E
        "D_fit_bits": C * E_fit_bits,
        "D_fit_pb": C * E_fit_pb,
        "D_disc_bits": C * E_disc_bits,
        "D_disc_pb": C * E_disc_pb,
        # Redundancy: C × (a_1 - a_∞)
        "redundancy_bits": C * (a_1_bits - a_inf_fit_bits),
        "redundancy_pb": C * (a_1_pb - a_inf_fit_pb),
        # Raw curves for group-level plots
        "a_k_bits": a_k_bits,
        "a_k_pb": list(a_k_pb),
    }


# ---------------------------------------------------------------------------
# Load all datasets
# ---------------------------------------------------------------------------

def load_all_datasets(
    data_dir: Path, tag: str, skip_fit: bool = False,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, dict[str, str]]]]:
    """Load all datasets, compute metrics.

    Returns:
        (all_results, csv_rows_by_dataset) where:
        - all_results: {dataset_key: [sample_metrics_dict]}
        - csv_rows_by_dataset: {dataset_key: {sidecar_key: csv_row_dict}}
    """
    all_results: dict[str, list[dict]] = {}
    csv_rows_by_dataset: dict[str, dict[str, dict[str, str]]] = {}

    for sidecar_path in sorted(data_dir.rglob(f"*.icl_curves.{tag}.json")):
        csv_stem = sidecar_path.name.replace(f".icl_curves.{tag}.json", "")
        csv_path = sidecar_path.with_name(csv_stem + ".csv")
        if not csv_path.exists():
            continue

        sidecar = load_sidecar(sidecar_path)
        csv_rows = load_csv_rows(csv_path)

        dataset_key = f"{sidecar_path.parent.name}/{csv_stem}"
        csv_rows_by_dataset[dataset_key] = csv_rows
        samples: list[dict] = []

        for sid, entry in sidecar.items():
            row = csv_rows.get(sid)
            if row is None:
                continue
            label = float(row.get("label_value", "nan"))
            if np.isnan(label):
                continue
            m = compute_sample_metrics(entry, label, tag, row, skip_fit=skip_fit)
            if m is not None:
                m["_sidecar_key"] = sid
                samples.append(m)

        all_results[dataset_key] = samples
        print(f"  {dataset_key}: {len(samples)} samples")

    return all_results, csv_rows_by_dataset


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

# Metrics to evaluate, in display order
METRICS_TO_EVAL = [
    # (display_name, dict_key, higher_is_more_diverse)
    ("C×a_inf_fit (pb)", "C_a_inf_fit_pb", True),
    ("C×a_inf_fit (bits)", "C_a_inf_fit_bits", True),
    ("C×a_n (pb)", "C_a_n_pb", True),
    ("C×a_n (bits)", "C_a_n_bits", True),
    ("C×a_{n-1} (pb)", "C_a_n1_pb", True),
    ("C×a_{n-1} (bits)", "C_a_n1_bits", True),
    ("a_n (pb)", "a_n_pb", True),
    ("a_n (bits)", "a_n_bits", True),
    ("a_{n-1} (pb)", "a_n1_pb", True),
    ("a_{n-1} (bits)", "a_n1_bits", True),
    ("a_inf_fit (pb)", "a_inf_fit_pb", True),
    ("a_inf_fit (bits)", "a_inf_fit_bits", True),
    ("a_1 (pb)", "a_1_pb", True),
    ("a_1 (bits)", "a_1_bits", True),
    ("D_fit (pb)", "D_fit_pb", True),
    ("D_fit (bits)", "D_fit_bits", True),
    ("D_disc (pb)", "D_disc_pb", True),
    ("D_disc (bits)", "D_disc_bits", True),
    ("E_fit (pb)", "E_fit_pb", True),
    ("E_fit (bits)", "E_fit_bits", True),
    ("E_disc (pb)", "E_disc_pb", True),
    ("E_disc (bits)", "E_disc_bits", True),
    ("redundancy (pb)", "redundancy_pb", False),  # inverse: low for diverse
]


def is_binary_dataset(samples: list[dict]) -> bool:
    labels = {s["label"] for s in samples}
    return labels == {0.0, 1.0}


def print_summary_table(
    all_results: dict[str, list[dict]],
    dataset_filter: str,
    group_name: str,
) -> list[str]:
    """Print and return summary table lines."""
    lines: list[str] = []
    lines.append(f"\n{'=' * 100}")
    lines.append(f"  {group_name}")
    lines.append(f"{'=' * 100}")

    header = f"  {'Dataset':<40s} {'Metric':<22s} {'Spearman ρ':>12s} {'ROC AUC':>10s} {'OCA':>8s}"
    lines.append(header)
    lines.append(f"  {'-'*40} {'-'*22} {'-'*12} {'-'*10} {'-'*8}")

    for dataset_name in sorted(all_results):
        if dataset_filter.lower() not in dataset_name.lower():
            continue
        samples = all_results[dataset_name]
        if len(samples) < 10:
            continue

        binary = is_binary_dataset(samples)
        task = _pretty_title(dataset_name)

        for i, (display_name, key, higher_diverse) in enumerate(METRICS_TO_EVAL):
            scores = [s[key] for s in samples]
            labels = [s["label"] for s in samples]

            # Filter NaN
            valid = [(sc, lab) for sc, lab in zip(scores, labels)
                     if not np.isnan(sc) and not np.isnan(lab)]
            if len(valid) < 10:
                rho_str = "nan"
                auc_str = "nan"
                oca_str = "nan"
            else:
                v_scores, v_labels = zip(*valid)
                rho = safe_rho(list(v_scores), list(v_labels))
                rho_str = f"{rho:+.3f}" if not np.isnan(rho) else "nan"

                if binary:
                    auc = compute_auc(list(v_scores), list(v_labels))
                    if not higher_diverse:
                        auc = 1.0 - auc if not np.isnan(auc) else auc
                    auc_str = f"{auc:.3f}" if not np.isnan(auc) else "nan"

                    # OCA via Tevet's utils.optimal_classification_accuracy
                    high = [sc for sc, lab in zip(v_scores, v_labels) if lab == 1.0]
                    low = [sc for sc, lab in zip(v_scores, v_labels) if lab == 0.0]
                    oca, _ = optimal_classification_accuracy(high, low)
                    oca_str = f"{oca:.3f}"
                else:
                    auc_str = "—"
                    oca_str = "—"

            ds_col = task if i == 0 else ""
            lines.append(f"  {ds_col:<40s} {display_name:<22s} {rho_str:>12s} {auc_str:>10s} {oca_str:>8s}")

        lines.append("")

    for line in lines:
        print(line)
    return lines


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _pretty_title(dataset_name: str) -> str:
    """Turn 'McDiv_nuggets/mcdiv_nuggets_200_with_hds_prompt_gen' into 'prompt_gen (with_hds)'."""
    stem = dataset_name.split("/")[-1]
    for marker in ("_with_hds_", "_no_hds_"):
        if marker in stem:
            task = stem.split(marker, 1)[1]
            subset = marker.strip("_")
            return f"{task} ({subset})"
    # DecTest / McDiv: try extracting task from end
    for suffix in ("_prompt_gen", "_resp_gen", "_story_gen"):
        if stem.endswith(suffix):
            return stem
    return stem


def plot_roc_curves(
    all_results: dict[str, list[dict]],
    dataset_filter: str,
    group_name: str,
    output_path: Path,
) -> None:
    """ROC curves for binary datasets, one subplot per task."""
    datasets = {k: v for k, v in sorted(all_results.items())
                if dataset_filter in k.lower() and is_binary_dataset(v) and len(v) >= 10}
    if not datasets:
        return

    # Metrics to overlay on ROC
    roc_metrics = [
        ("C×a_n (pb)", "C_a_n_pb", "tab:red", "-"),
        ("a_n (pb)", "a_n_pb", "tab:orange", "--"),
        ("D_fit (pb)", "D_fit_pb", "tab:blue", "-"),
        ("D_disc (pb)", "D_disc_pb", "tab:cyan", "--"),
        ("a_1 (pb)", "a_1_pb", "tab:gray", ":"),
    ]

    n_plots = len(datasets)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"ROC Curves: {group_name}", fontsize=14)

    for ax, (dataset_name, samples) in zip(axes, datasets.items()):
        task = _pretty_title(dataset_name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)

        for display_name, key, color, ls in roc_metrics:
            scores = [s[key] for s in samples]
            labels = [s["label"] for s in samples]
            valid = [(sc, lab) for sc, lab in zip(scores, labels)
                     if not np.isnan(sc)]
            if len(valid) < 10:
                continue
            v_scores, v_labels = zip(*valid)
            fpr, tpr = compute_roc_curve(list(v_scores), list(v_labels))
            auc = compute_auc(list(v_scores), list(v_labels))
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                    label=f"{display_name} (AUC={auc:.3f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(task, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_distributions(
    all_results: dict[str, list[dict]],
    dataset_filter: str,
    group_name: str,
    output_path: Path,
) -> None:
    """Distribution plots for C×a_inf vs D_fit, high vs low diversity."""
    datasets = {k: v for k, v in sorted(all_results.items())
                if dataset_filter in k.lower() and is_binary_dataset(v) and len(v) >= 10}
    if not datasets:
        return

    metrics_to_plot = [
        ("C×a_n (pb)", "C_a_n_pb"),
        ("a_n (pb)", "a_n_pb"),
        ("D_disc (pb)", "D_disc_pb"),
    ]

    n_cols = len(datasets)
    n_rows = len(metrics_to_plot)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(f"Score Distributions: {group_name}", fontsize=14, y=1.01)

    for col, (dataset_name, samples) in enumerate(sorted(datasets.items())):
        task = _pretty_title(dataset_name)
        for row, (display_name, key) in enumerate(metrics_to_plot):
            ax = axes[row, col]
            high = [s[key] for s in samples if s["label"] == 1.0 and not np.isnan(s[key])]
            low = [s[key] for s in samples if s["label"] == 0.0 and not np.isnan(s[key])]

            if high and low:
                all_vals = high + low
                p1, p99 = np.percentile(all_vals, [1, 99])
                hist_range = (p1, p99)
                ax.hist(high, bins=25, alpha=0.5, color="tab:red",
                        label=f"High div (n={len(high)})", range=hist_range)
                ax.hist(low, bins=25, alpha=0.5, color="tab:blue",
                        label=f"Low div (n={len(low)})", range=hist_range)
                ax.axvline(np.mean(high), color="tab:red", ls="--", lw=2)
                ax.axvline(np.mean(low), color="tab:blue", ls="--", lw=2)
                ax.legend(fontsize=7)

            ax.set_xlabel(display_name)
            if row == 0:
                ax.set_title(task, fontweight="bold")
            ax.set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_mean_ak_with_ainf(
    all_results: dict[str, list[dict]],
    dataset_filter: str,
    group_name: str,
    output_path: Path,
) -> None:
    """Mean a_k curves (bits/byte) with a_∞ annotation."""
    datasets = {k: v for k, v in sorted(all_results.items())
                if dataset_filter in k.lower() and "with_hds" in k
                and is_binary_dataset(v) and len(v) >= 10}
    if not datasets:
        return

    n_plots = len(datasets)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"Mean a_k (bits/byte) with a_∞: {group_name}", fontsize=14)

    for ax, (dataset_name, samples) in zip(axes, sorted(datasets.items())):
        task = _pretty_title(dataset_name)

        for target_label, group_label, color in [
            (1.0, "High diversity", "tab:red"),
            (0.0, "Low diversity", "tab:blue"),
        ]:
            group = [s for s in samples if s["label"] == target_label]
            if not group:
                continue

            curves = np.array([s["a_k_pb"] for s in group])
            mean_curve = np.mean(curves, axis=0)
            sem_curve = np.std(curves, axis=0) / np.sqrt(len(group))
            n_k = len(mean_curve)
            k = np.arange(1, n_k + 1, dtype=float)

            ax.plot(k, mean_curve, "o-", color=color, markersize=5, linewidth=2,
                    label=group_label)
            ax.fill_between(k, mean_curve - sem_curve, mean_curve + sem_curve,
                            alpha=0.15, color=color)

            # Exponential fit
            fit_params, fit_ok = fit_exponential(k, mean_curve)
            if fit_ok:
                k_fine = np.linspace(0.5, n_k + 3, 100)
                fitted = exponential_ak(k_fine, fit_params["a_inf"],
                                        fit_params["alpha"], fit_params["beta"])
                ax.plot(k_fine, fitted, "--", color=color, linewidth=1.5, alpha=0.7)

                # a_∞ horizontal line
                a_inf = fit_params["a_inf"]
                ax.axhline(a_inf, color=color, linestyle=":", linewidth=1, alpha=0.5)
                ax.annotate(f"a_∞={a_inf:.3f}", xy=(n_k + 1, a_inf),
                            fontsize=8, color=color, va="bottom")

        ax.set_xticks(range(1, n_k + 1))
        ax.set_title(task, fontweight="bold")
        ax.set_xlabel("k (response index)")
        ax.set_ylabel("a_k (bits/byte)")
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scatter_vs_label(
    all_results: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Scatterplots of diversity label vs metric scores.

    Binary datasets: strip plot (jittered) with box overlay.
    Continuous datasets (DecTest): scatterplot.
    Both annotate Pearson r and Spearman ρ.
    """
    metrics_to_scatter = [
        ("C×a_n (pb)", "C_a_n_pb"),
        ("a_n (pb)", "a_n_pb"),
        ("D_fit (pb)", "D_fit_pb"),
    ]

    # --- Binary datasets ---
    binary_datasets = {k: v for k, v in sorted(all_results.items())
                       if is_binary_dataset(v) and len(v) >= 10
                       and ("with_hds" in k)}
    if binary_datasets:
        n_cols = len(binary_datasets)
        n_rows = len(metrics_to_scatter)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
        if n_cols == 1:
            axes = axes[:, np.newaxis]
        fig.suptitle("Score vs Diversity Label (binary datasets)", fontsize=14, y=1.01)

        for col, (ds_name, samples) in enumerate(binary_datasets.items()):
            title = _pretty_title(ds_name)
            for row, (metric_name, metric_key) in enumerate(metrics_to_scatter):
                ax = axes[row, col]
                high = [s[metric_key] for s in samples
                        if s["label"] == 1.0 and not np.isnan(s[metric_key])]
                low = [s[metric_key] for s in samples
                       if s["label"] == 0.0 and not np.isnan(s[metric_key])]

                if not high or not low:
                    ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                            ha="center")
                    continue

                # Jittered strip plot
                rng = np.random.default_rng(42)
                jitter_low = rng.uniform(-0.15, 0.15, len(low))
                jitter_high = rng.uniform(-0.15, 0.15, len(high))
                ax.scatter(0 + jitter_low, low, alpha=0.3, s=12, color="tab:blue")
                ax.scatter(1 + jitter_high, high, alpha=0.3, s=12, color="tab:red")

                # Box overlay
                bp = ax.boxplot([low, high], positions=[0, 1], widths=0.4,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", linewidth=2))
                bp["boxes"][0].set_facecolor((*plt.cm.tab10(0)[:3], 0.2))
                bp["boxes"][1].set_facecolor((*plt.cm.tab10(3)[:3], 0.2))

                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Low div", "High div"])

                # Annotate correlations
                all_scores = low + high
                all_labels = [0.0] * len(low) + [1.0] * len(high)
                rho = safe_rho(all_scores, all_labels)
                r, _ = pearsonr(all_scores, all_labels)
                ax.text(0.03, 0.97, f"ρ={rho:+.3f}\nr={r:+.3f}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

                ax.set_ylabel(metric_name)
                if row == 0:
                    ax.set_title(title, fontweight="bold")

        plt.tight_layout()
        path = output_dir / "scatter_binary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # --- Continuous datasets (DecTest) ---
    cont_datasets = {k: v for k, v in sorted(all_results.items())
                     if not is_binary_dataset(v) and len(v) >= 10
                     and "dec_test" in k.lower() and "1000" in k}
    if cont_datasets:
        n_cols = len(cont_datasets)
        n_rows = len(metrics_to_scatter)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_cols == 1:
            axes = axes[:, np.newaxis]
        fig.suptitle("Score vs Temperature (DecTest)", fontsize=14, y=1.01)

        for col, (ds_name, samples) in enumerate(cont_datasets.items()):
            title = _pretty_title(ds_name)
            for row, (metric_name, metric_key) in enumerate(metrics_to_scatter):
                ax = axes[row, col]
                pairs = [(s["label"], s[metric_key]) for s in samples
                         if not np.isnan(s[metric_key]) and not np.isnan(s["label"])]
                if len(pairs) < 10:
                    continue
                labels, scores = zip(*pairs)

                ax.scatter(labels, scores, alpha=0.15, s=8, color="tab:blue")

                # Best-fit line
                z = np.polyfit(labels, scores, 1)
                x_fit = np.linspace(min(labels), max(labels), 100)
                ax.plot(x_fit, np.polyval(z, x_fit), "r-", linewidth=2, alpha=0.8)

                rho = safe_rho(list(scores), list(labels))
                r, _ = pearsonr(scores, labels)
                ax.text(0.03, 0.97, f"ρ={rho:+.3f}\nr={r:+.3f}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

                ax.set_ylabel(metric_name)
                ax.set_xlabel("Temperature")
                if row == 0:
                    ax.set_title(title, fontweight="bold")

        plt.tight_layout()
        path = output_dir / "scatter_dectest.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

# Table 5: ConTest ρ + OCA (binary datasets)
# 3 dataset groups, each with 5 metrics × 3 tasks × 2 stats
_TABLE5_GROUPS: list[tuple[str, str, list[str]]] = [
    # (LaTeX group header, key_filter, key_must_contain_all)
    (
        r"\multicolumn{7}{@{}l}{\textit{ConTest (200, with\_hds)}} \\[2pt]",
        "conTest",
        ["with_hds"],
    ),
    (
        r"\multicolumn{7}{@{}l}{\textit{McDiv\_nuggets ($\sim$1K, no\_hds)}} \\[2pt]",
        "McDiv_nuggets",
        ["no_hds"],
    ),
    (
        r"\multicolumn{7}{@{}l}{\textit{McDiv (full, no\_hds, $\sim$2K)}} \\[2pt]",
        "McDiv/",
        ["no_hds"],
    ),
]

_TABLE5_METRICS: list[tuple[str, str | None, bool]] = [
    # (LaTeX name, our_metric_key or None for baseline, higher_is_more_diverse)
    # None means read from CSV column instead
    (r"$C \!\times\! a_n$ (ours)", "C_a_n_pb", True),
    (r"$a_n$ (ours)", "a_n_pb", True),
    ("SentBERT", None, False),
    ("BERTsts", None, False),
    (r"distinct-$n$", None, True),
]

_BASELINE_CSV_COLS: dict[str, str] = {
    "SentBERT": "metric_sent_bert",
    "BERTsts": "metric_bert_sts",
    r"distinct-$n$": "metric_averaged_distinct_ngrams",
    "BERTScore": "metric_bert_score",
    "cos-sim": "metric_averaged_cosine_similarity",
}

# Table 6: DecTest ρ (continuous temperature labels)
_TABLE6_METRICS: list[tuple[str, str | None]] = [
    (r"$C \!\times\! a_n$ (ours)", "C_a_n_pb"),
    (r"$a_n$ (ours)", "a_n_pb"),
    (r"distinct-$n$", None),
    ("BERTScore", None),
    ("SentBERT", None),
    ("cos-sim", None),
]

TASKS = ["prompt_gen", "resp_gen", "story_gen"]


def _find_dataset_key(
    all_keys: list[str], dir_filter: str, must_contain: list[str], task: str,
) -> str | None:
    """Find the dataset key matching a filter + task."""
    for key in all_keys:
        if dir_filter not in key:
            continue
        if not all(mc in key for mc in must_contain):
            continue
        if key.endswith(f"_{task}"):
            return key
    return None


def _get_baseline_scores(
    csv_rows: dict[str, dict[str, str]],
    samples: list[dict],
    csv_col: str,
) -> tuple[list[float], list[float]]:
    """Extract baseline metric scores from CSV rows, aligned with sample labels."""
    scores: list[float] = []
    labels: list[float] = []
    for s in samples:
        skey = s["_sidecar_key"]
        row = csv_rows.get(skey)
        if row is None:
            raise ValueError(f"Missing CSV row for sidecar_key={skey!r}")
        val_str = row.get(csv_col, "")
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            continue
        if np.isnan(val):
            continue
        scores.append(val)
        labels.append(s["label"])
    return scores, labels


def _compute_rho_oca(
    scores: list[float], labels: list[float], is_binary: bool,
) -> tuple[float, float]:
    """Compute Spearman ρ and OCA (OCA is NaN for non-binary)."""
    rho = safe_rho(scores, labels)
    if is_binary:
        high = [sc for sc, lab in zip(scores, labels) if lab == 1.0]
        low = [sc for sc, lab in zip(scores, labels) if lab == 0.0]
        if len(high) < 2 or len(low) < 2:
            return rho, float("nan")
        oca, _ = optimal_classification_accuracy(high, low)
        return rho, oca
    return rho, float("nan")


def _fmt_rho(rho: float) -> str:
    """Format ρ for LaTeX: +0.584 or $-0.003$."""
    if np.isnan(rho):
        return "---"
    if rho < 0:
        return f"$-{abs(rho):.3f}$"
    return f"+{rho:.3f}"


def _fmt_oca(oca: float) -> str:
    if np.isnan(oca):
        return "---"
    return f"{oca:.3f}"


def write_latex_tables(
    all_results: dict[str, list[dict]],
    csv_rows_by_dataset: dict[str, dict[str, dict[str, str]]],
    tables_dir: Path,
    run_tag: str,
) -> None:
    """Generate LaTeX tabular-body .tex files for paper tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    all_keys = list(all_results.keys())

    # --- Table 5: ConTest ρ + OCA ---
    lines: list[str] = []
    lines.append(f"% Generated by: scripts/analyze_c_ainf.py --run-tag {run_tag}")
    lines.append("% Baseline source: diversity-eval/data/with_metrics/")
    lines.append(r"\begin{tabular}{@{}l rr rr rr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{2}{c}{prompt\_gen} & \multicolumn{2}{c}{resp\_gen} "
        r"& \multicolumn{2}{c}{story\_gen} \\"
    )
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"Metric & $\rho$ & OCA & $\rho$ & OCA & $\rho$ & OCA \\")
    lines.append(r"\midrule")

    for group_idx, (group_header, dir_filter, must_contain) in enumerate(_TABLE5_GROUPS):
        if group_idx > 0:
            lines.append(r"\midrule")
        lines.append(group_header)

        # Collect all rho/oca values per task for bolding
        group_rhos: dict[str, list[tuple[int, float]]] = {t: [] for t in TASKS}
        group_ocas: dict[str, list[tuple[int, float]]] = {t: [] for t in TASKS}
        row_data: list[list[tuple[float, float]]] = []  # metric_idx -> [(rho, oca) per task]

        for metric_idx, (latex_name, our_key, _higher) in enumerate(_TABLE5_METRICS):
            task_vals: list[tuple[float, float]] = []
            for task in TASKS:
                ds_key = _find_dataset_key(all_keys, dir_filter, must_contain, task)
                if ds_key is None:
                    task_vals.append((float("nan"), float("nan")))
                    continue

                samples = all_results[ds_key]
                if our_key is not None:
                    scores = [s[our_key] for s in samples if not np.isnan(s[our_key])]
                    labels = [s["label"] for s in samples if not np.isnan(s[our_key])]
                else:
                    csv_col = _BASELINE_CSV_COLS[latex_name]
                    csv_rows = csv_rows_by_dataset[ds_key]
                    scores, labels = _get_baseline_scores(csv_rows, samples, csv_col)

                rho, oca = _compute_rho_oca(scores, labels, is_binary=True)
                task_vals.append((rho, oca))
                if not np.isnan(rho):
                    group_rhos[task].append((metric_idx, abs(rho)))
                if not np.isnan(oca):
                    group_ocas[task].append((metric_idx, oca))

            row_data.append(task_vals)

        # Find best (highest |ρ| and highest OCA) per task
        best_rho: dict[str, int] = {}
        best_oca: dict[str, int] = {}
        for task in TASKS:
            if group_rhos[task]:
                best_rho[task] = max(group_rhos[task], key=lambda x: x[1])[0]
            if group_ocas[task]:
                best_oca[task] = max(group_ocas[task], key=lambda x: x[1])[0]

        # Emit rows
        for metric_idx, (latex_name, _our_key, _higher) in enumerate(_TABLE5_METRICS):
            cells: list[str] = [latex_name]
            for task_idx, task in enumerate(TASKS):
                rho, oca = row_data[metric_idx][task_idx]
                rho_s = _fmt_rho(rho)
                oca_s = _fmt_oca(oca)
                if best_rho.get(task) == metric_idx:
                    rho_s = r"\textbf{" + rho_s + "}"
                if best_oca.get(task) == metric_idx:
                    oca_s = r"\textbf{" + oca_s + "}"
                cells.append(rho_s)
                cells.append(oca_s)
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table5_path = tables_dir / "contest_rho_oca.tex"
    table5_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {table5_path}")

    # --- Table 6: DecTest ρ ---
    lines = []
    lines.append(f"% Generated by: scripts/analyze_c_ainf.py --run-tag {run_tag}")
    lines.append("% Baseline source: diversity-eval/data/with_metrics/")
    lines.append(r"\begin{tabular}{@{}l rrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Metric & prompt\_gen & resp\_gen & story\_gen \\")
    lines.append(r"\midrule")

    # Collect values for bolding
    task_rhos: dict[str, list[tuple[int, float]]] = {t: [] for t in TASKS}
    row_data_dec: list[list[float]] = []

    for metric_idx, (latex_name, our_key) in enumerate(_TABLE6_METRICS):
        task_vals: list[float] = []
        for task in TASKS:
            ds_key = _find_dataset_key(all_keys, "decTest", ["1000_no_hds"], task)
            if ds_key is None:
                task_vals.append(float("nan"))
                continue

            samples = all_results[ds_key]
            if our_key is not None:
                scores = [s[our_key] for s in samples if not np.isnan(s[our_key])]
                labels = [s["label"] for s in samples if not np.isnan(s[our_key])]
            else:
                csv_col = _BASELINE_CSV_COLS[latex_name]
                csv_rows = csv_rows_by_dataset[ds_key]
                scores, labels = _get_baseline_scores(csv_rows, samples, csv_col)

            rho = safe_rho(scores, labels)
            task_vals.append(rho)
            if not np.isnan(rho):
                task_rhos[task].append((metric_idx, abs(rho)))

        row_data_dec.append(task_vals)

    # Find best per task
    best_rho_dec: dict[str, int] = {}
    for task in TASKS:
        if task_rhos[task]:
            best_rho_dec[task] = max(task_rhos[task], key=lambda x: x[1])[0]

    # Emit rows
    for metric_idx, (latex_name, _our_key) in enumerate(_TABLE6_METRICS):
        cells: list[str] = [latex_name]
        for task_idx, task in enumerate(TASKS):
            rho = row_data_dec[metric_idx][task_idx]
            rho_s = _fmt_rho(rho)
            if best_rho_dec.get(task) == metric_idx:
                rho_s = r"\textbf{" + rho_s + "}"
            cells.append(rho_s)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table6_path = tables_dir / "dectest_rho.tex"
    table6_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {table6_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test C × a_∞ as diversity score")
    parser.add_argument("--run-tag", type=str, default="qwen25_completion_v2")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-fit", action="store_true",
                        help="Skip exponential curve fitting (faster, fit-based metrics will be NaN)")
    args = parser.parse_args()

    data_dir = RESULTS_BASE / args.run_tag
    if not data_dir.exists():
        print(f"No data found at {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "figures" / "tevet_validation" / "c_ainf_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets and computing metrics...")
    all_results, csv_rows_by_dataset = load_all_datasets(
        data_dir, args.run_tag, skip_fit=args.skip_fit,
    )

    # Summary tables
    all_lines: list[str] = []
    for filt, name in [
        ("mcdiv_nuggets", "McDiv_nuggets (content diversity — PRIMARY)"),
        ("con_test", "ConTest (binary content diversity)"),
        ("dec_test", "DecTest (temperature correlation)"),
        ("mcdiv/", "McDiv (full, continuous labels)"),
    ]:
        lines = print_summary_table(all_results, filt, name)
        all_lines.extend(lines)

    # Save table
    table_path = output_dir / "summary_table.txt"
    with open(table_path, "w") as f:
        f.write("\n".join(all_lines))
    print(f"\nSaved: {table_path}")

    # ROC curves
    for filt, name in [
        ("mcdiv_nuggets", "McDiv_nuggets"),
        ("con_test", "ConTest"),
    ]:
        plot_roc_curves(all_results, filt, name,
                        output_dir / f"roc_curves_{filt}.png")

    # Distribution plots
    for filt, name in [
        ("mcdiv_nuggets", "McDiv_nuggets"),
        ("con_test", "ConTest"),
    ]:
        plot_distributions(all_results, filt, name,
                           output_dir / f"distributions_{filt}.png")

    # Mean a_k with a_∞ annotation
    plot_mean_ak_with_ainf(all_results, "mcdiv_nuggets", "McDiv_nuggets",
                           output_dir / "mean_ak_with_ainf.png")

    # Scatterplots: label vs score
    plot_scatter_vs_label(all_results, output_dir)

    # LaTeX tables for paper
    tables_dir = PROJECT_ROOT / "results" / "tables"
    write_latex_tables(all_results, csv_rows_by_dataset, tables_dir, args.run_tag)

    print(f"\nAll outputs in {output_dir}")
    print(f"LaTeX tables in {tables_dir}")


if __name__ == "__main__":
    main()
