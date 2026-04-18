#!/usr/bin/env python3
"""
Trie Path Probability Convergence — Analysis and Plots
Usage: python3 rnd/convergence_analysis.py [--depths 4,8,16,32] [--output-dir rnd]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(path):
    df = pd.read_csv(path)
    # log_pi_min = 0.0 for the root-adjacent case means "no min set yet"
    # treat 0.0 as a sentinel meaning "take log_pi_mean as the floor"
    return df

def stratum_bins(log_pi_col):
    """Return bin edges covering the data range in integer log10 steps."""
    lo = np.floor(log_pi_col.min())
    hi = np.ceil(log_pi_col.max())
    return np.arange(lo, hi + 1, 1.0)

def coverage_at_threshold(df, threshold, log_pi_mean_col="log_pi_mean"):
    """
    Fraction of probability mass covered by paths with log_abs_err < threshold.
    We use exp(log_pi_mean) as the weight for each path.
    """
    pi = np.exp(df["log_pi_mean"].values)
    mask = df["log_abs_err"].values < threshold
    total = pi.sum()
    if total == 0:
        return 0.0
    return pi[mask].sum() / total

def make_convergence_plot(df, depth, output_dir):
    """Median and 90th-pctile log_abs_err vs log10(pi_mean) stratum bins."""
    df = df[df["depth"] == depth].copy()
    if df.empty:
        print(f"  No data for depth={depth}, skipping plot")
        return

    df["log10_pi_mean"] = df["log_pi_mean"] / np.log(10)  # convert nat to log10

    bins = stratum_bins(df["log10_pi_mean"])
    if len(bins) < 2:
        print(f"  Insufficient range for depth={depth}, skipping plot")
        return

    bin_centers = []
    medians = []
    p90s = []
    counts = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (df["log10_pi_mean"] >= lo) & (df["log10_pi_mean"] < hi)
        sub = df[mask]["log_abs_err"]
        if len(sub) < 3:
            continue
        bin_centers.append((lo + hi) / 2)
        medians.append(sub.median())
        p90s.append(sub.quantile(0.90))
        counts.append(len(sub))

    if not bin_centers:
        print(f"  No bins with enough data for depth={depth}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, medians, "o-", label="Median log|err|", color="steelblue")
    ax.plot(bin_centers, p90s, "s--", label="90th pctile log|err|", color="tomato")

    # annotate counts
    for x, m, c in zip(bin_centers, medians, counts):
        ax.annotate(f"n={c}", (x, m), textcoords="offset points", xytext=(0, 6),
                    fontsize=7, ha="center", color="steelblue")

    ax.set_xlabel("log₁₀(π_mean) stratum")
    ax.set_ylabel("log |err| = |log π_A − log π_B|")
    ax.set_title(f"Path probability convergence — depth ≤ {depth}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"convergence_plot_D{depth}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def make_depth_scaling_plot(df, target_depth, output_dir):
    """log_abs_err vs depth for several log10(pi_mean) strata."""
    df = df.copy()
    df["log10_pi_mean"] = df["log_pi_mean"] / np.log(10)

    # Pick a few representative strata
    lo = np.floor(df["log10_pi_mean"].min())
    hi = np.ceil(df["log10_pi_mean"].max())
    strata_edges = np.arange(lo, hi + 1, 2.0)  # wider bins for this plot

    if len(strata_edges) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0

    for i in range(len(strata_edges) - 1):
        slo, shi = strata_edges[i], strata_edges[i + 1]
        mask = (df["log10_pi_mean"] >= slo) & (df["log10_pi_mean"] < shi)
        sub = df[mask]
        if sub.empty:
            continue

        by_d = sub.groupby("depth")["log_abs_err"].median()
        if len(by_d) < 2:
            continue

        label = f"log₁₀π ∈ [{slo:.0f},{shi:.0f})"
        ax.plot(by_d.index, by_d.values, "o-", label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Depth")
    ax.set_ylabel("Median log |err|")
    ax.set_title(f"Convergence vs depth by stratum — max depth {target_depth}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"depth_scaling_D{target_depth}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def write_coverage(df, depth, output_dir):
    """Write coverage stats at multiple rel_err thresholds."""
    dfd = df[df["depth"] == depth]
    if dfd.empty:
        return

    thresholds = [0.1, 0.05, 0.01]
    lines = [f"Coverage report for depth={depth}", "=" * 40]
    lines.append(f"Total paths at this depth: {len(dfd)}")
    lines.append(f"Median log_abs_err: {dfd['log_abs_err'].median():.4f}")
    lines.append(f"90th pctile log_abs_err: {dfd['log_abs_err'].quantile(0.9):.4f}")
    lines.append("")
    lines.append("Probability mass coverage (by exp(log_pi_mean) weight):")
    for thr in thresholds:
        cov = coverage_at_threshold(dfd, thr)
        lines.append(f"  log_abs_err < {thr:.2f}: {cov*100:.1f}%")

    out_path = os.path.join(output_dir, f"coverage_D{depth}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")

def make_fingerprint_stability_plot(dfs, output_dir):
    """
    Two-panel figure across all depths:
      Left:  Spearman ρ and Kendall τ (A vs B ranking agreement)
      Right: Signal-to-noise ratio (between-path σ / within-path median noise)
    """
    from scipy import stats as scipy_stats

    depths_sorted = sorted(dfs.keys())
    rhos, taus, snrs, ns = [], [], [], []

    for d in depths_sorted:
        df = dfs[d]
        dfd = df[df["depth"] == d].copy()
        if len(dfd) < 10:
            continue
        rho, _ = scipy_stats.spearmanr(dfd["log_pi_a"], dfd["log_pi_b"])
        tau, _ = scipy_stats.kendalltau(dfd["log_pi_a"], dfd["log_pi_b"])
        spread = dfd["log_pi_mean"].std()
        noise  = dfd["log_abs_err"].median()
        snr    = spread / noise if noise > 0 else float("nan")
        rhos.append(rho)
        taus.append(tau)
        snrs.append(snr)
        ns.append(len(dfd))

    if not rhos:
        return

    plot_depths = depths_sorted[:len(rhos)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Path Probability Fingerprint Stability across Depth", fontsize=13)

    # --- Left: rank correlation ---
    ax1.plot(plot_depths, rhos, "o-", color="steelblue", label="Spearman ρ")
    ax1.plot(plot_depths, taus, "s--", color="tomato",   label="Kendall τ")
    for x, r, t, n in zip(plot_depths, rhos, taus, ns):
        ax1.annotate(f"n={n}", (x, r), textcoords="offset points",
                     xytext=(0, 7), fontsize=7, ha="center", color="steelblue")
    ax1.axhline(0.8, color="gray", linestyle=":", linewidth=1, label="ρ = 0.80")
    ax1.set_xlabel("Max path depth")
    ax1.set_ylabel("Rank correlation (A vs B)")
    ax1.set_title("Ranking agreement between corpus halves")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Right: signal-to-noise ---
    color_snr = ["forestgreen" if s > 1 else "tomato" for s in snrs]
    bars = ax2.bar(plot_depths, snrs, color=color_snr, alpha=0.75, width=0.8)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="S/N = 1 (noise floor)")
    for bar, s in zip(bars, snrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{s:.1f}×", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Max path depth")
    ax2.set_ylabel("Signal / Noise  (between-path σ / within-path median |Δlog π|)")
    ax2.set_title("Fingerprint signal-to-noise ratio")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    out_path = os.path.join(output_dir, "fingerprint_stability.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_ranking_scatter(dfs, output_dir):
    """
    Scatter of log π_A vs log π_B for depths 4, 8, 16 side by side.
    Shows how tightly the two halves agree on path ordering.
    """
    target_depths = [d for d in [4, 8, 16] if d in dfs]
    if not target_depths:
        return

    fig, axes = plt.subplots(1, len(target_depths), figsize=(5 * len(target_depths), 5))
    if len(target_depths) == 1:
        axes = [axes]

    for ax, d in zip(axes, target_depths):
        df = dfs[d]
        dfd = df[df["depth"] == d].copy()
        if dfd.empty:
            continue

        # Subsample for readability if large
        sample = dfd.sample(min(3000, len(dfd)), random_state=42)

        sc = ax.scatter(sample["log_pi_a"], sample["log_pi_b"],
                        c=sample["count_a"], cmap="viridis_r",
                        s=8, alpha=0.5, norm=plt.matplotlib.colors.LogNorm())

        # Perfect agreement line
        lo = min(sample["log_pi_a"].min(), sample["log_pi_b"].min())
        hi = max(sample["log_pi_a"].max(), sample["log_pi_b"].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="perfect agreement")

        plt.colorbar(sc, ax=ax, label="count_a")
        ax.set_xlabel("log π_A")
        ax.set_ylabel("log π_B")
        ax.set_title(f"depth = {d}  (n={len(dfd)})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Path probability products: corpus A vs corpus B", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "ranking_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_n_eff_check(dfs, depths, corpus_size, output_dir):
    """
    For each stratum and depth, estimate N_eff = 2d / (rel_err^2 * pi_min).
    rel_err approximated as exp(log_abs_err) - 1 ≈ log_abs_err for small err.
    pi_min = exp(log_pi_min).
    """
    lines = ["N_eff check: estimated vs actual corpus size", "=" * 50,
             f"Actual corpus size: ~{corpus_size} tokens (each half ~{corpus_size//2})",
             ""]

    for depth in depths:
        all_dfs = [df[df["depth"] == depth] for df in dfs.values() if depth in df["depth"].values]
        if not all_dfs:
            continue
        combined = pd.concat(all_dfs)
        if combined.empty:
            continue

        # Use log_abs_err as approximation for rel_err in log space
        # N_eff = 2d / (log_abs_err^2 * |pi_min|) where pi_min = exp(log_pi_min)
        # Filter out zero/trivial errors and zero pi_min
        sub = combined[(combined["log_abs_err"] > 1e-9) & (combined["log_pi_min"] < 0)]
        if sub.empty:
            continue

        rel_err_sq = sub["log_abs_err"].values ** 2
        pi_min = np.exp(sub["log_pi_min"].values)
        # Clip to avoid divide by zero
        pi_min = np.clip(pi_min, 1e-12, 1.0)
        n_eff = (2 * depth) / (rel_err_sq * pi_min)

        lines.append(f"depth={depth}:")
        lines.append(f"  median N_eff = {np.median(n_eff):.0f}")
        lines.append(f"  10th pctile N_eff = {np.percentile(n_eff, 10):.0f}")
        lines.append(f"  90th pctile N_eff = {np.percentile(n_eff, 90):.0f}")
        lines.append(f"  paths used: {len(sub)}")
        lines.append("")

    out_path = os.path.join(output_dir, "n_eff_check.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convergence analysis for trie path experiment")
    parser.add_argument("--depths", default="4,8,16,32",
                        help="Comma-separated list of depths to analyze (default: 4,8,16,32)")
    parser.add_argument("--output-dir", default="rnd",
                        help="Directory with CSV inputs and for output files (default: rnd)")
    parser.add_argument("--corpus-size", type=int, default=1115394,
                        help="Approximate corpus size in tokens (default: 1115394)")
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    output_dir = args.output_dir

    print(f"Convergence analysis: depths={depths}, output_dir={output_dir}")

    dfs = {}
    for depth in depths:
        csv_path = os.path.join(output_dir, f"results_D{depth}.csv")
        if not os.path.exists(csv_path):
            print(f"  Missing: {csv_path}, skipping depth {depth}")
            continue
        df = load_csv(csv_path)
        print(f"  Loaded depth={depth}: {len(df)} rows")
        dfs[depth] = df

    if not dfs:
        print("No CSV files found. Run the Crystal program first.")
        sys.exit(1)

    # Combine all depths for cross-depth plots
    combined = pd.concat(dfs.values(), ignore_index=True)

    print("\nGenerating plots...")
    for depth in dfs:
        make_convergence_plot(dfs[depth], depth, output_dir)
        make_depth_scaling_plot(combined, depth, output_dir)
        write_coverage(dfs[depth], depth, output_dir)

    write_n_eff_check(dfs, depths, args.corpus_size, output_dir)
    make_fingerprint_stability_plot(dfs, output_dir)
    make_ranking_scatter(dfs, output_dir)

    # Print summary table
    print("\nSummary table:")
    print(f"{'depth':>8} {'n_paths':>10} {'median_err':>12} {'p90_err':>12}")
    for depth in sorted(dfs.keys()):
        df = dfs[depth]
        dfd = df[df["depth"] == depth]
        if dfd.empty:
            continue
        n = len(dfd)
        med = dfd["log_abs_err"].median()
        p90 = dfd["log_abs_err"].quantile(0.9)
        print(f"{depth:>8} {n:>10} {med:>12.4f} {p90:>12.4f}")

    print("\nFirst 10 rows of combined results:")
    show_cols = ["depth", "log_pi_a", "log_pi_b", "log_pi_mean", "log_abs_err", "count_a", "count_b", "log_pi_min"]
    show_cols = [c for c in show_cols if c in combined.columns]
    print(combined[show_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
