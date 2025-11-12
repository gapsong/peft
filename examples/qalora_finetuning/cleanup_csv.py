#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Fixed parameters
# =========================
CSV_PATH = "/home/nudel/Documents/peft/mein_eigenes_leaderboard_2bit_und_3bit.csv"
OUT_BASENAME = "/home/nudel/Documents/peft/leaderboard_clean"  # without extension

RANK_COLS_DEFAULT = ["r16", "r128", "r256"]
# Desired method order
METHOD_ORDER = ["INITIAL QUANTIZATION", "GPTQ LORA", "POST QUANTIZATION", "QALORA"]

sns.set_theme(style="whitegrid")

def load_and_melt(csv_path: str, rank_cols=None) -> pd.DataFrame:
    if rank_cols is None:
        rank_cols = RANK_COLS_DEFAULT
    df = pd.read_csv(csv_path)
    needed = ["bits", "method"] + [c for c in rank_cols if c in df.columns]
    df = df[needed].copy()

    # Wide -> long
    long = df.melt(id_vars=["bits", "method"], var_name="rank_col", value_name="win_rate")
    long["rank"] = long["rank_col"].str.replace("r", "", regex=False).astype("Int64")
    long["win_rate"] = pd.to_numeric(long["win_rate"], errors="coerce")
    long["bits"] = pd.to_numeric(long["bits"], errors="coerce").astype("Int64")
    long["rank"] = long["rank"].astype("Int64")
    long = long.sort_values(["bits", "method", "rank"])
    return long

def plot_heatmap(long: pd.DataFrame, out_basename: str):
    rank_order = sorted(long["rank"].dropna().unique())
    present_methods = long["method"].dropna().unique().tolist()
    method_order = [m for m in METHOD_ORDER if m in present_methods] + \
                   sorted([m for m in present_methods if m not in METHOD_ORDER])
    bits_list = sorted(long["bits"].dropna().unique())

    ncols = len(bits_list)
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 6.6), dpi=140, squeeze=False)
    axes = axes[0]

    cmap = sns.color_palette("light:teal", as_cmap=True)
    vmin, vmax = 0, 100

    for i, bits in enumerate(bits_list):
        ax = axes[i]
        sub = long[long["bits"] == bits].copy()
        pivot = sub.pivot_table(index="method", columns="rank", values="win_rate", aggfunc="mean")
        pivot = pivot.reindex(index=method_order, columns=rank_order)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=False,
            fmt="",
            linewidths=0.6,
            linecolor="white",
            cbar=(i == ncols - 1),
            square=True,
        )

        # Cell annotations (values)
        for r_i, method in enumerate(pivot.index):
            for c_i, rank in enumerate(pivot.columns):
                val = pivot.loc[method, rank]
                text = "—" if pd.isna(val) else f"{val:.1f}"
                ax.text(c_i + 0.5, r_i + 0.5, text, ha="center", va="center", fontsize=11, color="black")

        ax.set_title(f"{bits}-bit", fontsize=14, weight="bold", pad=10)
        ax.set_xlabel("LoRA Rank", fontsize=12)
        ax.set_ylabel("Methode", fontsize=12)
        ax.set_xticklabels([str(x) for x in rank_order], rotation=0)
        ax.set_yticklabels([str(x) for x in pivot.index], rotation=0)

    fig.suptitle("Win-Rate (%): Methoden × Ranks, facettiert nach Bits", fontsize=16, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])
    png = f"{out_basename}_heatmap.png"
    pdf = f"{out_basename}_heatmap.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}\nSaved: {pdf}")

def plot_linepoints(long: pd.DataFrame, out_basename: str):
    rank_order = sorted(long["rank"].dropna().unique())
    bits_list = sorted(long["bits"].dropna().unique())

    present_methods = long["method"].dropna().unique().tolist()
    methods = [m for m in METHOD_ORDER if m in present_methods] + \
              sorted([m for m in present_methods if m not in METHOD_ORDER])

    marker_cycle = ["o", "s", "D", "X", "^"]
    method_markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(methods)}
    palette = sns.color_palette("tab10", n_colors=len(methods))
    color_map = {m: palette[i] for i, m in enumerate(methods)}

    ncols = len(bits_list)
    fig, axes = plt.subplots(1, ncols, figsize=(6.8 * ncols, 5.8), dpi=140, squeeze=False)
    axes = axes[0]

    for i, bits in enumerate(bits_list):
        ax = axes[i]
        sub = long[long["bits"] == bits].copy()
        lines_by_method = {}

        for method in methods:
            s = sub[sub["method"] == method].sort_values("rank")
            line, = ax.plot(
                s["rank"],
                s["win_rate"],
                label=method,
                marker=method_markers[method],
                linewidth=2.0,
                markersize=7,
                color=color_map[method],
            )
            lines_by_method[method] = line

        ax.set_title(f"{bits}-bit", fontsize=14, weight="bold")
        ax.set_xlabel("LoRA Rank", fontsize=12)
        ax.set_ylabel("Win-Rate (%)", fontsize=12)
        ax.set_xticks(rank_order)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.3)

        if i == ncols - 1:
            ordered_handles = [lines_by_method[m] for m in methods if m in lines_by_method]
            ax.legend(
                handles=ordered_handles,
                labels=[h.get_label() for h in ordered_handles],
                title="Methode",
                loc="best",
                frameon=True,
            )

    fig.suptitle("Win-Rate: Linien/Marker je Methode, facettiert nach Bits", fontsize=16, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])
    png = f"{out_basename}_lines.png"
    pdf = f"{out_basename}_lines.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}\nSaved: {pdf}")

def main():
    os.makedirs(os.path.dirname(OUT_BASENAME) or ".", exist_ok=True)
    long = load_and_melt(CSV_PATH)
    plot_heatmap(long, OUT_BASENAME)
    plot_linepoints(long, OUT_BASENAME)

if __name__ == "__main__":
    main()