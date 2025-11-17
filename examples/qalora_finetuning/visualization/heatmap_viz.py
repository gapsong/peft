#!/usr/bin/env python3
"""Visualize evaluation results from CSV"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np
from matplotlib.patches import Rectangle


def add_axis_break(ax, break_start, break_end, d=0.015):
    """Add break marks (diagonal lines) to indicate axis discontinuity"""
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.5)
    
    # Vertical break lines for y-axis
    if break_start and break_end:
        # Calculate relative position on y-axis
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        break_pos = (break_start - ylim[0]) / y_range
        
        # Draw diagonal lines
        ax.plot((-d, +d), (break_pos - d, break_pos + d), **kwargs)
        ax.plot((-d, +d), (break_pos + d, break_pos - d), **kwargs)


def plot_perplexity_by_rank(df, out):
    df_steps = df[df['step'] != 'final'].copy()
    df_steps['step'] = df_steps['step'].astype(int)
    ranks = sorted(df_steps['rank'].dropna().unique())
    
    fig, axes = plt.subplots(1, len(ranks), figsize=(6*len(ranks), 5), squeeze=False)
    
    for idx, rank in enumerate(ranks):
        ax = axes[0][idx]
        for mode in df_steps[df_steps['rank'] == rank]['training_mode'].dropna().unique():
            data = df_steps[(df_steps['rank'] == rank) & (df_steps['training_mode'] == mode)]
            grouped = data.groupby('step')['wikitext_word_perplexity'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=mode, linewidth=2)
        
        # Check if we need axis break
        all_values = df_steps[df_steps['rank'] == rank]['wikitext_word_perplexity'].dropna()
        if len(all_values) > 0:
            q75 = all_values.quantile(0.75)
            max_val = all_values.max()
            if max_val > 3 * q75:  # Large outlier detected
                # Set ylim to exclude outlier, show break
                ax.set_ylim(0, q75 * 1.5)
                add_axis_break(ax, q75 * 1.3, q75 * 1.4)
        
        ax.set_title(f'Rank {int(rank)}', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Perplexity')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out / 'perplexity_by_rank.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_best_heatmap(df, out):
    """Heatmap showing BEST performance (minimum perplexity) across all steps"""
    # Get best (minimum) perplexity for each experiment
    df_best = df.loc[df.groupby('experiment')['wikitext_word_perplexity'].idxmin()]
    
    pivot = df_best.pivot_table(values='wikitext_word_perplexity', index='training_mode', columns='rank', aggfunc='mean')
    
    # Use log scale if there's a large outlier
    vmin, vmax = pivot.min().min(), pivot.max().max()
    use_log = (vmax / vmin) > 5
    
    if use_log:
        pivot_log = np.log10(pivot)
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(pivot_log, annot=pivot, fmt='.1f', cmap='RdYlGn_r', 
                        cbar_kws={'label': 'Perplexity (log scale)'})
    else:
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                        cbar_kws={'label': 'Perplexity'})
    
    plt.title('Best WikiText Perplexity: Method × Rank', fontweight='bold')
    plt.xlabel('Rank')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(out / 'best_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rank_scaling(df, out):
    """Use best perplexity for rank scaling"""
    df_best = df.loc[df.groupby('experiment')['wikitext_word_perplexity'].idxmin()]
    methods = df_best['training_mode'].dropna().unique()
    
    fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5), squeeze=False)
    
    for idx, mode in enumerate(methods):
        ax = axes[0][idx]
        grouped = df_best[df_best['training_mode'] == mode].groupby('rank')['wikitext_word_perplexity'].mean().sort_index()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=8)
        
        # Check for outliers
        q75 = grouped.quantile(0.75)
        max_val = grouped.max()
        if max_val > 3 * q75:
            ax.set_ylim(0, q75 * 1.5)
            add_axis_break(ax, q75 * 1.3, q75 * 1.4)
        
        ax.set_title(mode, fontweight='bold')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Best Perplexity')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out / 'rank_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results.csv')
    parser.add_argument('--output_dir', default='./plots')
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)
    
    plot_perplexity_by_rank(df, out)
    plot_best_heatmap(df, out)
    plot_rank_scaling(df, out)
    
    df_best = df.loc[df.groupby('experiment')['wikitext_word_perplexity'].idxmin()]
    cols = ['experiment', 'rank', 'training_mode', 'step', 'wikitext_word_perplexity']
    df_best[cols].to_csv(out / 'best_results.csv', index=False)
    
    print(f"✅ Saved to {out}/")


if __name__ == "__main__":
    main()