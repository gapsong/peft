#!/usr/bin/env python3
"""Visualize evaluation results from CSV"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_wikitext_over_steps(df, output_dir):
    """Plot WikiText perplexity over training steps"""
    df_steps = df[df['step'] != 'final'].copy()
    df_steps['step'] = df_steps['step'].astype(int)
    
    for mode in df_steps['training_mode'].unique():
        df_mode = df_steps[df_steps['training_mode'] == mode]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for rank in sorted(df_mode['rank'].dropna().unique()):
            grouped = df_mode[df_mode['rank'] == rank].groupby('step')['wikitext_word_perplexity'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=f'r={rank}')
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('WikiText Perplexity')
        ax.set_title(f'{mode}')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'wikitext_{mode}.png', dpi=300)
        plt.close()


def save_final_table(df, output_dir):
    """Save final performance as CSV"""
    df_final = df[df['step'] == 'final'] if 'final' in df['step'].values else df.groupby('experiment').tail(1)
    cols = ['experiment', 'rank', 'training_mode', 'wikitext_word_perplexity', 'wikitext_bits_per_byte']
    cols += [c for c in df_final.columns if 'acc' in c and not c.endswith('_stderr')][:3]
    df_final[cols].to_csv(Path(output_dir) / 'final_results.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results.csv')
    parser.add_argument('--output_dir', default='./plots')
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    Path(args.output_dir).mkdir(exist_ok=True)
    
    plot_wikitext_over_steps(df, args.output_dir)
    save_final_table(df, args.output_dir)
    
    print(f"✅ Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
    # python /home/gap/Documents/peft/examples/qalora_finetuning/visualization/visualize_results_csv.py --csv=results.csv --output_dir=./plots