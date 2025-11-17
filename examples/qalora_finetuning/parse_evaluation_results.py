#!/usr/bin/env python3
"""Parse evaluation results to CSV"""
import json
import re
import pandas as pd
from pathlib import Path
import argparse


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def parse_lm_harness(json_path):
    data = load_json(json_path)
    metrics = {}
    
    for task_name, task_data in data.get('results', {}).items():
        for k, v in task_data.items():
            if not k.endswith('_stderr') and k != 'alias':
                metrics[f"{task_name}_{k.replace(',none', '')}"] = v
    
    return metrics


def discover_experiments(base_dir):
    experiments = []
    
    for exp_dir in sorted(Path(base_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Load config
        config_path = exp_dir / 'run_config.json'
        config = load_json(config_path) if config_path.exists() else {}
        
        eval_dir = exp_dir / 'evaluation'
        if not eval_dir.exists():
            continue
        
        # Parse all evaluation files
        for eval_file in sorted(eval_dir.glob('lm_harness_results*.json')):
            step_match = re.search(r'step_(\d+)', eval_file.name)
            step = int(step_match.group(1)) if step_match else 'final'
            
            metrics = parse_lm_harness(str(eval_file))
            
            # Add training metrics if available
            train_path = eval_dir / 'training_metrics.json'
            if train_path.exists():
                metrics.update(load_json(train_path))
            
            experiments.append({
                'experiment': exp_dir.name,
                'step': step,
                'base_model': config.get('model_name_or_path'),
                'training_mode': config.get('training_mode'),
                'rank': config.get('lora_r'),
                'bits': config.get('bits'),
                'group_size': config.get('qalora_group_size'),
                'learning_rate': config.get('learning_rate'),
                **metrics
            })
    
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--output', default='results.csv')
    args = parser.parse_args()
    
    experiments = discover_experiments(args.base_dir)
    df = pd.DataFrame(experiments).sort_values(['experiment', 'step'])
    df.to_csv(args.output, index=False)
    
    unique_steps = set(df['step'].unique())
    numeric_steps = sorted([s for s in unique_steps if isinstance(s, int)])
    string_steps = sorted([s for s in unique_steps if isinstance(s, str)])
    
    print(f"✅ Saved {len(df)} results to {args.output}")
    print(f"   Experiments: {df['experiment'].nunique()}")
    print(f"   Steps: {numeric_steps + string_steps}")


if __name__ == "__main__":
    main()