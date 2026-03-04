import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_input_comparison(codepatch_results, gemma_results, output_dir):
    """Generates a bar chart comparing the input token counts."""
    codepatch_tokens = [r['metrics']['total_input_constructs'] for r in codepatch_results]
    gemma_tokens = [r['metrics']['total_input_constructs'] for r in gemma_results]
    
    labels = [f'Sample {i+1}' for i in range(len(codepatch_tokens))]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, codepatch_tokens, width, label='CodePatch')
    rects2 = ax.bar(x + width/2, gemma_tokens, width, label='Gemma-2b')

    ax.set_ylabel('Number of Input Tokens/Constructs')
    ax.set_title('Input Size Comparison: CodePatch vs. Gemma-2b')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/input_comparison.png")
    print(f"Saved input comparison plot to {output_dir}/input_comparison.png")

def plot_rouge_comparison(codepatch_results, gemma_results, output_dir):
    """Generates a grouped bar chart for average ROUGE scores."""
    codepatch_rouge1 = np.mean([r['metrics']['rouge1'] for r in codepatch_results])
    codepatch_rouge2 = np.mean([r['metrics']['rouge2'] for r in codepatch_results])
    codepatch_rougeL = np.mean([r['metrics']['rougeL'] for r in codepatch_results])
    
    gemma_rouge1 = np.mean([r['metrics']['rouge1'] for r in gemma_results])
    gemma_rouge2 = np.mean([r['metrics']['rouge2'] for r in gemma_results])
    gemma_rougeL = np.mean([r['metrics']['rougeL'] for r in gemma_results])

    labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    codepatch_scores = [codepatch_rouge1, codepatch_rouge2, codepatch_rougeL]
    gemma_scores = [gemma_rouge1, gemma_rouge2, gemma_rougeL]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, codepatch_scores, width, label='CodePatch')
    rects2 = ax.bar(x + width/2, gemma_scores, width, label='Gemma-2b')

    ax.set_ylabel('F-Measure')
    ax.set_title('Average ROUGE Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)

    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)

    fig.tight_layout()
    plt.savefig(f"{output_dir}/rouge_comparison.png")
    print(f"Saved ROUGE comparison plot to {output_dir}/rouge_comparison.png")

def plot_kv_cache_comparison(codepatch_results, gemma_results, output_dir):
    """Generates a bar chart comparing the initial KV cache size in MB."""
    codepatch_cache = [r['metrics']['kv_cache_size_mb'] for r in codepatch_results]
    gemma_cache = [r['metrics']['kv_cache_size_mb'] for r in gemma_results]
    
    labels = [f'Sample {i+1}' for i in range(len(codepatch_cache))]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, codepatch_cache, width, label='CodePatch')
    rects2 = ax.bar(x + width/2, gemma_cache, width, label='Gemma-2b')

    ax.set_ylabel('Initial KV Cache Size (MB)')
    ax.set_title('KV Cache Size Comparison: CodePatch vs. Gemma-2b')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/kv_cache_comparison.png")
    print(f"Saved KV Cache comparison plot to {output_dir}/kv_cache_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Plot comparison graphs from evaluation results.")
    parser.add_argument("--codepatch_results_path", type=str, required=True, help="Path to CodePatch evaluation results JSON file.")
    parser.add_argument("--gemma_results_path", type=str, required=True, help="Path to Gemma evaluation results JSON file.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the plots.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.codepatch_results_path, 'r') as f:
        codepatch_results = json.load(f)
    
    with open(args.gemma_results_path, 'r') as f:
        gemma_results = json.load(f)

    plot_input_comparison(codepatch_results, gemma_results, args.output_dir)
    plot_rouge_comparison(codepatch_results, gemma_results, args.output_dir)
    plot_kv_cache_comparison(codepatch_results, gemma_results, args.output_dir)

if __name__ == "__main__":
    main()
