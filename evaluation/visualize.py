"""
Generate visualization figures for the evaluation report.

Produces publication-quality figures saved to results/figures/:
1. baseline_comparison.png - AI Agent vs Baselines on automated metrics
2. llm_judge_dimensions.png - 5-dimension quality assessment breakdown
3. judge_agreement.png - Human-LLM judge calibration metrics
4. intent_confusion_matrix.png - Intent classification heatmap

Usage:
    python -m evaluation.visualize
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, FIGURES_DIR, JUDGE_DIMENSIONS

# Apple-inspired aesthetic palette
COLOR_PRIMARY = "#0071e3"     # Apple Blue
COLOR_SECONDARY = "#34c759"   # Apple Green
COLOR_TERTIARY = "#ff9500"    # Apple Orange
COLOR_DANGER = "#ff3b30"      # Apple Red
COLOR_PURPLE = "#af52de"      # Apple Purple
COLOR_BG_DARK = "#1c1c1e"     # Dark background
COLOR_TEXT_LIGHT = "#f5f5f7"  # Light text
COLOR_GRAY = "#86868b"        # Apple Gray


def set_plot_style():
    """Apply modern, clean aesthetic style to matplotlib."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'axes.edgecolor': '#d2d2d7',
        'axes.linewidth': 0.8,
        'grid.color': '#e5e5ea',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.7,
        'figure.autolayout': True,
        'figure.dpi': 200,
    })


def plot_baseline_comparison(summary_path=None):
    """Plot AI Agent vs Baselines comparison across key metrics."""
    if summary_path is None:
        summary_path = RESULTS_DIR / "metrics_summary.json"

    if not summary_path.exists():
        print(f"[!] {summary_path} not found. Run evaluation first.")
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    metrics = ['Intent Accuracy', 'Intent Macro-F1', 'BLEU-4 (x10)', 'ROUGE-L']
    
    agent_vals = [
        summary['agent']['intent']['accuracy'],
        summary['agent']['intent']['macro_f1'],
        summary['agent']['reply']['avg_bleu4'] * 10,  # Scaled for visibility
        summary['agent']['reply']['avg_rouge_l'],
    ]
    
    simple_vals = [
        summary['baselines'].get('simple', {}).get('intent', {}).get('accuracy', 0),
        summary['baselines'].get('simple', {}).get('intent', {}).get('macro_f1', 0),
        summary['baselines'].get('simple', {}).get('reply', {}).get('avg_bleu4', 0) * 10,
        summary['baselines'].get('simple', {}).get('reply', {}).get('avg_rouge_l', 0),
    ]
    
    trivial_vals = [
        summary['baselines'].get('trivial', {}).get('intent', {}).get('accuracy', 0),
        summary['baselines'].get('trivial', {}).get('intent', {}).get('macro_f1', 0),
        summary['baselines'].get('trivial', {}).get('reply', {}).get('avg_bleu4', 0) * 10,
        summary['baselines'].get('trivial', {}).get('reply', {}).get('avg_rouge_l', 0),
    ]

    x = np.arange(len(metrics))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width, agent_vals, width, label='AI Agent (Ours)', color=COLOR_PRIMARY, edgecolor='none', zorder=3)
    rects2 = ax.bar(x, simple_vals, width, label='Simple (TF-IDF NN)', color=COLOR_TERTIARY, edgecolor='none', zorder=3)
    rects3 = ax.bar(x + width, trivial_vals, width, label='Trivial (Most-Freq)', color=COLOR_GRAY, edgecolor='none', zorder=3)

    ax.set_ylabel('Score / Value', fontsize=11, fontweight='500', color='#1d1d1f')
    ax.set_title('AI Support Agent vs. Baselines on Evaluation Benchmark', fontsize=13, fontweight='600', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight='500')
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#e5e5ea', fontsize=9.5)
    ax.grid(axis='y', zorder=0)
    ax.set_ylim(0, 1.15)

    # Value labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=COLOR_PRIMARY)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, color='#48484a')

    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, color='#48484a')

    output_path = FIGURES_DIR / "baseline_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved {output_path}")


def plot_llm_judge_dimensions(summary_path=None):
    """Plot LLM-as-Judge 5-dimension quality breakdown."""
    if summary_path is None:
        summary_path = RESULTS_DIR / "metrics_summary.json"

    if not summary_path.exists():
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    judge = summary.get('llm_judge', {})
    if not judge or 'overall' not in judge:
        return

    dims = [d for d in JUDGE_DIMENSIONS if d in judge]
    means = [judge[d]['mean'] for d in dims]
    stds = [judge[d].get('std', 0.0) for d in dims]
    dim_labels = [d.capitalize() for d in dims]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(dims))

    bars = ax.barh(y_pos, means, xerr=stds, align='center', color=COLOR_PRIMARY,
                   alpha=0.85, capsize=4, ecolor='#1d1d1f', zorder=3)

    # Reference threshold line at 3.0 (acceptable) and 4.0 (good)
    ax.axvline(3.0, color=COLOR_TERTIARY, linestyle='--', linewidth=1, alpha=0.8, label='Acceptable (3.0)')
    ax.axvline(4.0, color=COLOR_SECONDARY, linestyle='--', linewidth=1, alpha=0.8, label='Target (4.0)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(dim_labels, fontsize=10.5, fontweight='500')
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Score (1 = Poor, 5 = Excellent)', fontsize=10.5, fontweight='500')
    ax.set_xlim(1.0, 5.2)
    ax.set_title('LLM-as-Judge Quality Assessment (5 Dimensions)', fontsize=13, fontweight='600', pad=15)
    ax.grid(axis='x', zorder=0)
    ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#e5e5ea', fontsize=9)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + (std if std > 0 else 0.1) + 0.12, bar.get_y() + bar.get_height()/2,
                f'{mean:.2f} ± {std:.2f}', va='center', ha='left', fontsize=9, fontweight='600', color='#1d1d1f')

    output_path = FIGURES_DIR / "llm_judge_dimensions.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved {output_path}")


def plot_judge_agreement(agreement_path=None):
    """Plot Human-LLM agreement across dimensions."""
    if agreement_path is None:
        agreement_path = RESULTS_DIR / "judge_agreement.json"

    if not agreement_path.exists():
        return

    with open(agreement_path, 'r', encoding='utf-8') as f:
        agreement = json.load(f)

    dims = [d for d in JUDGE_DIMENSIONS if d in agreement]
    if not dims:
        return

    labels = [d.capitalize() for d in dims]
    pearson_r = [agreement[d]['pearson_r'] for d in dims]
    exact = [agreement[d]['exact_agreement'] * 100 for d in dims]
    within1 = [agreement[d]['within_1_agreement'] * 100 for d in dims]

    x = np.arange(len(dims))
    width = 0.28

    fig, ax = plt.subplots(figsize=(9, 4.8))
    rects1 = ax.bar(x - width/2, exact, width, label='Exact Score Match (%)', color='#5856d6', zorder=3)
    rects2 = ax.bar(x + width/2, within1, width, label='Within ±1 Match (%)', color=COLOR_SECONDARY, zorder=3)

    ax.set_ylabel('Agreement Rate (%)', fontsize=10.5, fontweight='500')
    ax.set_title('Human vs. LLM Judge Calibration & Agreement', fontsize=13, fontweight='600', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='500')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', zorder=0)
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e5e5ea', fontsize=9.5)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f'{h:.0f}%', xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='600')

    for r in rects2:
        h = r.get_height()
        ax.annotate(f'{h:.0f}%', xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='600')

    output_path = FIGURES_DIR / "judge_agreement.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved {output_path}")


def plot_confusion_matrix(details_path=None):
    """Plot intent classification confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    if details_path is None:
        details_path = RESULTS_DIR / "agent_results_detailed.jsonl"

    if not details_path.exists():
        return

    true_intents = []
    pred_intents = []
    with open(details_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if 'true_intent' in r and 'intent' in r:
                true_intents.append(r['true_intent'])
                pred_intents.append(r['intent'])

    if not true_intents:
        return

    labels = sorted(set(true_intents + pred_intents))
    cm = confusion_matrix(true_intents, pred_intents, labels=labels)
    cm_array = np.array(cm, dtype=float)
    display_labels = [l.replace('_', ' ').title() for l in labels]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Sample Count', rotation=-90, va="bottom", fontsize=10)

    ax.set(xticks=np.arange(cm_array.shape[1]),
           yticks=np.arange(cm_array.shape[0]),
           xticklabels=display_labels, yticklabels=display_labels,
           title="Intent Classification Confusion Matrix (Agent)",
           ylabel="Ground Truth Intent",
           xlabel="Predicted Intent")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8.5)
    plt.setp(ax.get_yticklabels(), fontsize=8.5)

    # Loop over data dimensions and create text annotations
    thresh = cm_array.max() / 2.
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            val = int(cm_array[i, j])
            ax.text(j, i, f"{val}" if val > 0 else "0",
                    ha="center", va="center",
                    color="white" if cm_array[i, j] > thresh else "#1d1d1f",
                    fontsize=9, fontweight='600' if val > 0 else 'normal')

    output_path = FIGURES_DIR / "intent_confusion_matrix.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved {output_path}")


def main():
    """Generate all figures."""
    print("Generating evaluation figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    plot_baseline_comparison()
    plot_llm_judge_dimensions()
    plot_judge_agreement()
    plot_confusion_matrix()
    print(f"\n[DONE] All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
