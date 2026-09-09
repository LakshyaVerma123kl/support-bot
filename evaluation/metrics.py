"""
Automated evaluation metrics.

Computes:
- Intent classification: accuracy, macro-F1, per-class precision/recall
- Reply quality: BLEU-4, ROUGE-L vs. actual brand reply
- Escalation: precision, recall, F1
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
# Intent Classification Metrics
# ──────────────────────────────────────────────

def compute_intent_metrics(true_intents, predicted_intents):
    """
    Compute intent classification metrics.

    Args:
        true_intents: List of ground-truth intent labels.
        predicted_intents: List of predicted intent labels.

    Returns:
        dict with accuracy, macro_f1, per_class metrics, confusion_matrix.
    """
    accuracy = accuracy_score(true_intents, predicted_intents)
    macro_f1 = f1_score(true_intents, predicted_intents, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_intents, predicted_intents, average='weighted', zero_division=0)

    # Per-class report
    labels = sorted(set(true_intents + predicted_intents))
    report = classification_report(
        true_intents, predicted_intents,
        labels=labels, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(true_intents, predicted_intents, labels=labels)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1': report[label]['f1-score'],
                'support': report[label]['support'],
            }
            for label in labels if label in report
        },
        'confusion_matrix': cm.tolist(),
        'labels': labels,
    }


# ──────────────────────────────────────────────
# Reply Quality Metrics
# ──────────────────────────────────────────────

def compute_bleu(reference, hypothesis):
    """Compute BLEU-4 score between reference and hypothesis."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if len(hyp_tokens) == 0:
            return 0.0

        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing
        )
        return score

    except ImportError:
        # Fallback: simple unigram overlap
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        if len(hyp_tokens) == 0:
            return 0.0
        overlap = len(ref_tokens & hyp_tokens)
        return overlap / max(len(hyp_tokens), 1)


def compute_rouge_l(reference, hypothesis):
    """Compute ROUGE-L score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    except ImportError:
        # Fallback: LCS-based ROUGE-L
        return _lcs_rouge_l(reference, hypothesis)


def _lcs_rouge_l(reference, hypothesis):
    """Fallback ROUGE-L using longest common subsequence."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS length
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_reply_metrics(references, hypotheses):
    """
    Compute reply quality metrics across a batch.

    Args:
        references: List of reference (actual brand) replies.
        hypotheses: List of generated replies.

    Returns:
        dict with avg_bleu, avg_rouge_l, individual scores.
    """
    bleu_scores = []
    rouge_scores = []

    for ref, hyp in zip(references, hypotheses):
        if ref and hyp:
            bleu_scores.append(compute_bleu(ref, hyp))
            rouge_scores.append(compute_rouge_l(ref, hyp))

    return {
        'avg_bleu4': np.mean(bleu_scores) if bleu_scores else 0.0,
        'avg_rouge_l': np.mean(rouge_scores) if rouge_scores else 0.0,
        'bleu_scores': bleu_scores,
        'rouge_scores': rouge_scores,
        'n_evaluated': len(bleu_scores),
    }


# ──────────────────────────────────────────────
# Escalation Metrics
# ──────────────────────────────────────────────

def compute_escalation_metrics(true_decisions, predicted_decisions):
    """
    Compute escalation decision metrics.

    Args:
        true_decisions: List of ground-truth decisions ('auto' or 'escalate').
        predicted_decisions: List of predicted decisions.

    Returns:
        dict with precision, recall, f1 for each class.
    """
    # Binary: escalate=1, auto=0
    true_binary = [1 if d == 'escalate' else 0 for d in true_decisions]
    pred_binary = [1 if d == 'escalate' else 0 for d in predicted_decisions]

    accuracy = accuracy_score(true_binary, pred_binary)

    # Escalation-specific metrics (treating 'escalate' as positive class)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)

    # Distribution
    true_dist = Counter(true_decisions)
    pred_dist = Counter(predicted_decisions)

    return {
        'accuracy': accuracy,
        'escalation_precision': precision,
        'escalation_recall': recall,
        'escalation_f1': f1,
        'true_distribution': dict(true_dist),
        'predicted_distribution': dict(pred_dist),
    }


def print_metrics_summary(intent_metrics, reply_metrics, escalation_metrics):
    """Print a formatted summary of all metrics."""
    print("\n" + "=" * 60)
    print("                 EVALUATION RESULTS")
    print("=" * 60)

    print("\n[STATS] Intent Classification:")
    print(f"   Accuracy:      {intent_metrics['accuracy']:.3f}")
    print(f"   Macro F1:      {intent_metrics['macro_f1']:.3f}")
    print(f"   Weighted F1:   {intent_metrics['weighted_f1']:.3f}")

    print("\n   Per-class breakdown:")
    for intent, scores in sorted(intent_metrics['per_class'].items()):
        if scores['support'] > 0:
            print(f"     {intent:30s}  P={scores['precision']:.2f}  "
                  f"R={scores['recall']:.2f}  F1={scores['f1']:.2f}  "
                  f"(n={scores['support']})")

    print("\n[NOTE] Reply Quality:")
    print(f"   Avg BLEU-4:    {reply_metrics['avg_bleu4']:.3f}")
    print(f"   Avg ROUGE-L:   {reply_metrics['avg_rouge_l']:.3f}")
    print(f"   N evaluated:   {reply_metrics['n_evaluated']}")

    print("\n[GATE] Escalation Decision:")
    print(f"   Accuracy:      {escalation_metrics['accuracy']:.3f}")
    print(f"   Precision:     {escalation_metrics['escalation_precision']:.3f}")
    print(f"   Recall:        {escalation_metrics['escalation_recall']:.3f}")
    print(f"   F1:            {escalation_metrics['escalation_f1']:.3f}")

    print("\n" + "=" * 60)
