"""
Tests for automated evaluation metrics.
"""

from evaluation.metrics import (
    compute_intent_metrics, compute_bleu, compute_rouge_l,
    compute_escalation_metrics
)


def test_intent_metrics_perfect():
    true_labels = ["battery", "billing", "hardware", "software"]
    pred_labels = ["battery", "billing", "hardware", "software"]
    res = compute_intent_metrics(true_labels, pred_labels)
    assert res['accuracy'] == 1.0
    assert res['macro_f1'] == 1.0


def test_intent_metrics_mismatch():
    true_labels = ["battery", "billing", "hardware", "software"]
    pred_labels = ["battery", "billing", "other", "other"]
    res = compute_intent_metrics(true_labels, pred_labels)
    assert res['accuracy'] == 0.5
    assert res['macro_f1'] < 1.0


def test_bleu_identical():
    ref = "Please DM us your Apple ID so we can look into this."
    hyp = "Please DM us your Apple ID so we can look into this."
    score = compute_bleu(ref, hyp)
    assert score > 0.9


def test_bleu_empty():
    assert compute_bleu("hello", "") == 0.0


def test_rouge_l():
    ref = "Try restarting your iPhone to fix the issue."
    hyp = "Try restarting your iPhone to resolve the problem."
    score = compute_rouge_l(ref, hyp)
    assert 0.0 < score <= 1.0


def test_escalation_metrics():
    true_dec = ["auto", "auto", "escalate", "escalate"]
    pred_dec = ["auto", "escalate", "escalate", "escalate"]
    res = compute_escalation_metrics(true_dec, pred_dec)
    assert res['accuracy'] == 0.75
    assert res['escalation_recall'] == 1.0
    assert 0.0 < res['escalation_precision'] < 1.0
