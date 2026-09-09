"""
Full evaluation harness.

Runs the agent pipeline (and baselines) on the golden evaluation set,
computes all metrics, runs LLM-as-judge, and saves results.

Usage:
    python -m evaluation.run_eval
"""

import sys
import json
import time
import csv
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GOLDEN_SET_DIR, RESULTS_DIR, PROCESSED_DATA_DIR,
    BRAND_NAME, GOLDEN_SET_SIZE, RANDOM_SEED
)
from evaluation.metrics import (
    compute_intent_metrics, compute_reply_metrics,
    compute_escalation_metrics, print_metrics_summary
)
from evaluation.llm_judge import judge_batch, summarize_judgments, print_judge_summary


def load_golden_set(limit=None):
    """Load the golden evaluation set."""
    golden_path = GOLDEN_SET_DIR / "golden_set.csv"

    if not golden_path.exists():
        print(f"Golden set not found at {golden_path}")
        print("Generating golden set from evaluation holdout...")
        generate_golden_set(size=limit or GOLDEN_SET_SIZE)

    examples = []
    with open(golden_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(row)

    if limit and limit < len(examples):
        examples = examples[:limit]

    print(f"Loaded {len(examples)} golden set examples")
    return examples


def generate_golden_set(size=GOLDEN_SET_SIZE):
    """
    Generate the golden evaluation set from the evaluation holdout.

    This creates the initial CSV that needs manual human labelling.
    The LLM provides initial labels that should be verified by a human.
    """
    import random
    from intents.classify import load_taxonomy, build_classification_prompt, classify_message

    holdout_path = PROCESSED_DATA_DIR / "eval_holdout.jsonl"
    if not holdout_path.exists():
        print(f"[X] Eval holdout not found at {holdout_path}")
        print("  Run: python -m data.sample")
        sys.exit(1)

    # Load holdout conversations
    conversations = []
    with open(holdout_path, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line))

    # Extract customer message + brand reply pairs
    pairs = []
    for conv in conversations:
        messages = conv['messages']
        for i, msg in enumerate(messages):
            if not msg['is_brand'] and len(msg['text']) > 20:
                # Find next brand reply
                brand_reply = ""
                for j in range(i + 1, len(messages)):
                    if messages[j]['is_brand'] and len(messages[j]['text']) > 10:
                        brand_reply = messages[j]['text']
                        break
                if brand_reply:
                    pairs.append({
                        'conversation_id': conv['conversation_id'],
                        'customer_message': msg['text'],
                        'actual_brand_reply': brand_reply,
                    })

    # Sample
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    sampled = pairs[:size]

    print(f"Generating golden set with {len(sampled)} examples...")
    print("Using LLM for initial intent labels (should be verified by human)...")

    # Load classifier for initial labelling
    taxonomy = load_taxonomy()
    system_prompt = build_classification_prompt(taxonomy)

    # Classify each example
    golden_examples = []
    for item in tqdm(sampled, desc="Labelling golden set"):
        try:
            classification = classify_message(item['customer_message'], system_prompt)
            intent = classification['intent']
            confidence = classification['confidence']
        except Exception:
            intent = "other"
            confidence = 0.0

        # Initial escalation decision
        expected_escalation = "escalate" if confidence < 0.5 else "auto"

        golden_examples.append({
            'conversation_id': item['conversation_id'],
            'customer_message': item['customer_message'],
            'true_intent': intent,  # LLM-labelled; should be human-verified
            'actual_brand_reply': item['actual_brand_reply'],
            'expected_escalation': expected_escalation,
            'quality_notes': f'LLM-labelled (confidence={confidence:.2f}); needs human verification',
        })

        time.sleep(0.3)  # Rate limit respect

    # Save
    golden_path = GOLDEN_SET_DIR / "golden_set.csv"
    with open(golden_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'conversation_id', 'customer_message', 'true_intent',
            'actual_brand_reply', 'expected_escalation', 'quality_notes'
        ])
        writer.writeheader()
        writer.writerows(golden_examples)

    print(f"[OK] Golden set saved to {golden_path}")
    print(f"  [!] Please review and verify the LLM-assigned labels!")
    return golden_examples


def run_agent_evaluation(golden_set):
    """Run the agent pipeline on the golden set and compute metrics."""
    from agent.pipeline import SupportAgent

    print("\n" + "=" * 60)
    print("Running Agent Evaluation")
    print("=" * 60)

    agent = SupportAgent()

    # Process each golden set example
    results = []
    for example in tqdm(golden_set, desc="Evaluating agent"):
        try:
            result = agent.process_message(example['customer_message'])
            result['true_intent'] = example['true_intent']
            result['actual_brand_reply'] = example['actual_brand_reply']
            result['expected_escalation'] = example['expected_escalation']
            results.append(result)
        except Exception as e:
            print(f"\n  [!] Error: {str(e)[:100]}")
            results.append({
                'customer_message': example['customer_message'],
                'true_intent': example['true_intent'],
                'actual_brand_reply': example['actual_brand_reply'],
                'expected_escalation': example['expected_escalation'],
                'intent': 'error',
                'drafted_reply': '',
                'escalation_decision': 'escalate',
                'error': str(e),
            })
        time.sleep(0.5)

    # Compute metrics
    true_intents = [r['true_intent'] for r in results]
    pred_intents = [r.get('intent', 'error') for r in results]
    intent_metrics = compute_intent_metrics(true_intents, pred_intents)

    references = [r.get('actual_brand_reply', '') for r in results]
    hypotheses = [r.get('drafted_reply', '') for r in results]
    reply_metrics = compute_reply_metrics(references, hypotheses)

    true_escalation = [r.get('expected_escalation', 'auto') for r in results]
    pred_escalation = [r.get('escalation_decision', 'auto') for r in results]
    escalation_metrics = compute_escalation_metrics(true_escalation, pred_escalation)

    print_metrics_summary(intent_metrics, reply_metrics, escalation_metrics)

    return results, intent_metrics, reply_metrics, escalation_metrics


def run_llm_judge_evaluation(results, sample_size=50):
    """Run LLM-as-judge on a subset of results."""
    print(f"\nRunning LLM-as-Judge on {min(sample_size, len(results))} examples...")

    # Prepare evaluation data
    eval_data = []
    for r in results[:sample_size]:
        if r.get('drafted_reply'):
            eval_data.append({
                'customer_message': r['customer_message'],
                'generated_reply': r['drafted_reply'],
                'actual_reply': r.get('actual_brand_reply', ''),
                'intent': r.get('intent', ''),
            })

    if not eval_data:
        print("  [!] No valid replies to judge")
        return None, None

    judgments = judge_batch(eval_data)
    summary = summarize_judgments(judgments)
    print_judge_summary(summary)

    return judgments, summary


def run_baseline_evaluation(golden_set):
    """Run baseline evaluations for comparison."""
    from baselines.trivial import TrivialBaseline
    from baselines.simple import SimpleBaseline

    baseline_results = {}

    for BaselineClass, name in [(TrivialBaseline, "Trivial"), (SimpleBaseline, "Simple")]:
        print(f"\n{'='*60}")
        print(f"Running {name} Baseline")
        print(f"{'='*60}")

        baseline = BaselineClass()
        baseline.fit()

        results = []
        for example in tqdm(golden_set, desc=f"Evaluating {name}"):
            result = baseline.predict(example['customer_message'])
            result['true_intent'] = example['true_intent']
            result['actual_brand_reply'] = example['actual_brand_reply']
            result['expected_escalation'] = example['expected_escalation']
            results.append(result)

        # Compute metrics
        true_intents = [r['true_intent'] for r in results]
        pred_intents = [r.get('intent', 'other') for r in results]
        intent_metrics = compute_intent_metrics(true_intents, pred_intents)

        references = [r.get('actual_brand_reply', '') for r in results]
        hypotheses = [r.get('reply', '') for r in results]
        reply_metrics = compute_reply_metrics(references, hypotheses)

        true_esc = [r.get('expected_escalation', 'auto') for r in results]
        pred_esc = [r.get('escalation', 'auto') for r in results]
        escalation_metrics = compute_escalation_metrics(true_esc, pred_esc)

        print_metrics_summary(intent_metrics, reply_metrics, escalation_metrics)

        baseline_results[name.lower()] = {
            'intent': intent_metrics,
            'reply': reply_metrics,
            'escalation': escalation_metrics,
        }

    return baseline_results


def save_results(agent_results, intent_metrics, reply_metrics,
                 escalation_metrics, judge_summary, baseline_results):
    """Save all evaluation results."""
    # Summary JSON
    summary = {
        'brand': BRAND_NAME,
        'golden_set_size': len(agent_results),
        'agent': {
            'intent': {k: v for k, v in intent_metrics.items()
                      if k != 'confusion_matrix'},
            'reply': {k: v for k, v in reply_metrics.items()
                     if k not in ('bleu_scores', 'rouge_scores')},
            'escalation': escalation_metrics,
        },
        'baselines': {},
        'llm_judge': judge_summary,
    }

    # Add baseline results
    for name, metrics in baseline_results.items():
        summary['baselines'][name] = {
            'intent': {k: v for k, v in metrics['intent'].items()
                      if k != 'confusion_matrix'},
            'reply': {k: v for k, v in metrics['reply'].items()
                     if k not in ('bleu_scores', 'rouge_scores')},
            'escalation': metrics['escalation'],
        }

    results_path = RESULTS_DIR / "metrics_summary.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[OK] Results saved to {results_path}")

    # Save detailed agent results
    details_path = RESULTS_DIR / "agent_results_detailed.jsonl"
    with open(details_path, 'w', encoding='utf-8') as f:
        for r in agent_results:
            # Remove non-serializable fields
            clean = {k: v for k, v in r.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            f.write(json.dumps(clean, default=str, ensure_ascii=False) + '\n')

    print(f"[OK] Detailed results saved to {details_path}")
    return summary


def print_comparison_table(summary):
    """Print a comparison table of agent vs baselines."""
    print("\n" + "=" * 70)
    print("                    COMPARISON: AGENT vs BASELINES")
    print("=" * 70)

    headers = ["Metric", "Agent", "Trivial", "Simple"]
    rows = [
        ("Intent Accuracy",
         summary['agent']['intent']['accuracy'],
         summary['baselines'].get('trivial', {}).get('intent', {}).get('accuracy', 0),
         summary['baselines'].get('simple', {}).get('intent', {}).get('accuracy', 0)),
        ("Intent Macro-F1",
         summary['agent']['intent']['macro_f1'],
         summary['baselines'].get('trivial', {}).get('intent', {}).get('macro_f1', 0),
         summary['baselines'].get('simple', {}).get('intent', {}).get('macro_f1', 0)),
        ("BLEU-4",
         summary['agent']['reply']['avg_bleu4'],
         summary['baselines'].get('trivial', {}).get('reply', {}).get('avg_bleu4', 0),
         summary['baselines'].get('simple', {}).get('reply', {}).get('avg_bleu4', 0)),
        ("ROUGE-L",
         summary['agent']['reply']['avg_rouge_l'],
         summary['baselines'].get('trivial', {}).get('reply', {}).get('avg_rouge_l', 0),
         summary['baselines'].get('simple', {}).get('reply', {}).get('avg_rouge_l', 0)),
        ("Escalation F1",
         summary['agent']['escalation']['escalation_f1'],
         summary['baselines'].get('trivial', {}).get('escalation', {}).get('escalation_f1', 0),
         summary['baselines'].get('simple', {}).get('escalation', {}).get('escalation_f1', 0)),
    ]

    print(f"\n{'Metric':<20} {'Agent':>10} {'Trivial':>10} {'Simple':>10}")
    print("-" * 52)
    for name, agent_val, trivial_val, simple_val in rows:
        print(f"{name:<20} {agent_val:>10.3f} {trivial_val:>10.3f} {simple_val:>10.3f}")
    print("=" * 70)


def main():
    """Run the full evaluation harness."""
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to evaluate")
    args = parser.parse_args()

    print("=" * 64)
    print("         AI Customer Support Agent -- Full Evaluation")
    print(f"         Brand: {BRAND_NAME:<45s}")
    print("=" * 64)

    # Load golden set
    golden_set = load_golden_set(limit=args.limit)

    # Run baseline evaluations (no API calls needed for trivial)
    baseline_results = run_baseline_evaluation(golden_set)

    # Run agent evaluation
    agent_results, intent_metrics, reply_metrics, escalation_metrics = \
        run_agent_evaluation(golden_set)

    # Run LLM-as-judge
    judgments, judge_summary = run_llm_judge_evaluation(agent_results)

    # Save all results
    summary = save_results(
        agent_results, intent_metrics, reply_metrics,
        escalation_metrics, judge_summary, baseline_results
    )

    # Print comparison table
    print_comparison_table(summary)

    print("\n[DONE] Evaluation complete!")
    print(f"   Results: {RESULTS_DIR / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
