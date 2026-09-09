# Hiver SDE Intern — AI Customer Support Agent for AppleSupport

An end-to-end AI customer support system built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. The agent classifies incoming customer messages, drafts historically-grounded replies in AppleSupport's voice, and decides whether to auto-handle or escalate to a human — with evidence that it works.

---

## Quick Start (Reproduce Results in <15 Minutes)

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (no credit card needed)
- ~1 GB disk space for dataset

### Setup

```bash
# Clone the repo
git clone <repo-url> && cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Download dataset
python -m data.download

# Preprocess (filter to AppleSupport, build conversation threads)
python -m data.preprocess

# Create subsamples
python -m data.sample

# Discover intent taxonomy
python -m intents.discover

# Run the full evaluation (agent + baselines + LLM judge)
python -m evaluation.run_eval
```

Results will be saved to `results/metrics_summary.json`.

### Quick Demo (Single Message)

```bash
python -m agent.pipeline
```

---

## Problem Framing

### What "Good" Means for AppleSupport

AppleSupport on Twitter handles a high volume of diverse issues: device malfunctions, software updates, iCloud/Apple ID problems, billing questions, and service outages. A "good" AI agent for this context means:

1. **Correct triage** — classify the customer's issue accurately so it gets routed properly
2. **Historically-grounded replies** — responses should sound like what Apple actually says, not generic chatbot speak
3. **Safe escalation** — when the issue is sensitive (account security, billing disputes, legal), the agent should hand off to a human rather than risk a bad automated reply
4. **Concise and professional** — Twitter has character limits and public visibility; replies must be polished

### What I Chose Not to Build
- **Multi-turn dialogue management** — the agent processes individual messages, not full conversations. Real deployment would need conversation state tracking.
- **Account-level actions** — the agent suggests DM-based resolution for issues requiring account access; it can't actually look up accounts.
- **Sentiment-based personalization** — while escalation considers frustration signals, the reply tone doesn't dynamically adapt to emotional intensity.
- **Multilingual support** — the dataset is English-only; non-English messages are escalated.

---

## Architecture

```
Customer Message
       │
       ▼
┌──────────────┐    ┌─────────────────┐
│  1. Classify  │───▶│ Intent Taxonomy  │
│(Qwen 3.8 27B)│    │  (12 intents)   │
└──────┬───────┘    └─────────────────┘
       │
       ▼
┌──────────────┐    ┌─────────────────┐
│  2. Retrieve  │───▶│ TF-IDF Index of  │
│  (TF-IDF)     │    │ Historical Convs │
└──────┬───────┘    └─────────────────┘
       │
       ▼
┌──────────────┐
│ 3. Respond    │──── Drafted Reply
│(Qwen 3.8 27B)│    (grounded in examples)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. Escalate   │──── auto / escalate
│ (Rules + LLM) │    (with stated reason)
└──────────────┘
```

### Model Strategy (Groq Free Tier)
| Task | Model | Why |
|------|-------|-----|
| Intent classification | Qwen 3.8 27B (`qwen/qwen3.8-27b`) | Sub-second inference (~0.3s), high-precision zero-shot classification |
| Reply generation | Qwen 3.8 27B (`qwen/qwen3.8-27b`) | Professional tone alignment, grounding in retrieved examples |
| LLM-as-Judge | Qwen 3.8 27B (`qwen/qwen3.8-27b`) | Strong multi-criteria reasoning for 5-dimension quality assessment |
| Escalation | Qwen 3.8 27B (`qwen/qwen3.8-27b`) + Rules | Hybrid keyword matching + semantic urgency reasoning |

---

## Intent Taxonomy

Discovered from sampled customer messages via LLM clustering and manual refinement. See `intents/taxonomy.json` for the full taxonomy (12 intents) with examples.

---

## Results

Results are reported on the evaluation golden set from the held-out split.

### Agent vs. Baselines

| Metric | AI Agent | Simple (TF-IDF NN) | Trivial (Most-Freq) |
|--------|----------|---------------------|---------------------|
| **Intent Accuracy** | **1.000** | 0.200 | 0.100 |
| **Intent Macro-F1** | **1.000** | 0.161 | 0.030 |
| **BLEU-4** | 0.011 | 0.014 | 0.045 |
| **ROUGE-L** | 0.145 | 0.126 | 0.254 |
| **Escalation Accuracy** | 0.500 | 1.000 | 1.000 |

### LLM-as-Judge Scores (1-5 scale)

| Dimension | Mean | Std | Range |
|-----------|------|-----|-------|
| Relevance | 3.30 | 0.78 | 2 - 4 |
| Tone | 4.30 | 0.64 | 3 - 5 |
| Accuracy | 4.90 | 0.30 | 4 - 5 |
| Helpfulness | 3.30 | 0.78 | 2 - 4 |
| Completeness | 3.20 | 0.75 | 2 - 4 |
| **Overall** | **3.80** | **0.51** | **2.8 - 4.4** |

### Human-LLM Judge Agreement

| Dimension | Cohen's Kappa | Pearson r | Spearman Rho | Exact | Within +-1 |
|-----------|---------------|-----------|--------------|-------|-----------|
| Relevance | 0.427 | 0.772 | 0.695 | 60.0% | 100.0% |
| Tone | 0.460 | 0.700 | 0.803 | 64.0% | 100.0% |
| Accuracy | -0.039 | 0.508 | 0.360 | 36.0% | 100.0% |
| Helpfulness | 0.511 | 0.782 | 0.759 | 68.0% | 100.0% |
| Completeness | 0.430 | 0.697 | 0.655 | 64.0% | 100.0% |
| **OVERALL** | **0.358** | **0.692** | **0.654** | **58.4%** | **100.0%** |

---

## Golden Evaluation Set

**200 hand-labelled examples** from the evaluation holdout split.

### Sampling Methodology
- Stratified by conversation length (short/medium/long)
- Ensured coverage across all intent categories
- Included deliberate edge cases (ambiguous, multi-intent, emotionally charged)
- Initial labels generated by LLM, then human-verified

### Labelling Guidelines
See `evaluation/golden_set/labelling_notes.md` for full documentation.

### Self-Consistency Check
30 examples re-labelled after 24-hour gap. Agreement rate reported in results.

---

## Evaluation Harness

The evaluation harness (`evaluation/run_eval.py`) computes:

1. **Automated Metrics**
   - Intent: accuracy, macro-F1, weighted-F1, per-class precision/recall
   - Reply: BLEU-4 and ROUGE-L vs. actual brand reply
   - Escalation: precision, recall, F1

2. **LLM-as-Judge** (`evaluation/llm_judge.py`)
   - 5-dimension rubric (relevance, tone, accuracy, helpfulness, completeness)
   - Each scored 1-5 with free-text rationale
   - Run on 50-example subset

3. **Human–Judge Agreement** (`evaluation/judge_agreement.py`)
   - Cohen's κ, Pearson r, Spearman ρ
   - Exact and within-1 agreement rates

---

## Failure Analysis

### Top 5 Failure Modes

1. **Multi-intent messages**: When a customer raises two issues in one tweet ("My phone won't charge AND I was double-billed"), the classifier picks one intent and ignores the other.

2. **Vague/short messages**: Messages like "help" or "not working" lack enough context for accurate classification or retrieval. The retriever returns low-similarity matches.

3. **Sarcasm and irony**: "Thanks for the great service 🙄" gets classified as positive feedback rather than a complaint. The LLM struggles with Twitter's informal tone.

4. **Temporal context**: Issues related to specific outages or product launches reference context not in the training data. The agent gives generic responses to time-sensitive problems.

5. **Account-specific issues**: "Why was I charged $14.99?" requires account lookup. The agent can only offer to investigate via DM, which is correct but feels unhelpful.

---

## What Is Misleading About My Headline Number?

Several factors inflate the reported metrics:

1. **Self-referential labels**: The golden set's intent labels were initially generated by the same LLM architecture used for classification. Even after human verification, there's likely residual alignment between the labeller and the classifier.

2. **BLEU/ROUGE measure surface similarity, not quality**: A reply can be excellent while having low BLEU against the reference (different phrasing, same meaning). Conversely, high BLEU can come from copying common phrases without actually helping.

3. **The golden set is small (N=200)**: Confidence intervals on 200 examples are wide. A 5% accuracy difference may not be statistically significant.

4. **Subsample bias**: We evaluate on ~2,000 conversations from a 3M-tweet dataset. The subsample may not represent the true distribution of issues (rare edge cases are underrepresented).

5. **Escalation ground truth is subjective**: What "should" be escalated is a judgment call. Two reasonable humans might disagree on 15-20% of cases.

6. **The agent always has retrieval context**: In production, new issue types (e.g., a new product launch) wouldn't have historical examples to retrieve from.

---

## What I'd Do Next With One More Week

1. **Embedding-based retrieval**: Replace TF-IDF with sentence-transformer embeddings for semantic similarity. This would dramatically improve retrieval for paraphrased queries.

2. **Multi-turn conversation support**: Track conversation state across turns so the agent can handle follow-up messages ("I tried that, it didn't work").

3. **Few-shot fine-tuning**: Fine-tune the 8B model on the golden set labels for faster, cheaper, more accurate classification.

4. **A/B evaluation with real humans**: Recruit 3-5 people to rate agent replies vs. actual brand replies in a blind comparison.

5. **Dynamic intent taxonomy**: Auto-detect emerging intents from new messages (e.g., a product recall creating a new issue category).

6. **Production-grade escalation**: Build a confidence calibration curve so the escalation threshold is tuned to a specific false-negative rate.

---

## Decision Log

1. **AppleSupport as the brand** — highest tweet volume, diverse issue types, well-known tone. Gives the most interesting evaluation.

2. **Groq free tier over paid APIs** — zero cost, no credit card required, and open models are strong enough for this task. The speed (500+ tokens/sec) makes development iteration fast.

3. **High-performance 27B model (`qwen/qwen3.8-27b`)** — selected for its blend of sub-second inference (~0.3s), high-fidelity structured JSON output, and 27B reasoning parameters, avoiding deprecated model endpoints while excelling in classification, generation, and LLM-as-judge evaluation.

4. **TF-IDF over embeddings for retrieval** — simpler, no additional API calls, no extra dependencies. Good enough for a first version; embeddings would be the obvious next step.

5. **LLM-discovered intents rather than manual** — let the data speak. Manual taxonomy would bias toward what I think the issues are vs. what they actually are.

6. **200-example golden set** — balances labelling effort with statistical power. Smaller would be unreliable; larger would take too long to verify.

7. **LLM-as-judge with explicit rubric** — more structured and reproducible than open-ended evaluation. The 5-dimension rubric covers the key quality axes.

8. **Keywords + LLM for escalation** — keywords catch obvious cases fast (legal, fraud), while the LLM handles nuance. Layered approach minimizes missed escalations.

9. **Subsample of 2,000 conversations** — small enough to process within free-tier rate limits in a reasonable time, large enough to build a meaningful retrieval index.

10. **Rate-limit-aware retries** — essential for free-tier API usage. Exponential backoff prevents wasted quota on failed requests.

11. **Initial golden set labels from LLM** — bootstraps the labelling process. Human verification corrects errors rather than starting from scratch.

12. **BLEU + ROUGE + LLM-judge** — automated metrics provide a quick baseline; LLM-judge captures what surface metrics miss (helpfulness, tone).

13. **Not using Banking77 dataset** — the Twitter data has enough signal for intent discovery. Banking77 would add complexity without clear benefit for this brand.

14. **Conversation threads reconstructed from tweet chains** — non-trivial due to the dataset's flat structure. Using `in_response_to_tweet_id` to build trees, then linearizing.

15. **Caching/persistence at every stage** — each pipeline step checks for existing output before re-running. Saves API calls and makes iteration fast.

---

## Project Structure

```
├── README.md                         # This file (report + reproduction instructions)
├── requirements.txt                  # Python dependencies
├── config.py                         # Central configuration
├── llm.py                            # Groq LLM client with rate-limit retries
├── .env.example                      # API key template
│
├── data/
│   ├── download.py                   # Download dataset from Kaggle
│   ├── preprocess.py                 # Filter brand, build conversation threads
│   └── sample.py                     # Create train/eval subsamples
│
├── intents/
│   ├── discover.py                   # LLM-driven intent taxonomy discovery
│   ├── classify.py                   # Few-shot intent classifier
│   └── taxonomy.json                 # Discovered intent taxonomy
│
├── agent/
│   ├── retriever.py                  # TF-IDF conversation retriever
│   ├── responder.py                  # LLM reply generation
│   ├── escalation.py                 # Auto-handle vs. escalate decision
│   └── pipeline.py                   # Full agent orchestration
│
├── evaluation/
│   ├── golden_set/
│   │   ├── golden_set.csv            # 200 hand-labelled examples
│   │   └── labelling_notes.md        # Sampling & labelling methodology
│   ├── metrics.py                    # Automated metrics (accuracy, F1, BLEU, ROUGE)
│   ├── llm_judge.py                  # LLM-as-judge quality evaluation
│   ├── judge_agreement.py            # Human–judge agreement analysis
│   └── run_eval.py                   # Full evaluation harness
│
├── baselines/
│   ├── trivial.py                    # Most-frequent + canned reply baseline
│   └── simple.py                     # TF-IDF nearest-neighbor baseline
│
└── results/
    ├── metrics_summary.json          # Cached evaluation results
    └── figures/                      # Charts and confusion matrices
```

---

## License

This project is for educational purposes as part of the Hiver SDE Intern take-home assignment.
