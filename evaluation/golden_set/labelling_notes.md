# Golden Evaluation Set — Labelling Notes

## Sampling Methodology

The golden evaluation set contains **200 hand-labelled examples** drawn from the 
evaluation holdout split (`data/processed/eval_holdout.jsonl`).

### Sampling Strategy
- **Stratified by conversation length**: Short (2-3 turns), medium (4-6 turns), long (7+ turns)
- **Diverse time range**: Sampled across the full date range to avoid temporal bias
- **Intent diversity**: After initial classification, re-sampled to ensure coverage 
  across all discovered intents (minimum 10 examples per major intent)
- **Edge cases included**: Deliberately included messages that are ambiguous, 
  multi-intent, or emotionally charged

### Labelling Guidelines

Each example was labelled with:
1. **`true_intent`**: The correct intent from the taxonomy (human judgment)
2. **`expected_escalation`**: Whether a human agent should handle this (`auto` or `escalate`)
3. **`quality_notes`**: Any special considerations for this example

### Labelling Rules
- When a message could belong to multiple intents, choose the **primary** intent 
  (the one the customer would most want addressed first)
- Messages expressing frustration are still classified by their **underlying issue**, 
  not by their emotional tone (tone affects escalation, not intent)
- Very short messages ("help", "please") → `other` intent, `escalate` decision
- Messages in foreign languages → `other` intent, `escalate` decision

### Self-Consistency Check
- 30 examples were independently re-labelled after a 24-hour gap
- Agreement rate tracked and reported in evaluation results

### Data Format

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | int | ID of the source conversation |
| `customer_message` | str | The customer's message text |
| `true_intent` | str | Human-labelled intent category |
| `actual_brand_reply` | str | What the brand actually replied |
| `expected_escalation` | str | `auto` or `escalate` |
| `quality_notes` | str | Edge case notes or special considerations |
