# Evaluation dataset location

Place your ground-truth evaluation dataset here as `ground_truth.jsonl`.

Use one JSON object per line. Required fields:

```json
{"user_input":"What does the paper conclude?","reference":"The expected correct answer."}
```

For retrieval-recall evaluation, add the supporting passages:

```json
{"user_input":"What does the paper conclude?","reference":"The expected correct answer.","reference_contexts":["A passage from the source that supports the answer."]}
```

Do not include API keys or confidential data that should not be sent to the evaluator.
