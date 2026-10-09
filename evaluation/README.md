# Evaluation

The 52.6% validation figure from the training log is soft: `main.py` picked its best
checkpoint using the first 300 validation reviews, and no review was held out from
that choice. These scripts measure the saved model on reviews it has never seen.

Neither script modifies a database, and `evaluate.py` verifies that.

## Run

From the repo root, with the virtual environment active:

```bash
# 1. Fresh test set: 2,000 balanced reviews, none of them in sentiment/test_data.db
python evaluation/build_test_set.py --reviews-gz Data/Software.jsonl.gz

# 2. Evaluate the weights currently in sentiment/
python evaluation/evaluate.py
```

Outputs: `Data/eval_test.jsonl` and `.manifest.json` (how many candidates were dropped
as already-trained-on), and `evaluation/results.json`. Run the checks first with
`python evaluation/tests/test_evaluation.py` (seconds, no real data).

## Reading the report

| Section | What it tells you |
|---|---|
| **Fresh test set** | The headline number. Quote this one, with its 95% CI. |
| **Training slice** | Accuracy on reviews the model trained on. Test minus this is the generalisation gap. |
| **Validation, first 300 vs rest** | The first 300 chose the checkpoint. If they score well above the rest, the old 52.6% was inflated by selection. |
| **Gaps** | z-statistics for both differences. `|z| < 1.96` means the difference is within noise. |
| **Leakage** | Exact and near-duplicate (5-word shingle Jaccard >= 0.8) overlap with the training reviews. Test should be 0 exact by construction. Validation may not be. |
| **Baselines** | Uniform random and best constant predictor on the test set. The model has to beat these. |
| **Words with no embedding** | The embedding table only holds the training corpus's words, so a fresh test set has more unknown words. This is a real handicap, not a bug in the evaluation. |

A good result is a test accuracy within a few points of the validation figure, a
train-minus-test gap that is not huge, and zero or negligible leakage. If the test
score is far below validation, the validation figure was optimistic.

`results.json` records the sha256 and modification time of the weight files so a
result can be tied to the checkpoint that produced it.
