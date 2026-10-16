# Wren 1.5

Five-class (1 to 5 star) Amazon review sentiment model: a Kim 2014 style text CNN
written from scratch in NumPy. See the repository README for the full description.

| | |
|---|---|
| Name | Wren |
| Version | 1.5 |
| Architecture | fastText wiki-news-300d-1M embeddings (300d, frozen) -> filter banks (regions 3, 4, 5 x 100 filters, ReLU, global max-pool, dropout 0.5) -> dense 300-256 (ReLU, dropout 0.2) -> 5 -> softmax |
| Parameters | 438,641 (360,300 filter banks, 78,341 dense) |
| Trained on | 575,500 Amazon Reviews 2023 reviews (577,500 balanced across stars, 115,500 each, less 2,000 held out for validation) from five categories: Software, Video Games, Musical Instruments, Office Products, Toys and Games |
| Training | 45,000 batches of 32, AdamW (lr 1e-3, cosine to 0, weight decay 0.01), two seeds, the better on 2,000 held-out validation reviews kept (seed 0) |
| Checkpoint written | 2026-10-15 03:34 |

## What changed from 1.0

- Five product categories instead of Software only, and about 5.9x more reviews
- AdamW with a cosine schedule instead of plain SGD
- One hidden layer (256) instead of five (256-128-64-32-16)
- Dropout on the pooled filter features (0.5) and the hidden layer (0.2); 1.0 had none

## Measured performance

Fresh test set `Data/eval_test_multi.jsonl`: 2,000 reviews balanced across stars and
drawn from the same five categories, none in either model's training data. Both models
on the same reviews (`evaluation/mcnemar.py`, `evaluation/mcnemar.json`):

| | Wren 1.0 | Wren 1.5 |
|---|---|---|
| Exact star | 49.2% | **52.5%** (95% CI 50.3 to 54.7) |
| Within one star | 87.9% | **90.9%** (95% CI 89.6 to 92.1) |
| Recall 1★ / 2★ / 3★ / 4★ / 5★ | 68 / 28 / 43 / 47 / 61% | 62 / 47 / 44 / 38 / 73% |

On the 441 reviews where exactly one model is right, Wren 1.5 wins 253 to 188: McNemar
exact test p = 0.0023. Chance is 20% exact.

Wren 1.5 is worse than 1.0 on 1★ and 4★ reviews, and better on 2★ and 5★.
It has not been scored on 1.0's Software-only test set, because that set was only
de-duplicated against 1.0's training data and may overlap 1.5's.

## Files

Produced by `tools/export_wren15.py` (run on 2026-10-16). Release assets, not committed.

| File | Bytes | sha256 |
|---|---|---|
| `wren-1.5-weights.npz` | 3,357,219 | `cd0b7addc30cf408880d24bb8bfcf81d81fa97d69f741cbd2334584c72cd8dcd` |
| `wren-1.5-embeddings.npz` | 47,616,985 | `f8fa4cacbe4a1dcb9450161ce44dfca1778799027ff8f602c2720593ef3a6e84` |

`wren-1.5-weights.npz` holds 10 float64 arrays (`conv{1,2,3}_kernel/_bias`,
`dense{1,2}_weights/_biases`), copied unchanged from the trained checkpoint.
`wren-1.5-embeddings.npz` holds `words` (65,726) and float32 `vectors` (65,726 x 300),
copied from the training run's embedding table: the words of the training corpus that
are in the fastText file. Any other word is read as zeros, as in training.

Checked at export: all 2,000 test reviews get the same prediction from these files as
from the original databases (52.5% exact).

## Embeddings

The embeddings are **fastText `wiki-news-300d-1M`** (Mikolov, Grave, Bojanowski,
Puhrsch, Joulin, *Advances in Pre-Training Distributed Word Representations*, LREC 2018),
licensed **CC BY-SA 3.0**. `wren-1.5-embeddings.npz` is a subset of those vectors and is
distributed under the same licence.
