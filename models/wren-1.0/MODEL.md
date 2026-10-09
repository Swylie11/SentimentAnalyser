# Wren 1.0

Five-class (1 to 5 star) Amazon review sentiment model: a Kim 2014 style text CNN
written from scratch in NumPy. See the repository README for the full description.

| | |
|---|---|
| Name | Wren |
| Version | 1.0 |
| Architecture | GloVe-style 300d embeddings (frozen) -> filter banks (regions 3, 4, 5 x 100 filters, ReLU, global max-pool) -> dense 300-256-128-64-32-16-5 -> softmax |
| Parameters | 481,201 (360,300 filter banks, 120,901 dense) |
| Trained on | 98,000 balanced Amazon Reviews 2023 (Software) reviews, 15,000 batches of 32, SGD lr 0.03, best-checkpoint restore |
| Checkpoint written | 2026-09-09 19:03 |

## Measured performance

Fresh test set of 2,000 balanced reviews never seen in training or checkpoint
selection (`evaluation/evaluate.py`, `evaluation/results.json`):

- Exact accuracy **47.0%** (95% CI 44.8 to 49.2), chance is 20%
- Within one star **86.2%** (95% CI 84.6 to 87.6)

## Files

| File | Bytes | sha256 |
|---|---|---|
| `neuron_weights.db` | 4,325,376 | `0b5f715cc00b68c2308786d7e34c9f8b39939f4085764becfa40e8eb2f9d191d` |
| `convolution_layers.db` | 11,292,672 | `f784ac7356c48a883ffd5b19fcb101bb2b7202bf4aab61ab6858fe6c9f14560d` |

Byte-identical to the `sentiment/` copies that produced the measured results above.

## Not included: the embedding table

The weights only work with the exact embedding table they were trained against.
That table is `word_embeddings.db` (3.9 GB, 949,658 words, last modified 2025-02-03),
sha256 `eec09aec1fa3928cd1189820709bd7a70b29ef7c0c6989db25193fe55bb2ab4c`.

**Provenance is unresolved.** The repository README says the embeddings are GloVe 6B
300d, but this table has 949,658 entries (GloVe 6B has 400,000) and its vector for
"the" does not match GloVe 6B 300d. Rebuilding the table from `glove.6B.300d.txt` would
produce different vectors and these weights would not work with it. Before release,
either identify the source of this table or ship a version restricted to the
vocabulary Wren needs.
