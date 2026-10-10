# Wren 1.0

Five-class (1 to 5 star) Amazon review sentiment model: a Kim 2014 style text CNN
written from scratch in NumPy. See the repository README for the full description.

| | |
|---|---|
| Name | Wren |
| Version | 1.0 |
| Architecture | fastText wiki-news-300d-1M embeddings (300d, frozen) -> filter banks (regions 3, 4, 5 x 100 filters, ReLU, global max-pool) -> dense 300-256-128-64-32-16-5 -> softmax |
| Parameters | 481,201 (360,300 filter banks, 120,901 dense) |
| Trained on | 98,000 balanced Amazon Reviews 2023 (Software) reviews, 15,000 batches of 32, SGD lr 0.03, best-checkpoint restore |
| Checkpoint written | 2026-09-09 19:03 |

## Measured performance

Fresh test set of 2,000 balanced reviews never seen in training or checkpoint
selection (`evaluation/evaluate.py`, `evaluation/results.json`):

- Exact accuracy **47.0%** (95% CI 44.8 to 49.2), chance is 20%
- Within one star **86.2%** (95% CI 84.6 to 87.6)

## Files

Produced by `tools/export_model.py` (run on 2026-10-10). Release assets, not committed.

| File | Bytes | sha256 |
|---|---|---|
| `wren-1.0-weights.npz` | 3,686,841 | `25702430d6b6bb2b079108784eb958771e46230bc5796e5a0f82333284726485` |
| `wren-1.0-embeddings.npz` | 27,357,796 | `100c4c929b9d758e3c69f5ebac2378da4a605dde8e5f33fa3e13a53699eb7f3e` |

`wren-1.0-weights.npz` holds 18 float64 arrays (`conv{1,2,3}_kernel/_bias`,
`dense{1..6}_weights/_biases`, 481,201 parameters), copied unchanged from the trained
checkpoint. `wren-1.0-embeddings.npz` holds `words` (37,498) and float32 `vectors`
(37,498 x 300).

Checked at export: the 2,000 test reviews give the same probabilities from these files
as from the original databases (max difference 4.9e-08) and the same 47.0% exact accuracy.

## Embeddings

The embeddings are **fastText `wiki-news-300d-1M`** (Mikolov, Grave, Bojanowski,
Puhrsch, Joulin, *Advances in Pre-Training Distributed Word Representations*, LREC 2018),
licensed **CC BY-SA 3.0**. `wren-1.0-embeddings.npz` is a subset of those vectors and is
distributed under the same licence.

The model was trained against a 949,658-word SQLite import of that file. That import is
an exact subset of the 999,994-word `.vec` (identical vectors on 3,000 sampled words); the
50,336 words left out were mostly numbers, which the cleaner strips anyway.

The shipped file contains only the words Wren can look up: every token the cleaner
produces from the 100,000 training reviews and the 2,000 test reviews that is in the
`.vec` (37,498 of 75,203). The other 37,705 tokens had no vector in training either
and are read as zeros (1.35% of word occurrences, mostly glued `<br />` leftovers such
as `itbr`). A word outside this set that is in the full `.vec` is also read as zeros,
whereas it would have had a real vector with the full table; the effect on unseen text
has not been measured.
