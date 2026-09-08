# Sentiment Analyser

A five-class sentiment analyser for Amazon reviews (1 to 5 stars), built from
scratch. The convolution layers, the dense layers, the backward pass and the
optimiser are all written directly against NumPy and SciPy — there is no PyTorch,
TensorFlow, Keras or scikit-learn anywhere in the model. That constraint is the
point of the project.

This started as an A-Level final project.

## Architecture

```
review text
  -> 300-dimensional word embeddings, padded or truncated to 200 words   (200 x 300)
  -> reflection padding, conv 5x5 stride 5, ReLU                          (40 x 60)
  -> reflection padding, conv 3x3 stride 3, ReLU                          (14 x 20)
  -> flatten                                                              (280)
  -> dense 280 -> 256 -> 128 -> 64 -> 32 -> 16, ReLU throughout
  -> dense 16 -> 5, softmax
  -> categorical cross entropy loss
```

Weights use He initialisation, which is the right variance for ReLU; biases start
at a small positive value so units begin on the active side of the clamp. The
optimiser is plain SGD on the batch mean loss.

Layer widths are not hardcoded in the model — `InitDatabases.py` measures the
convolution output by running a forward pass and sizes the dense stack to match.

## Setup

```bash
pip install -r requirements.txt        # numpy, scipy
```

The four SQLite databases are not in the repository and must be built.

### With the real data

You need two source files: the Amazon reviews dump (`Software.jsonl` or similar)
and a GloVe-style word embeddings file. Convert them first:

- `TestDataSetup.py` — trims the review dump to the fields and length range used,
  writing a `.jsonl` of `{"rating": ..., "text": ...}` records.
- `WordEmbeddingsSetup.py` — `write_as_jsonl` converts the whitespace-separated
  embeddings file to `{"word": ..., "vector": [...]}` records.

Both scripts still have hardcoded paths at the bottom; point them at your files.
Then build the databases:

```bash
python InitDatabases.py --reviews /path/to/TestData.jsonl \
                        --embeddings /path/to/WordEmbeddings.jsonl
```

### Without the real data

For running the tests, or to see the thing work end to end, a synthetic corpus
can be generated instead. Each star rating draws from its own word pool, so the
task is learnable and the classes are balanced:

```bash
python InitDatabases.py --synthetic 3000
```

This tells you nothing about real sentiment data. It exists so the verification
below can be run on a machine that does not have the Amazon dump.

## Running

```bash
python main.py
```

Press `1` to train a new model or `2` to classify a single sentence with the
currently loaded one.

Training reinitialises every weight and kernel, so it destroys whatever model was
loaded. It now asks for confirmation first and writes a timestamped backup of
`neuron_weights.db` and `convolution_layers.db` before touching them.

Training reports loss, accuracy and the class distribution of both the true labels
and the model's predictions, on the training data and separately on a held-out
validation set of up to 2,000 reviews that is never trained on. The split comes
from a seeded shuffle, so it is the same on every run.

The prediction distribution is printed because it is the figure that matters most
when accuracy is poor: Amazon review data skews heavily to 5 star, and a model
that has collapsed onto a single output can otherwise hide behind a plausible
looking accuracy number. If every prediction is the same class, the run says so
explicitly.

Accuracy within one star is reported alongside exact accuracy. Predicting 4 when
the truth is 5 is the normal failure mode for exact star prediction, and the two
figures together say much more than either alone.

## Verification

The gradient checks are the most valuable part of this repository. They are what
turns "I think the gradients are right" into "I proved the gradients are right".
Neither needs a database or any data.

```bash
python tests/test_gradients.py        # dense stack
python tests/test_conv_gradients.py   # convolution layers
```

Each compares every analytic gradient against a central finite difference of the
loss:

```
numerical = (loss(w + eps) - loss(w - eps)) / (2 * eps)
relative_error = |analytic - numerical| / max(|analytic|, |numerical|, 1e-8)
```

Anything above `1e-5` means the chain rule is broken somewhere.

| Check | Worst relative error |
|---|---|
| Dense parameter gradients (81 weights and biases) | 3.9e-08 |
| Dense input gradient | 2.8e-09 |
| Convolution kernel and input gradients (6 stride and kernel configurations) | 4.5e-08 |

The end-to-end counterpart needs the databases. A correctly wired network with no
regularisation must be able to memorise a small sample outright:

```bash
python tests/test_overfit.py                            # 200 reviews, 60 epochs
python tests/test_overfit.py --reviews 40 --epochs 40   # quicker
```

If this cannot reach near-100% training accuracy, there is still a bug and there
is no point starting a full training run. On the synthetic corpus it reaches 100%
by epoch 19, with the loss falling from 1.675 to 0.0011.

## Accuracy on real data

**Not yet measured.** The earlier version of this README said no meaningful
accuracy had been achieved, and that was true, but the cause was a broken backward
pass rather than a shortage of compute: the ReLU "derivative" returned the layer's
own forward activations, and the gradient was never passed between layers at all,
so the chain rule was absent from the network entirely. It could not have learned
regardless of how long it ran.

That is fixed and proved fixed by the checks above, but the Amazon review dump and
the embeddings file are not present in this checkout, so no honest figure for real
data can be quoted here. Building the databases from the real sources and running
a training pass is what fills this section in.

For calibration when you do: exact five-class star prediction is much harder than
binary positive/negative, and a from-scratch CNN in the 45-60% range is a
respectable result.

## What was wrong

For anyone reading the history, the defects that mattered, in order:

1. **The backward pass was disconnected.** `ReLU_derivative` took no upstream
   gradient and returned `self.output`, which is already `max(0, layer_output)`, so
   its masking line could never fire. `main.py` then discarded the return value of
   `calculate_derivatives` on every layer but the last. Every layer updated its
   weights from its own forward output. This alone put accuracy at chance.
2. **Softmax overflowed.** No max subtraction, so `np.exp` went to `inf` and the
   output to `nan` once weights grew; and every row was divided by the sum of row
   zero, so multi-row batches were not distributions at all.
3. **The batch axis was always one.** Gradients were accumulated one review at a
   time and `calculate_derivatives` averaged the batch axis away.
4. **The convolution gradients were wrong in three ways.** `convPass` emitted
   duplicated rows and columns at its trailing edges; the backward pass rotated the
   kernel although the forward pass is a cross-correlation; and it assumed output
   position `i` came from input position `i * stride`, which those duplicates broke.
   Relative errors were around 1.5, including sign flips.
5. **`reflectMatrix` mutated its argument**, so calling it on `convLayer1.output`
   rewrote layer 1's stored forward output with a padded copy, which is what hid
   defect 4.
6. **The convolution layers had no activation between them**, so they composed to a
   single linear map.
7. **The optimiser divided by the wrong number**, and the two variables controlling
   it were named the wrong way round.
8. **Data was drawn in strict id order** with no train/validation split, so the
   reported accuracy was training accuracy on data seen in the same pass.
