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

Two source files are needed. Neither is in the repository.

**Reviews.** The Amazon Reviews 2023 dump from the McAuley Lab at UCSD, which is
the current version of the data this project was originally built against. The
Software category is a 473 MB gzip of 4.88 million reviews:

```bash
curl -L -o Data/Software.jsonl.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Software.jsonl.gz
```

**Embeddings.** GloVe 6B, 300 dimensional, to match `EMBEDDING_DIM`:

```bash
curl -L -o Data/glove.6B.zip https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
unzip -j Data/glove.6B.zip glove.6B.300d.txt -d Data/
```

Then build the two `.jsonl` files and the databases:

```bash
python RealDataSetup.py --reviews-gz Data/Software.jsonl.gz \
                        --glove      Data/glove.6B.300d.txt \
                        --per-class  1000 \
                        --out-dir    Data
python InitDatabases.py --reviews Data/TestData.jsonl --embeddings Data/WordEmbeddings.jsonl
```

`RealDataSetup.py` replaces the hardcoded-path scripts (`TestDataSetup.py` and
`WordEmbeddingsSetup.py`) for this route. It differs from them in three ways that
matter for getting an honest figure:

- **Classes are balanced.** Raw Amazon ratings are not close to uniform: in the
  Software category, 1.24 million reviews in the length window are 5 star against
  155 thousand that are 2 star, an eight to one spread. Sampling the file in order
  gives a corpus where a model can score well by learning the prior instead of
  reading the review. Each rating is sampled to the same count.
- **The sample is drawn from the whole file** by reservoir sampling, not taken
  from the front, so it is not confined to whichever products appear first.
- **JSON is written with `json.dumps`.** `TestDataSetup.py` built its lines with
  an f-string, so any review containing a double quote produced an invalid line
  that `InitDatabases.py` then skipped without saying so. That is 1.9% of reviews,
  and reviews containing quotes are not a random 1.9%.

The embedding table is restricted to the vocabulary the corpus actually uses.
Importing all 400,000 GloVe vectors would build a multi-gigabyte table the model
would never mostly read. Words with no GloVe vector keep the existing behaviour
and fall back to a zero vector.

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

The default 40 epoch budget is calibrated on the synthetic corpus, where each
rating draws from a disjoint word pool. Real reviews are much harder to memorise:
on the Amazon corpus 40 epochs reaches only 85% and reports a failure, while 150
epochs reaches 100% by epoch 76. A failure at the default budget is not by itself
evidence of a broken graph — check whether the loss is still falling before
believing it.

### End-to-end run on the synthetic corpus

A full training pass on 3,000 synthetic reviews (1,500 trained on, 1,500 held
out), 600 batches of 32, about 140 seconds:

```
Validation (1500 held out reviews, never trained on)
  Accuracy         : 97.07%
  Within one star  : 98.47%
  True labels      : 1*: 20.4%  2*: 18.7%  3*: 20.8%  4*: 19.5%  5*: 20.6%
  Predictions      : 1*: 20.7%  2*: 19.1%  3*: 20.6%  4*: 19.3%  5*: 20.2%
```

**This is a synthetic dataset and the figure is not a sentiment analysis result.**
Each rating draws from its own disjoint word pool, so the task is far easier than
real reviews. What it does demonstrate is that the network trains, generalises to
data it has not seen, and spreads its predictions across all five classes instead
of collapsing onto one. It is a test of the machinery, not of the model's ability
to read sentiment.

## Accuracy on real data

**Measured, and the honest answer is that the model barely learns.**

The corpus is 5,000 Amazon Software reviews, 1,000 per star rating, sampled from
the 2023 dump as described above. 3,000 are trained on and 2,000 are held out.
3,000 batches of 32, learning rate 0.03, about 12 minutes.

```
Validation (2000 held out reviews, never trained on)
  Accuracy         : 22.40%
  Within one star  : 58.35%
  True labels      : 1*: 18.9%  2*: 19.5%  3*: 21.1%  4*: 20.2%  5*: 20.2%
  Predictions      : 1*:  2.6%  2*:  2.9%  3*:  8.8%  4*: 61.9%  5*: 23.9%
```

That figure only means something next to the trivial baselines on the same split,
and this is the part that matters:

| Predictor | Exact | Within one star |
|---|---|---|
| Uniform random | 20.00% | — |
| Always predict 3* | 21.10% | 60.85% |
| Always predict 4* | 20.25% | 61.55% |
| **This model** | **22.40%** | **58.35%** |

The network beats the best constant predictor by 2.15 points on exact accuracy,
and is 3.20 points *worse* than it on accuracy within one star. Its per-class
recall shows why: 4.2% on 1 star, 2.6% on 2 star, 63.7% on 4 star. It puts 61.9%
of all its predictions in the 4 star bucket. It has not collapsed onto a single
class outright, but it is much closer to guessing the middle of the range than to
reading sentiment.

The gap between training and validation is the other half of the picture:

| Split | Exact | Within one star |
|---|---|---|
| Training (3,000 seen reviews) | 51.73% | 70.73% |
| Validation (2,000 held out) | 22.40% | 58.35% |

51.73% down to 22.40% is not a model that is learning slowly. It is a model that
is memorising 3,000 reviews and carrying almost none of it across. The gradient
checks and the overfit test both pass, so the machinery is right; what is wrong is
the setup around it.

### Why it is this low, in the order worth attacking

1. **3,000 training reviews is far too few** for five-way star prediction. This
   run was sized to finish in a session, not to produce the best number. This is
   the first thing to change.
2. **The convolution stack is one 5x5 kernel followed by one 3x3 kernel.** A
   200x300 embedding matrix is compressed to 280 features through a single
   channel at each stage. Text CNNs normally use a hundred or more filters per
   layer. This is the architectural bottleneck, and no amount of data fixes it.
3. **There is no regularisation at all** — no dropout, no weight decay, no early
   stopping. Given the train/validation gap above, that is exactly what the
   numbers say is missing.
4. **Most of every input is padding.** The median review here is 29 words, padded
   to 200, so roughly 85% of the matrix the convolutions see is 0.001 filler.
5. **Curly apostrophes are not stripped.** `format_entry_data` removes ASCII
   `string.punctuation` but not U+2019, so `it’s`, `don’t` and `i’m` miss GloVe
   and become zero vectors. They are the largest group of out-of-vocabulary
   tokens. Token level coverage is 98.9%, so this is a small effect, but it is a
   free fix.

For calibration: exact five-class star prediction is much harder than binary
positive/negative, and a from-scratch CNN in the 45-60% range is a respectable
result. This model is not there, and the honest summary is that a correct
backward pass was necessary to have any chance of learning but was not on its own
sufficient to learn.

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
