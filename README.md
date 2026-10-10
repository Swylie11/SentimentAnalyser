# Sentiment Analyser

A five-class sentiment analyser for Amazon reviews: give it a review and it predicts
1 to 5 stars. It is a text convolutional neural network (a Kim 2014 style CNN) built
from scratch. The convolution layers, the dense layers, the backward pass and the
optimiser are all written directly against NumPy and SciPy. There is no PyTorch,
TensorFlow, Keras or scikit-learn anywhere in the model. That constraint is the point
of the project.

This started as an A-Level computer science project. The first version did not work;
the history of why is at the bottom.

**On 2,000 reviews it had never seen, it gets the exact star rating 47.0% of the time
(95% CI 44.8 to 49.2%, chance is 20%) and is within one star 86.2% of the time.**
That is a modest result, not a state-of-the-art one, and the details below say exactly
what it is and what it is not.

## What the model is

A review goes in as text and comes out as a probability for each of the five ratings.

```
review text
  -> lowercase, strip punctuation and digits, split into words
  -> look each word up in fastText wiki-news 300-dimensional embeddings (frozen)
  -> pad with a tiny constant, or truncate, to 200 words              (200 x 300)
  -> three parallel filter banks over the word axis:
        region 3 words x 100 filters
        region 4 words x 100 filters
        region 5 words x 100 filters      each: correlate, + bias, ReLU,
                                                global max-pool over position
  -> concatenate the three pooled outputs                              (300)
  -> dense 300 -> 256 -> 128 -> 64 -> 32 -> 16, ReLU throughout
  -> dense 16 -> 5, softmax
  -> categorical cross entropy loss
```

**The filter banks.** Each filter is a small window of shape `(region, 300)`: it spans
the full embedding dimension and slides only along the word axis, so a region-3 filter
looks at three consecutive words at a time, like a learned phrase detector. Every
filter's response is passed through ReLU and then reduced to a single number by taking
its maximum over every word position ("did this pattern appear anywhere, and how
strongly?"). Three region sizes mean the model looks for 3, 4 and 5 word patterns in
parallel. Global max-pooling has two useful consequences: the feature vector is always
300 wide regardless of review length, and the padding rows, which are near-constant,
essentially never win the max, so no masking is needed.

**The dense stack.** Five ReLU layers narrow the 300 features down to 16 and a final
layer produces five logits, turned into probabilities by softmax. Weights use He
initialisation (the right variance for ReLU) and biases start at a small positive
value.

**Size and training.** About 481,000 trainable parameters: 360,300 in the filter banks
and 120,901 in the dense stack. The embeddings are not trained. The optimiser is plain
SGD (learning rate 0.03) on the batch mean loss, with no momentum, no dropout, no
weight decay and no learning rate schedule. During training the quick validation
score is checked every 200 batches and the best checkpoint is restored at the end.

**What it is not.** It is not a pretrained language model and it has no notion of word
order beyond 3 to 5 word windows, negation across a longer span, or sarcasm. A
word with no embedding is read as a zero vector (about 1% of the tokens in the
training and test reviews; mostly typos and glued tokens such as `itbr`).

## Results

### The trained model

Trained on a balanced corpus of 100,000 Amazon Software reviews (20,000 per star), of
which 2,000 were held out for validation and 98,000 trained on. 15,000 batches of 32, about 4.9 passes over the data, 70 minutes
on a CPU.

All figures below come from `evaluation/evaluate.py` on the same weights.

| Set | Reviews | Exact | Within one star |
|---|---|---|---|
| **Fresh test set** (never trained on, never used for selection) | 2,000 | **47.0%** (44.8 to 49.2) | **86.2%** (84.6 to 87.6) |
| Training slice (reviews it trained on) | 2,000 | 59.3% | 93.4% |
| Validation, first 300 (picked the checkpoint) | 300 | 55.7% | 91.0% |
| Validation, remaining | 1,700 | 52.1% | 89.5% |
| Uniform random guessing | | 20.0% | 52.0% |

Per-star recall on the fresh test set: 1 star 68.5%, 2 star 26.2%, 3 star 34.0%,
4 star 42.5%, 5 star 63.7%. The extreme ratings are much easier than the middle ones,
and 2 star reviews are mistaken for 1 star or 3 star more often than they are
correctly identified.

```
Confusion matrix, fresh test set (rows = true stars, columns = predicted)
    1* |   274    71    37    11     7
    2* |   137   105   122    28     8
    3* |    74    68   136    98    24
    4* |    13    25    90   170   102
    5* |    14    12    24    95   255
```

### How much to trust these numbers

- **The test set is clean.** `evaluation/build_test_set.py` draws new reviews from the
  full dump and discards any whose text is in the training database. It has 0 exact and
  4 near-duplicate (5-word-shingle Jaccard of 0.8 or more) matches with training.
- **The training log said 52.6%, and that was optimistic.** The best checkpoint was
  chosen using the first 300 validation reviews (a 3.6 point edge over the other 1,700,
  within noise but pointing the wrong way), and the validation set contains 21 exact
  duplicates of training reviews (about 1%). Validation (the other 1,700) and the fresh
  test set still differ by 5 points, which is statistically significant. The most likely
  cause is that validation shares products or users with the training reviews while the
  fresh test set mostly does not. The database stores only rating and text, so this
  cannot be checked. Quote the fresh test set figure.
- **There is a real generalisation gap.** The model scores 12.3 points higher on reviews
  it trained on than on fresh ones (z = 7.8). It is not memorising the training set, but
  with no regularisation it is overfitting moderately.
- **The training corpus has about 0.7% repeated texts**, since the dump contains
  duplicate reviews and the corpus was not deduplicated.

### How it got here

| Version | Fresh or held-out exact | Notes |
|---|---|---|
| Original (A-Level submission) | chance | The backward pass was disconnected; see below. |
| After the backprop fixes, 5,000 reviews, two stacked convolutions | 22.4% | No better than always guessing one class (21.1%). |
| Filter bank CNN, 98,000 reviews | **47.0%** | Today's model. |

Three things changed between the 22.4% and the 47.0%: the architecture (a single-channel
5x5 then 3x3 convolution compressed the 200 x 300 embedding matrix into 280 features,
which was the bottleneck, and most of every input was padding), the amount of data (3,000
training reviews to 98,000), and normalising curly apostrophes so words like `don’t`
reach the embedding table. Regularisation has not been tried yet.

## Install and use

```bash
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate
pip install .                      # numpy is the only dependency; use -e . to develop
```

The Wren 1.0 files (about 31 MB, see Pretrained weights below) are downloaded on first
use into `~/.cache/wren`, or `$WREN_HOME` if set. `wren fetch` downloads them up front.

```bash
wren predict "Does exactly what it says, works great"   # 5   Very positive
wren predict "waste of money" --proba             # stars, then P(1*) ... P(5*)
cat reviews.txt | wren predict - --json           # one review per line, JSON out
wren eval --data Data/eval_test.jsonl             # exact and within-one accuracy
```

```python
from wren import Analyser

wren = Analyser()
wren.predict("Does exactly what it says, works great")   # 5
wren.predict(["great", "awful"])                  # [5, 1]
wren.predict_proba("it is fine")                  # array of shape (1, 5)
wren.label(5)                                     # "Very positive"
```

Words the model has no vector for are read as zeros, and reviews are cut to 200 words.

## Training your own model

Training uses four SQLite databases in a work directory (`./wren-work`, or `--workdir`,
or `$WREN_WORKDIR`): `test_data.db` (the reviews), `word_embeddings.db`,
`neuron_weights.db` and `convolution_layers.db` (the parameters). They are not in the
repository and are built by `wren setup`.

### With the real data

Two source files are needed. Neither is in the repository.

**Reviews.** The Amazon Reviews 2023 dump from the McAuley Lab at UCSD. The Software
category is a 496 MB gzip of 4.88 million reviews:

```bash
curl -L -o Data/Software.jsonl.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Software.jsonl.gz
```

**Embeddings.** fastText `wiki-news-300d-1M.vec` (999,994 words, 300 dimensional), which
is what the shipped Wren 1.0 weights were trained with. The weights only work with
these exact vectors, so do not substitute GloVe or another fastText file:

```bash
curl -L -o Data/wiki-news-300d-1M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip -j Data/wiki-news-300d-1M.vec.zip -d Data/
```

Then build the two `.jsonl` files and the databases:

```bash
python -m wren.train.real_data_setup --reviews-gz Data/Software.jsonl.gz \
                                  --vec        Data/wiki-news-300d-1M.vec \
                                  --per-class  20000 \
                                  --out-dir    Data
wren setup --reviews Data/TestData.jsonl --embeddings Data/WordEmbeddings.jsonl
```

`real_data_setup.py` is built to give an honest figure:

- **Classes are balanced.** Raw Amazon ratings are very skewed: in the Software
  category, 1.24 million reviews in the length window are 5 star against 155 thousand
  that are 2 star. Sampling in file order gives a corpus where a model can score well
  by learning the prior. Each rating is sampled to the same count.
- **The sample is drawn from the whole file** by reservoir sampling, not taken from the
  front, so it is not confined to whichever products appear first.
- **JSON is written with `json.dumps`.** The older scripts built lines with an f-string,
  so any review containing a double quote produced an invalid line that was then skipped
  silently. That was 1.9% of reviews, and not a random 1.9%.

Reviews are kept if they are 15 to 200 words long. The embedding table is restricted to
the vocabulary the corpus uses (importing all 999,994 fastText vectors would build a
multi-gigabyte table). Words with no vector fall back to a zero vector.

### Without the real data

For running the tests, or to see the thing work end to end, a synthetic corpus can be
generated instead. Each star rating draws from its own word pool, so the task is
learnable and the classes are balanced:

```bash
wren setup --synthetic 3000
```

This tells you nothing about real sentiment. It exists so the tests can run on a
machine without the Amazon dump.

### Training

```bash
wren train --batches 15000 --batch-size 32       # add --yes to skip the confirmation
```

Training reinitialises every weight and kernel, so it destroys whatever model was
loaded in the work directory. It asks for confirmation first and writes a timestamped
backup of `neuron_weights.db` and `convolution_layers.db` before touching them. A
trained model lives in those databases; to use it with `Analyser` it has to be exported
with `tools/export_model.py` (see Pretrained weights).

Training reports loss, accuracy and the class distribution of the true labels and of
the model's predictions, on the training data and on a held-out validation set of up to
2,000 reviews that is never trained on. The split comes from a seeded shuffle, so it is
the same on every run. The prediction distribution is printed because a model that has
collapsed onto one output can otherwise hide behind a plausible-looking accuracy. If
every prediction is the same class, the run says so.

Accuracy within one star is reported alongside exact accuracy. Predicting 4 when the
truth is 5 is the normal failure mode for exact star prediction, and the two figures
together say much more than either alone.

## Pretrained weights

Wren 1.0 is exported by `tools/export_model.py` as two portable files in
`models/wren-1.0/` (they are release assets, not committed; sizes and checksums are in
`models/wren-1.0/MODEL.md`):

- `wren-1.0-weights.npz`: the filter bank kernels and dense layers.
- `wren-1.0-embeddings.npz`: the fastText vectors for the 37,498 words Wren can look up.

```bash
python tools/export_model.py --sentiment-dir <work dir with the trained .db files>
                                 # also needs Data/wiki-news-300d-1M.vec
```

The script runs the 2,000 test reviews through both the databases and the exported
files and stops unless the probabilities agree.

## Licences and credits

- The code in this repository is MIT licensed (see `LICENSE`).
- The word vectors are fastText `wiki-news-300d-1M` from Mikolov, Grave, Bojanowski,
  Puhrsch and Joulin, *Advances in Pre-Training Distributed Word Representations*
  (LREC 2018), licensed CC BY-SA 3.0. `wren-1.0-embeddings.npz` is a subset of those
  vectors and is distributed under the same licence.
- The training reviews are from Amazon Reviews 2023 (McAuley Lab, UCSD), published for
  research use. Check its terms before using the weights commercially.

## Verification

The gradient checks are the most valuable part of this repository. They turn "I think
the gradients are right" into "I proved the gradients are right". Neither needs a
database or any data.

```bash
python tests/test_gradients.py              # dense stack
python tests/test_filterbank_gradients.py   # filter bank layers
python tests/test_predict.py                # Analyser on the shipped Wren 1.0 files
```

Each compares every analytic gradient against a central finite difference of the loss:

```
numerical = (loss(w + eps) - loss(w - eps)) / (2 * eps)
relative_error = |analytic - numerical| / max(|analytic|, |numerical|, 1e-8)
```

Anything above `1e-5` means the chain rule is broken somewhere.

| Check | Worst relative error |
|---|---|
| Dense parameter gradients (81 weights and biases) | 3.9e-08 |
| Dense input gradient | 2.8e-09 |
| Filter bank kernel, bias and input gradients (4 region and filter configurations) | 2.4e-06 |

The end-to-end counterpart needs the databases. A correctly wired network with no
regularisation must be able to memorise a small sample outright:

```bash
wren setup --synthetic 400                              # first, in a scratch work directory
python tests/test_overfit.py                            # 200 reviews, 60 epochs
python tests/test_overfit.py --reviews 40 --epochs 40   # quicker
```

If this cannot reach near-100% training accuracy there is still a bug. A failure at the
default epoch budget is not by itself evidence of a broken graph; check whether the
loss is still falling before believing it.

### Evaluation

`evaluation/` measures the saved model honestly. See `evaluation/README.md` for how to
read the report.

```bash
python evaluation/build_test_set.py --reviews-gz Data/Software.jsonl.gz
python evaluation/evaluate.py
python evaluation/tests/test_evaluation.py     # quick checks, no real data
```

## Repository layout

```
src/wren/      the package: Analyser (model.py), the CLI (cli.py), text cleaning,
              layers/, and train/ (training loop, database setup, data builder)
tools/        export_model.py: databases -> the portable Wren 1.0 files
docs/         PLAN.md, the packaging plan
tests/        gradient checks, the overfit test, and prediction checks
evaluation/   fresh test set builder, evaluation script, logs of the training runs
```

## Next steps

Roughly in the order they are likely to help:

1. **Regularisation.** Dropout or weight decay, and stopping on a validation set that is
   not also used to choose the final number. The 12 point train/test gap says this is
   the first thing to try.
2. **Deduplicate the training corpus** and split by product or user rather than by
   review, so validation cannot share products with training.
3. **More filters or more region sizes**, and train the embeddings.
4. **Better handling of the middle ratings**, where recall is 26 to 42%.
5. **A faster training loop**, so each of these experiments is not a 70 minute wait.

## What was wrong originally

For anyone reading the history, the defects that mattered, in order:

1. **The backward pass was disconnected.** `ReLU_derivative` took no upstream gradient
   and returned `self.output`, which is already `max(0, layer_output)`, so its masking
   line could never fire. `main.py` then discarded the return value of
   `calculate_derivatives` on every layer but the last. Every layer updated its weights
   from its own forward output. This alone put accuracy at chance.
2. **Softmax overflowed.** No max subtraction, so `np.exp` went to `inf` and the output
   to `nan` once weights grew; and every row was divided by the sum of row zero, so
   multi-row batches were not distributions at all.
3. **The batch axis was always one.** Gradients were accumulated one review at a time
   and `calculate_derivatives` averaged the batch axis away.
4. **The convolution gradients were wrong in three ways.** The forward pass emitted
   duplicated rows and columns at its trailing edges; the backward pass rotated the
   kernel although the forward pass is a cross-correlation; and it assumed output
   position `i` came from input position `i * stride`, which those duplicates broke.
5. **`reflectMatrix` mutated its argument**, rewriting layer 1's stored forward output
   with a padded copy, which hid defect 4.
6. **The convolution layers had no activation between them**, so they composed to a
   single linear map.
7. **The optimiser divided by the wrong number**, and the two variables controlling it
   were named the wrong way round.
8. **Data was drawn in strict id order** with no train/validation split, so the
   reported accuracy was training accuracy on data seen in the same pass.

Fixing these produced a model that could learn but did not learn much (22.4%). Replacing
the stacked single-channel convolutions with the filter bank and training on 98,000
reviews took it to 47.0%.
