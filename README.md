# Sentiment Analyser (Wren)

**Give it a sentence, and it rates the sentiment from 1 to 5.**

Wren is a text *convolutional neural network* (CNN) built entirely from scratch. The
convolution layers, the dense layers, the backward pass (how the network works out
how to improve) and the optimiser (how it applies those improvements) are all written
by hand using NumPy. There is no PyTorch, TensorFlow, Keras or scikit-learn anywhere
in the model. **That constraint is the whole point of the project.**

It started life as an A-Level Computer Science project. The first version did not
work, and [the story of why](#what-was-wrong-originally) is at the bottom.

## Headline result

Tested on 2,000 reviews it had never seen before:

| | Wren | Random guessing |
|---|---|---|
| **Exact star rating** | **47.0%** (95% CI: 44.8–49.2%) | 20.0% |
| **Within one star** | **86.2%** | 52.0% |

> **What's a 95% CI?** A *confidence interval*. Because the test set is a sample, the
> true accuracy could be a little higher or lower than 47.0%. The interval says we
> can be 95% confident that the true figure lies between 44.8% and 49.2%.

This is a **modest result, not a state-of-the-art one**. The sections below say
exactly what it is and what it is not.

---

## Contents

- [Quick start](#quick-start)
- [How the model works](#how-the-model-works)
- [Results](#results)
- [Training your own model](#training-your-own-model)
- [Pretrained weights](#pretrained-weights)
- [Verification: proving the maths is right](#verification-proving-the-maths-is-right)
- [Repository layout](#repository-layout)
- [Next steps](#next-steps)
- [What was wrong originally](#what-was-wrong-originally)
- [Licences and credits](#licences-and-credits)

---

## Quick start

### Install

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
source .venv/bin/activate          # macOS / Linux
pip install .                      # NumPy is the only dependency (use -e . to develop)
```

The pretrained **Wren 1.0** files (about 31 MB, see [Pretrained weights](#pretrained-weights))
are downloaded automatically the first time you use the model. They are saved to
`~/.cache/wren`, or to `$WREN_HOME` if you set it. To download them in advance, run
`wren fetch`.

### Use it from the command line

```bash
wren predict "Does exactly what it says, works great"   # 5   Very positive
wren predict "waste of money" --proba                    # the star, then P(1★) … P(5★)
cat reviews.txt | wren predict - --json                  # one review per line, JSON output
wren eval --data Data/eval_test.jsonl                    # exact and within-one accuracy
```

### Use it from Python

```python
from wren import Analyser

wren = Analyser()
wren.predict("Does exactly what it says, works great")   # 5
wren.predict(["great", "awful"])                         # [5, 1]
wren.predict_proba("it is fine")                         # array of shape (1, 5)
wren.label(5)                                            # "Very positive"
```

The five labels are: 1 = Very negative, 2 = Negative, 3 = Neutral, 4 = Positive,
5 = Very positive.

**Two things to know:** reviews are cut off after 200 words, and any word the model
has no vector for is treated as a row of zeros (see below).

---

## How the model works

A review goes in as text. What comes out is a probability for each of the five star
ratings, and the most likely one is the prediction.

### The pipeline at a glance

```
review text
  → lowercase, strip punctuation and digits, split into words
  → look up each word's 300-number fastText vector (frozen)
  → pad with a tiny constant, or truncate, to exactly 200 words      (200 × 300)
  → three parallel filter banks that slide along the words:
        3-word windows × 100 filters
        4-word windows × 100 filters
        5-word windows × 100 filters      each: correlate, + bias, ReLU,
                                                global max-pool over position
  → join the three pooled outputs together                            (300)
  → dense layers 300 → 256 → 128 → 64 → 32 → 16, ReLU throughout
  → dense layer 16 → 5, softmax
  → categorical cross-entropy loss (training only)
```

Each step is explained below.

### Step 1: Turning words into numbers (word embeddings)

A neural network can only work with numbers, so each word is swapped for a
**word embedding**: a list of 300 numbers that captures something about its meaning.
Wren uses the pre-made fastText *wiki-news* embeddings. In these, words with similar
meanings (like "great" and "excellent") end up with similar lists of numbers.

The embeddings are **frozen**, meaning Wren uses them as they are and never adjusts
them during training.

Every review is then made exactly 200 words long. Short reviews are padded with a
tiny constant value, and long ones are truncated. This turns each review into a grid
of 200 rows (words) × 300 columns (embedding numbers).

### Step 2: Spotting phrases (the filter banks)

This is the "convolutional" part of the network.

A **filter** is a small window that slides down the review one word at a time. A
3-word filter has the shape `(3, 300)`, so it always sees three consecutive words in
full. At each position it produces a score for how well those words match the
pattern it has learned to look for. You can think of each filter as a **learned
phrase detector**. One might fire strongly on "waste of money", another on "works
perfectly fine".

For each filter, Wren then:

1. **Adds a bias and applies ReLU.** ReLU (*rectified linear unit*) simply replaces
   any negative score with zero. This "non-linearity" is what lets a network learn
   more than straight-line relationships.
2. **Takes the maximum score across the whole review** (*global max-pooling*). This
   boils the filter's output down to one number that answers: *"Did this pattern
   appear anywhere in the review, and how strongly?"*

There are 100 filters for each of three window sizes (3, 4 and 5 words), all running
in parallel. That gives 300 numbers in total.

Global max-pooling has two handy side effects:

- The output is **always 300 numbers wide**, however long the review is.
- The padding rows are almost constant, so they **essentially never produce the
  maximum**. That means the model doesn't need any extra logic ("masking") to ignore
  them.

### Step 3: Making a decision (the dense layers)

The 300 phrase-detector scores are passed through a stack of **dense layers**
(also called fully connected layers), in which every input is connected to every
output. Five ReLU layers narrow the information down step by step
(300 → 256 → 128 → 64 → 32 → 16). A final layer then produces five raw scores,
called **logits**, one per star rating.

**Softmax** turns those five logits into five probabilities that add up to 100%.

A few setup details:

- Weights start with **He initialisation**, which picks random starting values with
  the right spread for ReLU layers so signals neither die out nor blow up.
- Biases start at a small positive value.

### Step 4: Learning (training)

During training, the model's predicted probabilities are compared with the true
rating using **categorical cross-entropy loss**. This number is small when the model
gives high probability to the right answer and large when it doesn't.

**Backpropagation** then works out, for every one of the model's parameters, which
direction to nudge it to reduce the loss. The optimiser is plain **stochastic
gradient descent (SGD)**: it takes a small step (learning rate 0.03) in that direction,
using the average loss over a batch of reviews.

It is deliberately bare-bones. There is:

- no momentum
- no dropout
- no weight decay
- no learning-rate schedule

The model is checked against a quick validation score every 200 batches, and the
best-scoring version (*checkpoint*) is restored at the end of training.

### Model size

| Part | Trainable parameters |
|---|---|
| Filter banks | 360,300 |
| Dense layers | 120,901 |
| **Total** | **about 481,000** |
| Word embeddings | not trained (frozen) |

### What the model *cannot* do

- It is **not a pretrained language model** (like BERT or GPT). It learns only from
  the reviews it is trained on.
- It only understands word order **within 3- to 5-word windows**. It can't follow
  negation over a longer span (e.g. *"I expected to hate it, but… no"*), and it has
  no notion of sarcasm.
- A word with no embedding is read as a **row of zeros**, which carries no meaning.
  This affects about 1% of the words in the training and test reviews, mostly typos
  and words glued together such as `itbr`.

---

## Results

### Training setup

| | |
|---|---|
| **Data** | 100,000 Amazon *Software* reviews, balanced at 20,000 per star rating |
| **Split** | 98,000 for training, 2,000 held back for validation |
| **Training length** | 15,000 batches of 32 reviews, about 4.9 passes over the data |
| **Time** | 70 minutes on a CPU |

All figures below come from `evaluation/evaluate.py`, run on the same trained weights.

### Accuracy

| Data set | Reviews | Exact | Within one star |
|---|---|---|---|
| **Fresh test set** (never trained on, never used to pick the model) | 2,000 | **47.0%** (44.8–49.2) | **86.2%** (84.6–87.6) |
| Training slice (reviews it *did* train on) | 2,000 | 59.3% | 93.4% |
| Validation, first 300 (used to pick the checkpoint) | 300 | 55.7% | 91.0% |
| Validation, remaining | 1,700 | 52.1% | 89.5% |
| Uniform random guessing | | 20.0% | 52.0% |

**Use the fresh test set figure (47.0%) when quoting this model.** The section on
[how much to trust these numbers](#how-much-to-trust-these-numbers) explains why.

### Which ratings are easy, and which are hard

**Recall** is the share of reviews with a given true rating that the model got right.

| True rating | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---|---|---|---|---|
| Recall (fresh test set) | 68.5% | 26.2% | 34.0% | 42.5% | 63.7% |

The extreme ratings are much easier than the middle ones. Strong love or hate
uses distinctive language, while 2★, 3★ and 4★ reviews often mix praise and
complaint. In fact, 2★ reviews are mistaken for 1★ or 3★ more often than they are
correctly identified.

#### Confusion matrix

A **confusion matrix** shows every prediction the model made. Each row is a true
rating, and each column is what the model predicted. The diagonal (top-left to
bottom-right) holds the correct answers. Everything else is a mistake, and mistakes
close to the diagonal are near misses.

```
Fresh test set (rows = true stars, columns = predicted stars)

            pred 1★  pred 2★  pred 3★  pred 4★  pred 5★
  true 1★ |   274       71       37       11        7
  true 2★ |   137      105      122       28        8
  true 3★ |    74       68      136       98       24
  true 4★ |    13       25       90      170      102
  true 5★ |    14       12       24       95      255
```

Reading the 2★ row, for example: of 400 two-star reviews, only 105 were predicted
correctly. 137 were called 1★ and 122 were called 3★.

### How much to trust these numbers

**✅ The test set is clean.**
`evaluation/build_test_set.py` draws new reviews from the full dataset and throws away
any whose text appears in the training data. The result has 0 exact matches and only
4 near-duplicates with the training data. (A near-duplicate is a pair of reviews whose
5-word chunks overlap by 80% or more, measured by *5-word-shingle Jaccard similarity*
≥ 0.8.)

**⚠️ The training log said 52.6%, and that was optimistic.** There are three reasons:

1. **The checkpoint was chosen using the first 300 validation reviews.** Picking the
   best model on a set flatters that set. Here those 300 scored 3.6 points higher
   than the other 1,700. That gap is within noise, but it points the wrong way.
2. **The validation set contains 21 exact copies of training reviews** (about 1%).
3. **There is a significant 5-point gap** between the other 1,700 validation reviews
   and the fresh test set. The most likely cause is that validation reviews share
   products or reviewers with the training reviews, while the fresh test set mostly
   doesn't. That can't be checked, because the database stores only the rating and
   the text.

**⚠️ There is a real generalisation gap.** The model scores 12.3 points higher on
reviews it trained on than on fresh ones. That difference is far too large to be
chance (a *z-score* of 7.8; anything above about 2 is usually treated as significant).
The model isn't simply memorising the training set, but it is moderately
**overfitting**: it has learned some quirks of its training data that don't carry
over to new reviews. That's to be expected with no regularisation (techniques like
dropout or weight decay that discourage overfitting).

**ℹ️ About 0.7% of the training corpus is repeated text.** The raw dataset contains
duplicate reviews, and the corpus was not deduplicated.

### How it got here

| Version | Exact accuracy (fresh or held-out data) | Notes |
|---|---|---|
| Original A-Level submission | chance level | The backward pass was disconnected ([see below](#what-was-wrong-originally)). |
| After the backpropagation fixes: 5,000 reviews, two stacked convolutions | 22.4% | No better than always guessing the same rating (21.1%). |
| Filter bank CNN, 98,000 reviews | **47.0%** | Today's model. |

Three things changed between 22.4% and 47.0%:

1. **The architecture.** The old design ran a single 5×5 convolution followed by a
   3×3 one. That squashed the 200 × 300 embedding grid down to just 280 numbers,
   which was the bottleneck, and most of every input was padding anyway.
2. **The amount of data.** The training data grew from 3,000 reviews to 98,000.
3. **Curly apostrophes.** These are now normalised, so words like `don’t` can be found
   in the embedding table.

Regularisation has not been tried yet.

---

## Training your own model

Training stores everything in four SQLite databases inside a **work directory**. By
default this is `./wren-work`. You can change it with `--workdir` or `$WREN_WORKDIR`.

| Database | Contents |
|---|---|
| `test_data.db` | The reviews |
| `word_embeddings.db` | The word vectors |
| `neuron_weights.db` | Dense layer parameters |
| `convolution_layers.db` | Filter bank parameters |

These aren't in the repository. `wren setup` builds them.

### Option A: with the real data

You need two source files, and neither is in the repository.

**1. The reviews.** These come from the *Amazon Reviews 2023* dataset from the McAuley
Lab at UCSD. The Software category is a 496 MB gzip file containing 4.88 million
reviews:

```bash
curl -L -o Data/Software.jsonl.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Software.jsonl.gz
```

**2. The word embeddings.** You need fastText `wiki-news-300d-1M.vec` (999,994 words,
300 numbers each). The shipped Wren 1.0 weights were trained with these exact vectors
and **will only work with them**, so don't substitute GloVe or another fastText file.

```bash
curl -L -o Data/wiki-news-300d-1M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip -j Data/wiki-news-300d-1M.vec.zip -d Data/
```

**3. Build the `.jsonl` files and the databases:**

```bash
python -m wren.train.real_data_setup --reviews-gz Data/Software.jsonl.gz \
                                     --vec        Data/wiki-news-300d-1M.vec \
                                     --per-class  20000 \
                                     --out-dir    Data
wren setup --reviews Data/TestData.jsonl --embeddings Data/WordEmbeddings.jsonl
```

#### How `real_data_setup.py` keeps the results honest

- **It balances the ratings.** Real Amazon ratings are heavily skewed. In the Software
  category, 1.24 million reviews of a suitable length are 5★, compared with only
  155,000 that are 2★. On data like that, a model can score well just by learning to
  guess "5★" a lot. So each rating is sampled to the same count.
- **It samples from the whole file, not just the start.** It uses *reservoir
  sampling*, a technique for picking a fair random sample from a file too big to load
  into memory in one go. This means the sample isn't limited to whichever products
  happen to appear first.
- **It writes JSON safely, with `json.dumps`.** The older scripts built each line by
  hand with an f-string. Any review containing a double quote produced a broken line
  that was then silently skipped. That was 1.9% of reviews, and not a random 1.9%.

It also applies these rules:

- Reviews are kept only if they are **15 to 200 words** long.
- The embedding table only includes **words the corpus actually uses**. Importing all
  999,994 fastText vectors would create a table several gigabytes in size.
- Words with no vector fall back to a **row of zeros**.

### Option B: without the real data

To run the tests, or just to see everything work end to end, you can generate a
**synthetic** (made-up) corpus instead. Each star rating draws its words from its own
pool, so the task is learnable and the ratings are balanced:

```bash
wren setup --synthetic 3000
```

> This tells you **nothing** about real sentiment. It only exists so the tests can run
> on a machine without the Amazon dataset.

### Run training

```bash
wren train --batches 15000 --batch-size 32       # add --yes to skip the confirmation
```

> **⚠️ Training overwrites the existing model.** It resets every weight and filter,
> which destroys whatever model was in the work directory. To protect you, it asks
> for confirmation first. It also saves a timestamped backup of `neuron_weights.db`
> and `convolution_layers.db` before changing them.

A trained model lives in those databases. To use it with `Analyser`, export it with
`tools/export_model.py` (see [Pretrained weights](#pretrained-weights)).

#### What training reports

During training, Wren reports these for both the training data and a held-out
validation set of up to 2,000 reviews that it never trains on:

- **Loss and accuracy.**
- **Within-one-star accuracy**, alongside exact accuracy. Predicting 4★ when the truth
  is 5★ is the most common kind of mistake in star prediction, so the two figures
  together say much more than either alone.
- **The spread of true ratings and of predicted ratings.** A broken model that always
  predicts the same rating can still show a plausible-looking accuracy. Printing the
  spread of predictions exposes this, and if every prediction is the same rating, the
  run says so explicitly.

The train/validation split comes from a *seeded* shuffle (a shuffle that starts from a
fixed random seed), so it is identical on every run.

---

## Pretrained weights

`tools/export_model.py` exports Wren 1.0 as two portable files in `models/wren-1.0/`:

| File | Contents |
|---|---|
| `wren-1.0-weights.npz` | The filter bank and dense layer parameters |
| `wren-1.0-embeddings.npz` | The fastText vectors for the 37,498 words Wren knows |

These files are published as release assets rather than committed to the repository.
Their sizes and checksums are in `models/wren-1.0/MODEL.md`.

To export your own trained model:

```bash
python tools/export_model.py --sentiment-dir <work dir with the trained .db files>
                             # also needs Data/wiki-news-300d-1M.vec
```

As a safety check, the script runs the 2,000 test reviews through both the databases
and the exported files. It stops unless both give the same probabilities.

---

## Verification: proving the maths is right

**The gradient checks are the most valuable part of this repository.** They turn
*"I think the gradients are right"* into *"I proved the gradients are right."*

### Why this matters

Training depends on **gradients**, which tell the model which way to adjust each
parameter. Here they're calculated by hand-written backpropagation code (the
*analytic* gradient), and a small bug in that code can quietly ruin training. That is
exactly what happened to the [original version](#what-was-wrong-originally).

### How the check works

There is a slow but reliable way to estimate any gradient: nudge one parameter up by a
tiny amount (`eps`), then down by the same amount, and see how much the loss changes.
This is called a **central finite difference**. If the hand-written gradient matches
this estimate for every parameter, the backpropagation code is correct.

```
numerical      = (loss(w + eps) - loss(w - eps)) / (2 * eps)
relative_error = |analytic - numerical| / max(|analytic|, |numerical|, 1e-8)
```

**Any relative error above `1e-5` means the chain rule is broken somewhere.** (The
chain rule is the calculus rule that backpropagation is built on.)

```bash
python tests/test_gradients.py              # dense layers
python tests/test_filterbank_gradients.py   # filter bank layers
python tests/test_predict.py                # Analyser on the shipped Wren 1.0 files
```

None of these need a database or any data.

| Check | Worst relative error |
|---|---|
| Dense layer parameter gradients (81 weights and biases) | 3.9e-08 |
| Dense layer input gradient | 2.8e-09 |
| Filter bank kernel, bias and input gradients (4 window size and filter count configurations) | 2.4e-06 |

All are comfortably below the `1e-5` threshold.

### The end-to-end check: can it memorise?

The gradient checks test each layer on its own. This test checks that the whole
network is wired together correctly. A correctly built network with no regularisation
**must be able to memorise a small sample perfectly**. If it can't, something is broken.

This one does need the databases:

```bash
wren setup --synthetic 400                              # first, in a scratch work directory
python tests/test_overfit.py                            # 200 reviews, 60 epochs
python tests/test_overfit.py --reviews 40 --epochs 40   # quicker
```

If it can't get close to 100% training accuracy, there is still a bug. However,
failing within the default number of epochs (full passes over the data) isn't proof of
a bug on its own. Check whether the loss is still falling before drawing that
conclusion.

### Evaluation

The `evaluation/` folder measures the saved model honestly. See
`evaluation/README.md` for how to read the report.

```bash
python evaluation/build_test_set.py --reviews-gz Data/Software.jsonl.gz
python evaluation/evaluate.py
python evaluation/tests/test_evaluation.py     # quick checks, no real data needed
```

---

## Repository layout

```
src/wren/      the package: Analyser (model.py), the CLI (cli.py), text cleaning,
               layers/, and train/ (training loop, database setup, data builder)
tools/         export_model.py: turns the databases into the portable Wren 1.0 files
docs/          PLAN.md, the packaging plan
tests/         gradient checks, the overfit test, and prediction checks
evaluation/    fresh test set builder, evaluation script, and logs of the training runs
```

---

## Next steps

These are listed roughly in order of how likely they are to help:

1. **Add regularisation.** Try dropout or weight decay, and stop training based on a
   validation set that isn't also used to produce the final reported number. The
   12-point gap between training and test accuracy says this is the first thing to try.
2. **Deduplicate the training corpus**, and split the data by product or reviewer
   rather than by individual review. That way validation can't share products with
   training.
3. **Add more filters or more window sizes**, and let the model train the embeddings.
4. **Improve the middle ratings**, where recall is only 26 to 42%.
5. **Speed up the training loop**, so each experiment isn't a 70-minute wait.

---

## What was wrong originally

For anyone interested in the history, these were the bugs that mattered, in order of
impact:

1. **The backward pass was disconnected.** This was the critical bug.
   `ReLU_derivative` didn't take the gradient coming from the next layer. Instead it
   returned `self.output`, which is already `max(0, layer_output)`, so its masking line
   could never have any effect. `main.py` then threw away the return value of
   `calculate_derivatives` for every layer except the last. In effect, every layer
   updated its weights based on its own forward output rather than on the error. **This
   bug alone kept accuracy at chance level.**
2. **Softmax overflowed.** It didn't subtract the maximum value before exponentiating
   (the standard trick for numerical stability), so once the weights grew, `np.exp`
   returned infinity and the output became `nan` (not a number). On top of that, every
   row was divided by the sum of *row zero*, so in a batch of several reviews the
   outputs weren't valid probabilities at all.
3. **The batch size was effectively always one.** Gradients were added up one review at
   a time, and `calculate_derivatives` then averaged away the batch dimension.
4. **The convolution gradients were wrong in three ways:**
   - The forward pass produced duplicated rows and columns at its trailing edges.
   - The backward pass rotated the kernel, even though the forward pass is a
     *cross-correlation*, which doesn't flip the kernel. (Rotating is only correct for
     a true convolution.)
   - It assumed output position `i` came from input position `i * stride`, which the
     duplicated rows and columns broke.
5. **`reflectMatrix` changed its input in place.** It overwrote layer 1's stored forward
   output with a padded copy, which hid bug 4.
6. **There was no activation function between the convolution layers.** Without a
   non-linearity like ReLU between them, two linear layers collapse into a single
   linear layer, so the second one added nothing.
7. **The optimiser divided by the wrong number**, and the two variables controlling it
   were named the wrong way round.
8. **Data was used in strict ID order with no train/validation split.** The reported
   accuracy was therefore measured on data the model had just seen in the same pass.

Fixing these produced a model that could learn, but not much: 22.4%. Replacing the
stacked single-channel convolutions with the filter bank and training on 98,000
reviews took it to 47.0%.

---

## Licences and credits

- **Code:** MIT licensed (see `LICENSE`).
- **Word vectors:** fastText `wiki-news-300d-1M` by Mikolov, Grave, Bojanowski, Puhrsch
  and Joulin, *Advances in Pre-Training Distributed Word Representations* (LREC 2018),
  licensed CC BY-SA 3.0. `wren-1.0-embeddings.npz` is a subset of those vectors and is
  distributed under the same licence.
- **Training reviews:** Amazon Reviews 2023 (McAuley Lab, UCSD), published for research
  use. Check its terms before using the weights commercially.
