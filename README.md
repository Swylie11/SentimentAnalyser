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

**Wren 1.5** (the default) against Wren 1.0, on the same 2,000 reviews from five product
categories that neither model had ever seen:

| | Wren 1.5 | Wren 1.0 | Random guessing |
|---|---|---|---|
| **Exact star rating** | **52.5%** (95% CI: 50.3–54.7%) | 49.2% | 20.0% |
| **Within one star** | **90.9%** | 87.9% | 52.0% |

On the reviews where exactly one model is right, Wren 1.5 wins 253 to 188
(McNemar exact test, p = 0.002), so the improvement is not noise. The sections below
describe Wren 1.5; [How it got here](#how-it-got-here) covers 1.0 and what changed.

> **What's a 95% CI?** A *confidence interval*. Because the test set is a sample, the
> true accuracy could be a little higher or lower than 52.5%. The interval says we
> can be 95% confident that the true figure lies between 50.3% and 54.7%.

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
git clone https://github.com/Swylie11/SentimentAnalyser.git
cd SentimentAnalyser               # pip install . must run from the folder holding pyproject.toml
python -m venv .venv
.venv\Scripts\activate             # Windows
source .venv/bin/activate          # macOS / Linux
pip install .                      # NumPy is the only dependency (use -e . to develop)
pip install matplotlib             # only for the charts in tools/visualise.py
```

The pretrained **Wren 1.5** files (about 51 MB, see [Pretrained weights](#pretrained-weights))
are downloaded automatically the first time you use the model. They are saved to
`~/.cache/wren`, or to `$WREN_HOME` if you set it. To download them in advance, run
`wren fetch`. Wren 1.0 is still available: add `--model 1.0` to any command, or use
`Analyser(version="1.0")` in Python.

### Use it from the command line

```bash
wren predict "Does exactly what it says, works great"   # 5   Very positive
wren predict "waste of money" --proba                    # the star, then P(1★) … P(5★)
cat reviews.txt | wren predict - --json                  # one review per line, JSON output
wren eval --data Data/eval_test_multi.jsonl              # exact and within-one accuracy
wren predict --model 1.0 "waste of money"                # the original Wren 1.0
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
  → dropout 0.5 (training only)
  → dense layer 300 → 256, ReLU, dropout 0.2 (training only)
  → dense layer 256 → 5, softmax
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

The 300 phrase-detector scores are passed through **dense layers** (also called fully
connected layers), in which every input is connected to every output. One ReLU layer
of 256 neurons combines the phrase scores, and a final layer then produces five raw
scores, called **logits**, one per star rating.

Wren 1.0 had five hidden layers here (300 → 256 → 128 → 64 → 32 → 16). That deep,
narrowing stack sat at chance-level loss for thousands of batches before it started
learning, so 1.5 keeps only the first, widest layer.

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
direction to nudge it to reduce the loss, using the average loss over a batch of 32
reviews. All of it is hand-written:

- **AdamW optimiser.** Each parameter gets its own step size, scaled by how large its
  recent gradients have been, plus a momentum term. This is what got 1.5 past the
  long stall that plain SGD (used by 1.0) suffered. Weight decay of 0.01 gently pulls
  weights towards zero so no single one grows too large.
- **Cosine learning-rate schedule.** The learning rate starts at 0.001 and eases down
  to zero along a cosine curve, so training takes big steps early and fine ones at the end.
- **Dropout.** During training, half of the 300 phrase scores and a fifth of the 256
  hidden neurons are switched off at random on every batch. The model can't lean on
  any single feature, which reduces overfitting. Dropout is off when predicting.
- **Several seeds.** Training runs from more than one random starting point, and only
  the run that scores best on 2,000 held-out validation reviews is kept.

Within each run, the model is checked on 1,000 of those validation reviews every 500
batches, and the best-scoring version (*checkpoint*) is restored at the end.

### Model size

| Part | Trainable parameters |
|---|---|
| Filter banks | 360,300 |
| Dense layers | 78,341 |
| **Total** | **about 439,000** |
| Word embeddings | not trained (frozen) |

### What the model *cannot* do

- It is **not a pretrained language model** (like BERT or GPT). It learns only from
  the reviews it is trained on.
- It only understands word order **within 3- to 5-word windows**. It can't follow
  negation over a longer span (e.g. *"I expected to hate it, but… no"*), and it has
  no notion of sarcasm.
- A word with no embedding is read as a **row of zeros**, which carries no meaning.
  Wren 1.5 knows 65,726 words: those in its training reviews that fastText has a
  vector for. Typos, glued words such as `itbr` and anything rarer come out as zeros.

---

## Results

### Training setup

| | |
|---|---|
| **Data** | 577,500 Amazon reviews from five categories (Software, Video Games, Musical Instruments, Office Products, Toys and Games), balanced at 115,500 per star rating |
| **Split** | 575,500 for training, 2,000 held back for validation |
| **Training length** | 45,000 batches of 32 reviews, about 2.5 passes over the data, two seeds |
| **Time** | about 3 hours 50 minutes per seed on a CPU |

The test set is `Data/eval_test_multi.jsonl`: 2,000 reviews from the same five
categories, 400 per star rating, none of them in either model's training data. The
figures below come from `evaluation/mcnemar.py`, which runs Wren 1.0 and Wren 1.5 on
exactly the same reviews.

### Accuracy

| Data set | Reviews | Exact | Within one star |
|---|---|---|---|
| **Fresh test set, Wren 1.5** (never trained on, never used to pick the model) | 2,000 | **52.5%** (50.3–54.7) | **90.9%** (89.6–92.1) |
| Fresh test set, Wren 1.0 | 2,000 | 49.2% | 87.9% |
| Validation, Wren 1.5 (used to pick the seed and checkpoint) | 2,000 | 57.0% | 90.9% |
| Uniform random guessing | | 20.0% | 52.0% |

**Use the fresh test set figure (52.5%) when quoting this model.** The section on
[how much to trust these numbers](#how-much-to-trust-these-numbers) explains why.

#### Is 1.5 really better, or is it noise?

The two confidence intervals overlap, but that's the wrong test here: both models
answered the *same* 2,000 reviews, so they can be compared review by review.
**McNemar's test** does exactly that. It ignores the reviews both models got right
(797) or both got wrong (762) and looks only at the ones where they disagree:

| | Count |
|---|---|
| Only Wren 1.0 right | 188 |
| Only Wren 1.5 right | 253 |

If the two models were equally good, those 441 disagreements would split roughly 50/50.
A 253 to 188 split happens by chance with probability **p = 0.0023** (exact two-sided
test), so the improvement is real.

### Which ratings are easy, and which are hard

**Recall** is the share of reviews with a given true rating that the model got right.

| True rating | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---|---|---|---|---|
| Recall, Wren 1.5 | 61.8% | 47.0% | 43.8% | 37.5% | 72.5% |
| Recall, Wren 1.0 | 68.0% | 27.8% | 42.5% | 47.2% | 60.8% |

The extreme ratings are still much easier than the middle ones. Strong love or hate
uses distinctive language, while 2★, 3★ and 4★ reviews often mix praise and
complaint. Wren 1.5 gained most on 2★ (+19 points) and 5★ (+12), but lost ground on
1★ (−6) and 4★ (−10): it now calls more 1★ reviews 2★, and more 4★ reviews 3★ or 5★.

#### Confusion matrix

A **confusion matrix** shows every prediction the model made. Each row is a true
rating, and each column is what the model predicted. The diagonal (top-left to
bottom-right) holds the correct answers. Everything else is a mistake, and mistakes
close to the diagonal are near misses.

```
Wren 1.5, fresh test set (rows = true stars, columns = predicted stars)

            pred 1★  pred 2★  pred 3★  pred 4★  pred 5★
  true 1★ |   247      124       23        1        5
  true 2★ |   103      188       94       10        5
  true 3★ |    32      118      175       54       21
  true 4★ |     5       33      102      150      110
  true 5★ |    10       11       26       63      290
```

Reading the 4★ row, for example: of 400 four-star reviews, 150 were predicted
correctly. 102 were called 3★ and 110 were called 5★. Almost every mistake is a near
miss: only 6 of the 400 one-star reviews were called 4★ or 5★.

### How much to trust these numbers

**✅ The test set is clean.**
`evaluation/build_test_set.py` draws new reviews from the full dataset and throws away
any whose text appears in the training data. `evaluation/mcnemar.py` checks again
against both models' training reviews and finds 0 exact matches.

**⚠️ The validation figure (57.0%) is optimistic.** The same 2,000 validation reviews
picked the best checkpoint within each run and then the best of the two seeds, and
picking the best model on a set flatters that set. The 4.5-point gap to the fresh test
set is in line with Wren 1.0's. The test set was used once, after training finished,
and never to choose anything.

**ℹ️ Two seeds landed close together.** The two training runs scored 56.95% and
56.35% on validation, so the result doesn't depend on a lucky starting point.

**ℹ️ Some of the gain may come from the data rather than the model.** Wren 1.0 was
trained only on Software reviews, while the test set spans five categories that 1.5
trained on. 1.5 was not scored on 1.0's Software-only test set, because that set was
only checked against 1.0's training data and may overlap 1.5's.

### How it got here

| Version | Exact accuracy (fresh or held-out data) | Notes |
|---|---|---|
| Original A-Level submission | chance level | The backward pass was disconnected ([see below](#what-was-wrong-originally)). |
| After the backpropagation fixes: 5,000 reviews, two stacked convolutions | 22.4% | No better than always guessing the same rating (21.1%). |
| Wren 1.0: filter bank CNN, 98,000 Software reviews | 47.0% (Software test set), 49.2% (five-category test set) | Plain SGD, five hidden layers, no regularisation. |
| **Wren 1.5**: 575,500 reviews from five categories | **52.5%** (five-category test set) | AdamW, cosine schedule, one hidden layer, dropout. Today's model. |

Between 1.0 and 1.5, four things changed: about 5.9 times more training data from five
categories instead of one, AdamW with a cosine schedule instead of SGD, one hidden
layer instead of five, and dropout plus weight decay. Wren 1.0 scored 12.3 points
higher on its training reviews than on fresh ones; regularisation was the first item
on the list of next steps.

Three things changed between 22.4% and 47.0%:

1. **The architecture.** The old design ran a single 5×5 convolution followed by a
   3×3 one. That squashed the 200 × 300 embedding grid down to just 280 numbers,
   which was the bottleneck, and most of every input was padding anyway.
2. **The amount of data.** The training data grew from 3,000 reviews to 98,000.
3. **Curly apostrophes.** These are now normalised, so words like `don’t` can be found
   in the embedding table.

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
Lab at UCSD, one gzip file per category. Wren 1.5 uses five (the Software file alone is
496 MB and holds 4.88 million reviews):

```bash
for c in Software Video_Games Musical_Instruments Office_Products Toys_and_Games; do
  curl -L -o Data/$c.jsonl.gz \
    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/$c.jsonl.gz
done
```

**2. The word embeddings.** You need fastText `wiki-news-300d-1M.vec` (999,994 words,
300 numbers each). The shipped Wren weights were trained with these exact vectors
and **will only work with them**, so don't substitute GloVe or another fastText file.

```bash
curl -L -o Data/wiki-news-300d-1M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip -j Data/wiki-news-300d-1M.vec.zip -d Data/
```

**3. Build the `.jsonl` files and the databases:**

Wren 1.5's code lives in `src/wren-1.5` and runs through `tools/wren15.py` (the
installed `wren train` still trains the 1.0 architecture). From the repo root:

```bash
python tools/wren15.py data --reviews-gz Data/Software.jsonl.gz Data/Video_Games.jsonl.gz \
                                         Data/Musical_Instruments.jsonl.gz Data/Office_Products.jsonl.gz \
                                         Data/Toys_and_Games.jsonl.gz \
                            --vec        Data/wiki-news-300d-1M.vec \
                            --per-class  20000 \
                            --exclude    Data/eval_test_multi.jsonl \
                            --out-dir    Data/multi
python tools/wren15.py setup --workdir wren-work \
                             --reviews Data/multi/TestData.jsonl --embeddings Data/multi/WordEmbeddings.jsonl
```

`--per-class` is sampled from each file, so this gives 5 × 5 × 20,000 = 500,000 reviews.
`--base FILE` keeps an existing corpus and adds to it, and `--exclude` makes sure the
test set's reviews are never sampled.

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
python tools/wren15.py train --workdir wren-work --batches 45000 --seeds 0 1   # add --yes to skip the confirmation
python tools/wren15.py watch --workdir wren-work                               # live stats, in a second window
```

`bash tools/train.sh` does the whole thing unattended (data, databases, multi-seed
training) and keeps Windows awake while it runs. When training ends, the saved model
is scored once on `Data/eval_test_multi.jsonl` and the run is appended to
`evaluation/wren15_history.jsonl`. `wren train` still trains the Wren 1.0 architecture.

> **⚠️ Training overwrites the existing model.** It resets every weight and filter,
> which destroys whatever model was in the work directory. To protect you, it asks
> for confirmation first. It also saves a timestamped backup of `neuron_weights.db`
> and `convolution_layers.db` before changing them.

A trained model lives in those databases. To use it with `Analyser`, export it with
`tools/export_wren15.py` (see [Pretrained weights](#pretrained-weights)).

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

Each release is two portable files, published as assets on the matching GitHub release
rather than committed to the repository. `wren` downloads the ones it needs and checks
their sha256.

| Release | File | Contents |
|---|---|---|
| `v1.5` (default) | `wren-1.5-weights.npz` | The filter bank and dense layer parameters (3.4 MB) |
| | `wren-1.5-embeddings.npz` | The fastText vectors for the 65,726 words Wren 1.5 knows (47.6 MB) |
| `v1.0` | `wren-1.0-weights.npz` | The filter bank and dense layer parameters (3.7 MB) |
| | `wren-1.0-embeddings.npz` | The fastText vectors for the 37,498 words Wren 1.0 knows (27.4 MB) |

Checksums and full details are in `models/wren-1.5/MODEL.md` and `models/wren-1.0/MODEL.md`.

To export your own trained model:

```bash
python evaluation/mcnemar.py --wren15 <work dir>    # first: the reference predictions
python tools/export_wren15.py --workdir <work dir>  # Wren 1.5 architecture
python tools/export_model.py --sentiment-dir <work dir>   # Wren 1.0 architecture; needs Data/wiki-news-300d-1M.vec
```

As a safety check, both scripts run the 2,000 test reviews through the databases and
the exported files, and stop unless they give the same answers.

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
python tests/test_predict.py                # Analyser on the shipped Wren 1.5 and 1.0 files
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
# Wren 1.0's test set (Software only)
python evaluation/build_test_set.py --reviews-gz Data/Software.jsonl.gz
# Wren 1.5's test set: the same five categories as the training corpus
python evaluation/build_test_set.py --reviews-gz Data/Software.jsonl.gz Data/Video_Games.jsonl.gz \
    Data/Musical_Instruments.jsonl.gz Data/Office_Products.jsonl.gz Data/Toys_and_Games.jsonl.gz \
    --db wren-work-multi/test_data.db --out Data/eval_test_multi.jsonl
python evaluation/evaluate.py                  # the detailed report for a Wren 1.0 work directory
python evaluation/mcnemar.py                   # Wren 1.0 vs 1.5 on the same reviews, with McNemar's test
python evaluation/tests/test_evaluation.py     # quick checks, no real data needed
```

---

## Repository layout

```
src/wren/      the package: Analyser (model.py, loads 1.5 or 1.0), the CLI (cli.py),
               text cleaning, layers/, and train/ (the Wren 1.0 training code)
src/wren-1.5/  Wren 1.5's training code: AdamW, dropout, multi-seed runs, live dashboard
tools/         wren15.py: runs the Wren 1.5 training code; train.sh: unattended training
               export_model.py / export_wren15.py: turn the databases into the portable
               Wren 1.0 / 1.5 files
               visualise.py: charts of results and training runs (needs matplotlib)
docs/          PLAN.md, the packaging plan
tests/         gradient checks, the overfit test, and prediction checks
evaluation/    fresh test set builder, evaluation script, and logs of the training runs
```

---

## Next steps

These are listed roughly in order of how likely they are to help:

1. **Fix 1★ and 4★.** Wren 1.5 lost 6 and 10 points of recall on these while gaining
   on 2★ and 5★. The middle ratings are still the weak spot (38 to 47% recall).
2. **Merge the two code bases**, so `wren train` trains the 1.5 architecture and
   `src/wren-1.5` can go.
3. **Deduplicate the training corpus**, and split the data by product or reviewer
   rather than by individual review. That way validation can't share products with
   training.
4. **Add more filters or more window sizes**, and let the model train the embeddings.
5. **Speed up the training loop**, so each experiment isn't a four-hour wait per seed.

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
  licensed CC BY-SA 3.0. `wren-1.5-embeddings.npz` and `wren-1.0-embeddings.npz` are
  subsets of those vectors and are distributed under the same licence.
- **Training reviews:** Amazon Reviews 2023 (McAuley Lab, UCSD), published for research
  use. Check its terms before using the weights commercially.
