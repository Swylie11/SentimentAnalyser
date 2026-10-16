# Kestrel 1.5 Technical Specification: General-Domain Sentiment

*Written 2026-10-12. Status: proposed. Developed as Kestrel 1.1, released as 1.5. Depends on
Kestrel 1.0 being released first.*

Kestrel 1.0 reads Software reviews. Kestrel 1.5 should read **a sentence about anything**:
a tweet, a news headline, a line from a message, and still give a sensible sentiment. It keeps
1.0's architecture exactly and changes three things: what it's trained on, what it outputs, and
what it's gated on. Like the 1.0 spec, this explains every decision, links reading, and contains
no code.

Read the [Kestrel 1.0 spec](kestrel-1.0-spec.md) first, especially section 11.2. That section
is where the failures this spec fixes are measured.

---

## Contents

1. [Goal and what changes](#1-goal-and-what-changes)
2. [What 1.0 is expected to get wrong](#2-what-10-is-expected-to-get-wrong)
3. [Data changes](#3-data-changes)
4. [Architecture changes](#4-architecture-changes)
5. [Training changes](#5-training-changes)
6. [Ablation: domain-adversarial training](#6-ablation-domain-adversarial-training)
7. [Evaluation and release gate](#7-evaluation-and-release-gate)
8. [Build order, ablations and risks](#8-build-order-ablations-and-risks)
9. [Out of scope](#9-out-of-scope)
10. [Reading list](#10-reading-list)

---

## 1. Goal and what changes

| | Kestrel 1.0 | Kestrel 1.5 | Why |
| --- | --- | --- | --- |
| Trunk (input stage, conv, 2 transformer blocks, pooling) | as specified | **unchanged** | The trunk can represent context over any distance in any text. Nothing in it is review-specific. The problems are in the data and the labels, so that's what changes. Leaving the trunk alone also means any difference from 1.0 is caused by the data or the head and nothing else. |
| Training text | Software reviews | Reviews from 4 categories, review titles, random crops, general 3-class sentence sets | Each addition targets a failure listed in section 2 |
| Output | 5 stars | 5 stars **plus** negative / neutral / positive | Stars can't express "no opinion". A separate head learns its own boundary for neutral. |
| Parameters | 1,955,889 (1.0 spec A1) | **1,957,428** (+1,539 for the new head) | Still well inside the mid class |
| Gate | Beats Wren 1.5 on reviews | Beats Kestrel 1.0 on non-review text, **without losing** review accuracy | General-domain sentiment is the new goal. Review performance is the guard that stops it being bought by forgetting. |

**Why not a new class or a bigger model:** domain shift is a data and label problem, not a
capacity one. 1.0 already has the context mechanism. Adding parameters without changing the
data would just fit Software reviews more closely. Real "understands any text" capacity comes
from pretraining (section 9), and that belongs to the next class up.

---

## 2. What 1.0 is expected to get wrong

The 1.0 out-of-domain evaluation (1.0 spec, 11.2) replaces these predictions with measured
numbers. Each row says which change in this spec targets it.

| Failure | Example | Cause | Fix in 1.5 |
| --- | --- | --- | --- |
| No neutral | "The meeting is at 3pm" → 3★ or worse | Reviews are almost never opinion-free. 3★ was trained on *mixed* reviews. | 3-class head with real neutral examples (4.1, 3.4) |
| Domain-specific polarity | "unpredictable plot" vs "unpredictable steering" | Every word's polarity was learned only from software contexts | More review categories (3.1), general sets (3.4), DANN ablation (6) |
| Review-only cues | "returned it", "five stars", "works as described" | The strongest cues in training don't exist in other text | Dilute them with more varied text (3.1, 3.4) |
| Short inputs | A 9-word tweet | Every training review has at least 15 words, so short inputs were never seen | Titles (3.2) and crops (3.3) |
| Vocabulary | `@user`, `#love`, URLs, emoji | Not in fastText's vocabulary, so they become UNK | Tokeniser normalisation (3.5) |
| Register | Slang, formal finance language | Never seen in training | General sets (3.4). Only partly fixable at this scale (section 9). |

**Why write down predictions before measuring:** if 1.0's polarity accuracy on TweetEval turns
out high while 3-class F1 is low, the neutral row is the whole problem and the general-data work
can shrink. Knowing in advance which measurement would change the plan makes the 1.0 numbers
actionable instead of just interesting.

---

## 3. Data changes

### 3.1 More review categories, same total size

| Item | Specification | Why |
| --- | --- | --- |
| Categories | Software, Electronics, Video Games, Books (all from Amazon Reviews 2023) | Each has a different vocabulary of praise and complaint. Electronics is physical hardware ("overheats", "flimsy"). Video Games is entertainment, closest to films. Books uses narrative and opinion language ("predictable", "moving"). Together they break the link between a sentiment word and software contexts. |
| Size | **About 132,000 per class in total, as in 1.0** (1.0 spec A1), split evenly: about 33,000 per class per category | Keeping the total fixed means 1.5 tests *diversity*, not *more data*. It also keeps the compute budget at 1.0's. If 1.5 beats 1.0, it's because the text was more varied, not because there was more of it. |
| Cleaning and split | 1.0 rules exactly: dedup, at least 15 words and at most 128 tokens, `parent_asin` hash 90/5/5, applied per category | One pipeline for every category. Splitting by product per category keeps leakage out. |
| Locked test | 1.0's Software test set stays the in-domain gate. Each new category also gets a balanced 2,000-per-class test set, reported but not gated. | The Software test set is the only one with Wren and Kestrel 1.0 results, so the "no regression" check has to use it. The per-category sets show whether the model now handles reviews in general, at no extra cost. |

### 3.2 Review titles as extra examples

Amazon titles are short, sentence-like and carry the review's sentiment ("Broke after two
days", "Exactly what I needed"). They are the cheapest source of short labelled text there is.

- **Pipeline change:** `real_data_setup.py` currently reads only `text` and `rating` from each
  record (line 84) and drops `title`. Keep it.
- **Label:** the review's star rating.
- **Split:** a title goes in the same split as its review. Why: the title and body of one review
  are near-duplicates in meaning. Putting a title in train and its body in test would leak the
  answer.
- **Filter out auto-titles.** Amazon has historically generated titles like "Five Stars" and
  "One Star" from the rating. They are the label written out as text, and training on them
  teaches nothing but "the word five means 5★". Drop any title that matches
  `(one|two|three|four|five) stars?` after lowercasing, and any title under 2 words.
- **Dedup** titles on cleaned text, like the reviews. Why: "Great product" will appear tens of
  thousands of times, and without dedup a few stock phrases would dominate the title batches.
- **No 15-word minimum** for titles. Short is the point.

### 3.3 Random crops

During training, with probability **0.25** per review, replace the review with a random
contiguous window of **8–40 words** from it (length chosen uniformly). The crop is redrawn every
epoch.

- **Why:** it turns every long review into many short examples, so the model learns to judge
  sentiment from a fragment, which is what a single sentence is.
- **Why only 0.25:** a crop can miss the sentiment entirely. A window saying "the box arrived on
  Tuesday" from a 1★ review is still labelled 1★, which is label noise. A quarter is enough for
  the model to see plenty of short inputs while three quarters of the reviews stay whole and
  clean. The 0 / 0.25 / 0.5 ablation (8.2) checks this choice.
- **Why 8–40 words:** that's the range a typical sentence or tweet falls in. Below 8, too many
  crops have no sentiment at all.
- **Why on the fly rather than precomputed:** each epoch sees different crops of the same
  review, which is free extra variety and costs nothing to store.

### 3.4 General-domain 3-class training sets

| Set | Train size | Validation | Use | Why |
| --- | --- | --- | --- | --- |
| SST, collapsed to 3 classes (very neg + neg → neg, neutral, pos + very pos → pos) | 8,544 sentences | SST dev, 1,101 | Trains the 3-class head | Clean sentence-level labels, including real neutral sentences |
| TweetEval sentiment | 45,615 tweets | TweetEval val, 2,000 | Trains the 3-class head | Short, informal text on any topic, mostly neutral: the closest data to "a sentence about anything" |
| Financial PhraseBank | — | — | **Never trained on** | Kept as a domain the model has never seen (7.1). It's the only honest test of whether generalisation happened, rather than learning two more domains. |

**Why SST goes to the 3-class head, not the star head:** SST's middle label means *neutral*,
while 3★ means *mixed*. Training the star head on SST would teach it that 3★ means "no opinion",
which would damage the review predictions. Collapsing SST to 3 classes keeps each head's labels
meaning one thing.

**Why test sets stay locked:** the SST-5 test and TweetEval test sets are 1.0's out-of-domain
evaluation sets. 1.5 trains on their train splits only, so the test numbers are directly
comparable between 1.0 and 1.5.

### 3.5 Tokeniser normalisation

Three rules, added before the existing tokeniser. Since 1.0 spec A1, that tokeniser turns
punctuation into spaces, so the URL and mention rules must run **before** it. Otherwise it
splits `@user` into `user` and a URL into `http www example com`, and the rules no longer match.

| Rule | Example | Why |
| --- | --- | --- |
| Strip a leading `#` from a word | `#love` → `love` | Hashtags are usually ordinary words. Without the `#` they're in the vocabulary and keep their meaning. **Already done by 1.0's punctuation rule (A1)**, which turns `#` into a space, so no new code is needed. Keep it as a test case. |
| Drop tokens starting with `http` or `www.` | | URLs carry no sentiment, and every one would otherwise be a different UNK |
| Drop `@user` and `@name` mentions | | Names carry no sentiment. TweetEval already replaces them with `@user`. |

- **Why only these three:** they're the biggest sources of UNKs in tweets, and each is one
  simple rule that does nothing to review text. The UNK rate from 1.0's 11.2 measurement should
  be re-measured after this change. If emoji turn out to be a large share of what's left, mapping
  the commonest ones to words is a follow-up, but it isn't specified until the numbers show it's
  needed.
- **Retokenise everything** with the new rules, including Wren's locked test set, and confirm
  the review tokens are unchanged. Why: a change that silently altered review tokenisation would
  make the in-domain comparison with 1.0 unfair.

---

## 4. Architecture changes

### 4.1 The 3-class head

```
pooled (B, 512) ─┬─ dropout 0.3 → Linear 512→5 → softmax → stars      (1.0, unchanged)
                 └─ dropout 0.3 → Linear 512→3 → softmax → polarity   (new, 1,539 params)
```

- **Same form as the star head** (1.0 spec, 6.4): one linear layer, dropout 0.3, He init,
  softmax plus cross-entropy with the combined `(p − y)/B` gradient. Why: it reuses code that's
  already gradient-checked, and the reasons given there for a single linear layer apply equally
  here.
- **Each batch trains one head.** Review batches compute the star loss, general batches the
  polarity loss. Both gradients flow into the same trunk. Why: every example has exactly one
  label type, so there is nothing to give the other head. Inventing a "polarity" label for
  reviews (for example 1–2★ → negative) would bring back the mixed-vs-neutral confusion that
  justifies having a second head.
- **Why a second head rather than retraining the 5 classes:** stars are still the right output
  for reviews, and 1.0 users depend on them. The trunk learns from both tasks, and each head
  keeps its own meaning. This is standard multi-task learning (Caruana 1997, Ruder 2017).

### 4.2 Inference and the `Analyser` API

`Analyser(model="kestrel-1.5").predict(text)` returns the star result exactly as Wren and
Kestrel 1.0 do, plus a new `polarity` field: the label and its three probabilities.

- **Why keep the stars output:** the `Analyser` API must work the same across models (1.0 spec,
  section 1). Code written for Wren keeps working.
- **Document which output to use:** stars for reviews, polarity for anything else. Why: on a
  tweet the star head is still extrapolating, and saying so is more honest than hiding it.

---

## 5. Training changes

Everything not listed here is unchanged from 1.0 spec section 7: AdamW, learning-rate schedule,
clipping, dropout rates, batch size 32 and bucketing.

| Item | Specification | Why |
| --- | --- | --- |
| Batch sources | Each batch comes from **one** source: reviews (with crops) **60%**, titles **15%**, general sets **25%** | One source per batch means one head and one loss per step, and bucketing still works because each source has a typical length. The proportions keep reviews the majority, so the star head doesn't regress, while giving the general sets a regular share of updates. |
| Inside general batches | Sample from SST and TweetEval pooled together, in proportion to their size | Simple and unbiased. TweetEval dominates (84%), which suits the goal, because tweets are the furthest from reviews. |
| Epoch | One pass over the review data. Other sources are sampled with replacement as needed. | Reviews are the largest source, so the epoch counts in their units, as in 1.0. At 25% of batches, the general sets are seen about 5 times per epoch (see the overfitting risk in 8.3). |
| Selection score | `S = mean(Amazon val 5-class macro-F1, SST dev 3-class macro-F1, TweetEval val 3-class macro-F1)` | 1.0 stopped on validation MAE, but MAE doesn't exist for the 3-class head. Macro-F1 is on the same 0–1 scale for both heads, so an unweighted mean gives each domain equal say. Financial PhraseBank is left out on purpose (3.4). |
| Early stopping | On `S`, same patience as 1.0 | The same mechanism with a score that covers both tasks |
| Initialisation | **From scratch** for the release run. Warm-starting from 1.0 is an ablation. | From scratch matches 1.0's procedure, so the comparison is about data and head only. Warm-starting is cheaper but inherits 1.0's Software-only habits, and it's unclear how much those would wash out. The ablation (8.2) answers that. |

**Compute:** title and general batches are short (about 5–30 words), so bucketing makes them
much cheaper than review batches. With 40% of batches short, expect each epoch to cost about
**10–20% more than 1.0's**. Re-measure at milestone 2 (8.1) before planning the full run.

---

## 6. Ablation: domain-adversarial training

A small extra classifier tries to guess **which source** a pooled vector came from (Amazon, SST
or TweetEval). The trunk is trained to make that impossible: between the pooling and the domain
classifier sits a **gradient reversal layer** (GRL).

| Part | Specification |
| --- | --- |
| GRL | Forward: identity. Backward: multiply the incoming gradient by `−λ`. |
| Domain classifier | 512 → 64 (ReLU) → 3, softmax plus cross-entropy, 33,027 parameters. Training only: dropped at inference. |
| `λ` schedule | `λ = 2 / (1 + exp(−10p)) − 1`, where `p` goes from 0 to 1 over training |
| Applied to | Every batch, alongside that batch's sentiment loss |

- **Why it works:** the domain classifier gets better at telling sources apart. The reversed
  gradient pushes the trunk the other way, removing whatever in the pooled vector reveals the
  source. What's left has to be features that work in every domain, which is the definition of
  transferable sentiment (Ganin et al. 2016).
- **Why the `λ` ramp:** early in training the domain classifier is random, so its reversed
  gradient is noise, and full strength from step 1 tends to destabilise training. The schedule
  is Ganin's: it starts near 0 and reaches about 1 by the end.
- **Why it suits this codebase:** the GRL is the simplest backward pass in the whole model.
  It still gets a gradient check: the check must confirm the sign flip.
- **Why it's an ablation and not the default:** gains from DANN vary a lot between setups, and
  it adds a hyperparameter schedule and a second classifier to debug. It stays in the release
  only if it improves Financial PhraseBank, the unseen domain, without breaking the
  in-domain gate.

---

## 7. Evaluation and release gate

### 7.1 Evaluation

Use the 1.0 spec's 11.2 harness and sets unchanged, plus:

- the 3-class head scored directly on SST-3, TweetEval and Financial PhraseBank;
- the per-category review test sets (3.1);
- the probe set extended with **100 general-topic pairs** (weather, sport, work, news) that
  differ only in negation or contrast. Why: the 1.0 probe pairs are all reviews, so they can't
  show whether negation handling carried over.

Compare against **Kestrel 1.0's numbers from its own release**, measured with the star-to-3-class
mapping.

### 7.2 Release gate

Checked once, after the configuration is fixed.

| Gate | Why it's there |
| --- | --- |
| **Financial PhraseBank 3-class macro-F1 beats Kestrel 1.0**, McNemar `p < 0.05` | The headline gate. 1.5 never trains on this domain, so an improvement here is real generalisation, not just learning more domains. |
| TweetEval and SST-3 macro-F1 each beat Kestrel 1.0, McNemar `p < 0.05` | Required, but expected: 1.5 trains on their train splits. Failing this would point to a bug, not a research result. |
| TweetEval neutral recall higher than Kestrel 1.0's | Neutral is the failure the second head exists to fix |
| Software locked test: exact accuracy no more than **1 point** below Kestrel 1.0, within-one accuracy not lower | Generality mustn't be bought by forgetting reviews. 1 point is roughly the 95% CI half-width at n = 10,000, so the gate means "not measurably worse". |
| Every gradient check passes, including the new head and (if kept) the GRL | Same reason as 1.0: unverified gradients make every other number untrustworthy |
| `MODEL.md` updated: training data now includes SST and TweetEval, with each licence checked **for training**, not just evaluation | Training on a dataset is a different use from evaluating on it, and some terms treat the two differently. Tweet text in particular comes with Twitter/X terms. |
| Out-of-domain limits published | Even 1.5 isn't a general-purpose model (section 9). Users need to know where it stops working. |

**Expected effect:** the biggest gains should be TweetEval neutral recall and the
3-class F1 on all three sets. Gains on Financial PhraseBank should be smaller, because finance
language stays far from anything in training. This is an estimate to check results against, not
a gate.

---

## 8. Build order, ablations and risks

### 8.1 Milestones

| # | Milestone | Done when | Depends on |
| --- | --- | --- | --- |
| 0 | Kestrel 1.0 released with 11.2 numbers | Numbers in 1.0 `MODEL.md` | — |
| 1 | Data: titles kept and filtered, 4-category corpus built, general sets prepared, tokeniser rules added | Corpus files written. Review tokens unchanged by the new rules. | 0 |
| 2 | Run A: 1.0 setup + titles + crops (Software only, star head only) | 11.2 measured. Throughput re-measured. | 1 |
| 3 | Run B: + 4 categories | 11.2 measured | 2 |
| 4 | Run C: + 3-class head + general sets | 7.1 measured | 3 |
| 5 | Ablations (8.2) | Each run once, decision logged | 4 |
| 6 | Full run from scratch, best configuration | Early stopping triggers | 5 |
| 7 | Release | Every gate in 7.2 passes | 6 |

**Why runs A, B and C are separate:** each adds one group of changes, so each change's effect on
transfer is measured on its own, which is the same reasoning as 1.0's one-block-at-a-time
milestones. It also shows which changes earn their place. If run A already fixes most of
the problem, the rest might not be worth their cost.

**Why runs A–C use 25% of the data:** at full scale each run takes 1.0's 7–12 hours. At 25%,
three runs fit in roughly the time of one full run. They are for choosing a direction, and only
milestone 6 is a full run.

### 8.2 Ablations, one change each, scored on `S`

| Ablation | Question it answers | Why it's worth a run |
| --- | --- | --- |
| DANN on / off (section 6) | Does removing domain information help unseen domains? | Its effect varies between setups, so it only stays in if it helps here |
| Crop probability 0 / 0.25 / 0.5 | How much fragment training is worth its label noise | The 0.25 default is a judgement, not evidence |
| General batch share 15% / 25% / 35% | The trade-off between transfer and review accuracy | It's the main dial between the two goals |
| Warm start from 1.0 vs from scratch | Can 1.5 reuse 1.0's training? | If warm start matches, future versions get much cheaper |
| Titles on / off | Do titles add anything once crops exist? | Both target short input. If one covers it, drop the other. |

### 8.3 Risks

| Risk | Likelihood | Effect | Mitigation |
| --- | --- | --- | --- |
| In-domain accuracy drops | medium | Gate fails | Reviews stay 60% of batches. The 1-point gate catches it. The batch-share ablation finds the balance. |
| Overfitting the small general sets | medium | Good on validation, worse on test | Dropout, early stopping on `S` (which includes their validation sets), the 15% batch-share ablation |
| Label noise from crops | medium | Slightly worse calibration | Crops are a minority. The crop ablation measures the cost. |
| Gains only on trained domains, not on Financial PhraseBank | medium | Headline gate fails | DANN ablation. If it still fails, report honestly: it means the limit is pretraining (section 9), not engineering. |
| Licence terms forbid training on a set | low–medium | Set can't be used | Check at milestone 1, before any training. TweetEval can be swapped for another 3-class set. |

---

## 9. Out of scope

| Not in 1.5 | Why |
| --- | --- |
| **Language-model pretraining** (ULMFiT, BERT) | The real reason modern models generalise: they learn language from billions of words before seeing a single label. At 1.0's 7–12 hours per labelled run on CPU, meaningful pretraining isn't feasible in NumPy at this size. It's the defining feature to reserve for the heavy class. |
| Sarcasm and irony | Needs world knowledge and context beyond the sentence. Even large pretrained models do poorly on it. |
| Emotions (joy, anger, fear …) | A different label space with different datasets (for example GoEmotions). Polarity comes first. |
| Aspect-based sentiment ("battery great, screen bad") | Needs per-target outputs, which is a different head and different data |
| Other languages | fastText wiki-news is English-only |

---

## 10. Reading list

| # | Resource | Covers | Section |
| --- | --- | --- | --- |
| 1 | **Blitzer et al. 2007, [*Biographies, Bollywood, Boom-boxes and Blenders*](https://aclanthology.org/P07-1056/)** | Domain shift in sentiment, with the classic multi-category Amazon setup | 2, 3.1 |
| 2 | **Ganin et al. 2016, [*Domain-Adversarial Training of Neural Networks*](https://arxiv.org/abs/1505.07818)** | DANN, the gradient reversal layer, the `λ` schedule | 6 |
| 3 | **Ruder 2017, [*An Overview of Multi-Task Learning in Deep Neural Networks*](https://arxiv.org/abs/1706.05098)** | Shared trunks with several heads, and why it helps | 4.1 |
| 4 | Caruana 1997, [*Multitask Learning*](https://link.springer.com/article/10.1023/A:1007379606734) | The original argument for multi-task learning | 4.1 |
| 5 | Socher et al. 2013, [*Sentiment Treebank*](https://aclanthology.org/D13-1170/) | SST and its label scale | 3.4 |
| 6 | Barbieri et al. 2020, [*TweetEval*](https://arxiv.org/abs/2010.12421) | The tweet dataset, its splits and metric | 3.4, 7 |
| 7 | Rosenthal et al. 2017, [*SemEval-2017 Task 4*](https://aclanthology.org/S17-2088/) | Where TweetEval's sentiment data comes from, and how it was labelled | 3.4 |
| 8 | Malo et al. 2014, [*Good Debt or Bad Debt*](https://arxiv.org/abs/1307.5336) | Financial PhraseBank and finance-specific polarity | 3.4, 7 |
| 9 | Howard and Ruder 2018, [*ULMFiT*](https://arxiv.org/abs/1801.06146) | Why pretraining is what makes text models general, and what the next class would add | 9 |
| 10 | Hou et al. 2024, [arXiv 2403.03952](https://arxiv.org/abs/2403.03952) | Amazon Reviews 2023 fields (including `title`) and category sizes | 3.1, 3.2 |
