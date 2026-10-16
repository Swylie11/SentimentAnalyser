**Wren 1.5**

A five-class (1 to 5 star) review sentiment analyser: a Kim 2014 style text CNN written from scratch in NumPy, with no PyTorch, TensorFlow or scikit-learn. `wren` now uses 1.5 by default; 1.0 is still one flag away.

**Results**
On 2,000 balanced Amazon reviews from five product categories that neither model saw:

| | Wren 1.5 | Wren 1.0 |
|---|---|---|
| Exact star rating | **52.5%** (95% CI 50.3 to 54.7) | 49.2% |
| Within one star | **90.9%** | 87.9% |

Where exactly one model is right, 1.5 wins 253 to 188: McNemar exact test p = 0.0023. Chance is 20%.

**What changed**
- Trained on 575,500 reviews from five categories (Software, Video Games, Musical Instruments, Office Products, Toys and Games), up from 98,000 Software reviews
- AdamW with a cosine learning-rate schedule instead of plain SGD
- One hidden layer of 256 instead of five
- Dropout on the filter features (0.5) and the hidden layer (0.2)

**Install and use**
```bash
pip install .
wren predict "Does exactly what it says, works great"
wren predict --model 1.0 "Does exactly what it says, works great"   # the original
```
```python
from wren import Analyser
Analyser().predict("Does exactly what it says, works great")       # Wren 1.5
Analyser(version="1.0")                                             # Wren 1.0
```

**Assets (required)**
Both files must keep these exact names, because `wren` downloads them from this release (tag `v1.5`) and checks their hashes.

| File | Size | sha256 |
|---|---|---|
| `wren-1.5-weights.npz` | 3.4 MB | `cd0b7addc30cf408880d24bb8bfcf81d81fa97d69f741cbd2334584c72cd8dcd` |
| `wren-1.5-embeddings.npz` | 47.6 MB | `f8fa4cacbe4a1dcb9450161ce44dfca1778799027ff8f602c2720593ef3a6e84` |

**Licences**
- Code: MIT.
- `wren-1.5-embeddings.npz` is a subset of fastText `wiki-news-300d-1M` (Mikolov et al., LREC 2018) and is distributed under **CC BY-SA 3.0**.
- Trained on Amazon Reviews 2023 (five categories above), which is published for research use. Check its terms before commercial use.

**Limits**
- Better than 1.0 on 2★ (+19 points) and 5★ (+12), but worse on 1★ (-6) and 4★ (-10).
- Words outside its 65,726-word vocabulary are read as zeros.
- It has no notion of sarcasm or of negation beyond 3 to 5 word windows.

Full details: `models/wren-1.5/MODEL.md`.

---

**Wren 1.0**

A five-class (1 to 5 star) review sentiment analyser. It is a Kim 2014 style text CNN written from scratch in NumPy, with no PyTorch, TensorFlow or scikit-learn.

**Results**
On 2,000 balanced Amazon reviews it never saw:
- Exact star rating: **47.0%** (95% CI 44.8 to 49.2; chance is 20%)
- Within one star: **86.2%**

**Install and use**
```bash
pip install .
wren predict "Does exactly what it says, works great"
```
```python
from wren import Analyser
Analyser().predict("Does exactly what it says, works great")   # 5
```
numpy is the only dependency. The model files download on first use into `~/.cache/wren`, or `$WREN_HOME` if set.

**Assets (required)**
Both files must keep these exact names, because `wren` downloads them from this release and checks their hashes.

| File | Size | sha256 |
|---|---|---|
| `wren-1.0-weights.npz` | 3.7 MB | `25702430d6b6bb2b079108784eb958771e46230bc5796e5a0f82333284726485` |
| `wren-1.0-embeddings.npz` | 27.4 MB | `100c4c929b9d758e3c69f5ebac2378da4a605dde8e5f33fa3e13a53699eb7f3e` |

**Licences**
- Code: MIT.
- `wren-1.0-embeddings.npz` is a subset of fastText `wiki-news-300d-1M` (Mikolov et al., LREC 2018) and is distributed under **CC BY-SA 3.0**.
- Trained on Amazon Reviews 2023 (Software), which is published for research use. Check its terms before commercial use.

**Limits**
- Middle ratings are hard: recall is 26% for 2 stars and 34% for 3 stars, against 69% for 1 star and 64% for 5 stars.
- No regularisation was used, and the model scores about 12 points higher on its training reviews than on new ones.
- Words outside the 37,498-word vocabulary are read as zeros.
- It has no notion of sarcasm or of negation beyond 3 to 5 word windows.

The full description, the breakdown by star rating and the training history are in the README and `models/wren-1.0/MODEL.md`.
