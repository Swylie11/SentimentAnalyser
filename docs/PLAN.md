# Plan: make Wren installable (CLI + Python library)

Goal: `pip install` then either `wren predict "text"` or
`from wren import Analyser; Analyser().predict("text")`, on a machine that has none of
this repo's local files.

Phase 1 is the real work and the only risky part. Phases 2 to 4 are mechanical.

---

## 0. What the audit found (why Phase 1 exists)

Measured on this machine, 2026-10-10.

| # | Finding | Consequence |
|---|---|---|
| 1 | `sentiment/word_embeddings.db` is **3.9 GB**, 949,658 rows, table `embeddings(word TEXT, embedding TEXT)`. Each vector is a Python-list string parsed with `ast.literal_eval`. | Cannot be shipped. Slow to load. |
| 2 | The README and `RealDataSetup.py` say the embeddings are **GloVe 6B 300d**. The table is not: GloVe 6B has 400,000 words, this has 949,658; the vector for "the" starts `0.0897, 0.016, -0.0571, 0.0405` (GloVe 6B starts `0.0466, 0.2132`); the vocabulary is case-sensitive and contains control characters and junk like `,,I`. | Source identified as fastText wiki-news-300d-1M (see 1.1); the docs are wrong, not the data. |
| 3 | `RealDataSetup.py` rebuilds the embedding table from `glove.6B.300d.txt`. | Anyone following the README builds a **different** table, and the shipped weights give garbage with it. The weights are only valid with the exact original vectors. |
| 4 | The model lowercases everything, but 603k of the 949k rows (all those with uppercase letters) can never be looked up. | Only the lowercase rows are reachable; dead weight. |
| 5 | Wren's real vocabulary (every word in the 100,000 training reviews and the 2,000 eval reviews, after the project's own cleaning) is 75,203 distinct tokens. **37,498 are in the table**; the other 37,705 get a zero vector (1.35% of word occurrences, mostly glued `br` tokens such as `itbr`). | A restricted table of ~37.5k words reproduces training-time behaviour exactly. At float32 it is about 45 MB. |
| 6 | The training corpus (`test_data.db`, old pipeline) contains glued `<br />` tokens (`goodbr`, `thembr`). `RealDataSetup.py` and the eval set strip HTML first. | Train and eval text are preprocessed differently. Harmless to ship as-is, but record it in MODEL.md, and do not "fix" the tokeniser without retraining. |
| 7 | Weights are stored as Python-literal text in two SQLite files (15 MB) and are in `.gitignore` (`*.db`). `models/wren-1.0/MODEL.md` lists them, but they are **not in git**. | A fresh clone has no model. |
| 8 | `venv/pyvenv.cfg` points at `Python39` which no longer exists on this machine. | The venv is dead; it is gitignored, so just recreate it. |
| 9 | Everything imports flat (`import Comms as com`), and `tests/` and `evaluation/` use `sys.path.insert` hacks. `Comms.connect` looks for `.db` files next to the module. | Cannot be pip-installed as is. Phase 2. |

Not yet verified: whether the 37,498-word restriction changes accuracy on **unseen**
text. Unseen words that exist in the full table but not in the 37.5k set would drop from
a real vector to zeros. Phase 1 step 4 measures this.

---

## Phase 1. Decide and build the shipped artifacts

### 1.1 Embedding source (RESOLVED 2026-10-10)
The table is **fastText `wiki-news-300d-1M.vec`** (Mikolov et al. 2018, "Advances in Pre-Training Distributed Word Representations"), licence **CC BY-SA 3.0**. Verified against `Data/wiki-news-300d-1M.vec`:
- the `.vec` has 999,994 words; the table is an exact subset (all 949,658 words are in it; 50,336 `.vec` words were not imported, mostly numbers such as `29-28`);
- 3,000 randomly sampled shared words have identical vectors (max difference 0);
- none of Wren's training/eval tokens that are missing from the table exist in the `.vec`, and only 2 of the 50,336 omitted words (both `�`) could ever survive the cleaner. So the `.vec` and the table behave identically for this model.

Consequences:
- The export script (1.2) can read the `.vec` directly. The 3.9 GB DB is no longer needed for anything, and the build is reproducible from a public download.
- Credit fastText in MODEL.md and the README, link the licence, and release the embeddings file under CC BY-SA 3.0. Your code keeps its own licence.
- Fix the README, `RealDataSetup.py` (it builds from `glove.6B.300d.txt`, which would break the weights), and the "Provenance is unresolved" section of MODEL.md.

### 1.2 Export script `tools/export_model.py` (one-off, committed)
Reads `Data/wiki-news-300d-1M.vec` (embeddings) and the existing weight DBs (read-only) and writes two files:

- `wren-1.0-embeddings.npz`: `words` (array of str, sorted) and `vectors` (float32, shape N x 300).
  Vocabulary = tokens produced by `format_entry_data` over the training reviews **and** `Data/eval_test.jsonl`, that exist in the `.vec`. Skip tokens not in it (they stay zero vectors, as today).
- `wren-1.0-weights.npz`: arrays `conv{1,2,3}_kernel`, `conv{1,2,3}_bias`, `dense{1..6}_weights`, `dense{1..6}_biases`. Parse once with `ast.literal_eval` (the stored `repr` floats round-trip exactly), save as float64.

### 1.3 Prove nothing changed
Write `tools/check_export.py` (or a test) that loads the npz files and the DBs and asserts:
- the embedding vectors match the DB rows to within 1e-6 for 500 random words;
- softmax outputs match to within 1e-9 for the 2,000 eval reviews;
- accuracy reproduces **47.0% exact / 86.2% within one** (see `evaluation/results.json`).

### 1.4 Measure the vocabulary restriction on unseen text
Build a second npz whose vocabulary comes from the **training reviews only**, then run `evaluation/evaluate.py` with it against the eval set, versus the full 949k table. The eval set is the only data the model has not seen, and its words are deliberately excluded here so the test is fair.
- Difference within the CI (~±2.2 points): ship the 37.5k set.
- Noticeably worse: widen the vocabulary to all lowercase rows that are in the training or eval text **plus** a larger list (for example the 100k most common words from any large word list), and re-measure. The full 346k lowercase rows is about 415 MB at float32, too big; float16 halves it but is lossy, so check it.

### 1.5 Distribution
Upload both npz files (about 50 MB total) as **GitHub Release assets** for tag `v1.0`, and record each file's sha256 and size in `models/wren-1.0/MODEL.md` (replacing the `.db` rows). Do not commit them to git; 50 MB blobs stay in history forever.

Alternative, simpler for users but heavier for you: bundle them in the wheel (under PyPI's 100 MB default limit). That removes all download code. Choose this if you are publishing to PyPI; choose download-on-first-use if installing from GitHub.

---

## Phase 2. Restructure into a package

Target layout (use `git mv` so history follows):

```
pyproject.toml
src/wren/
    __init__.py        # exports Analyser, __version__
    model.py           # Analyser: load(), predict()
    text.py            # was WordVectorConversions.py (cleaning, padding)
    layers/            # was FilterBankLayer.py, NeuralLayer.py
    store.py           # locate / download / verify the npz artifacts
    cli.py             # Phase 3
    train/             # was main.py training loop, Comms.py, InitDatabases.py, RealDataSetup.py
tests/
evaluation/
```

Steps:
1. Move files, switch `import Comms as com` style to relative imports (`from . import ...`). Delete every `sys.path.insert` in `tests/` and `evaluation/`; run them against the installed package (`pip install -e .`).
2. `FilterBankLayer` and `NeuralLayer`: do not touch the maths. Add nothing but the ability to be loaded from arrays: set `.kernel`/`.bias` and `.weights`/`.biases` directly (they are plain attributes, see `restore_model` in `main.py`). Leave `fetchKernel`/`fetch_values`/`save` and the DB calls in place for training.
3. `text.py`: keep `format_entry_data`, `pad_matrix`, `WORD_LIMIT = 200`, pad value `0.001`, and the zero vector for unknown words **exactly** as they are. These are part of the model. Change `get_vector2` to look up in the loaded npz dict instead of `Comms.fetch_embedding`. (This also stops training needing the 3.9 GB DB.)
4. `model.py`:
   ```python
   class Analyser:
       def __init__(self, weights=None, embeddings=None): ...   # default: store.ensure("wren-1.0")
       def predict(self, texts) -> list[int]                    # 1..5; str in, int out
       def predict_proba(self, texts) -> np.ndarray             # (n, 5)
       def label(self, stars) -> str                            # "Very negative" ... (SENTIMENT_LABELS)
   ```
   Body is `forward_pass` from `main.py`, with layers built from `FILTER_BANK_CONFIG` and filled from the weights npz. Process in chunks of `EVALUATION_CHUNK = 50` (each review is a 200x300 matrix, so a long list will not fit in memory otherwise). Accept a single `str` as well as a list.
5. `store.py`: cache dir = `$WREN_HOME` or `~/.cache/wren` (stdlib only, no `platformdirs`). `ensure(name)` returns the paths, downloading with `urllib.request` if missing and **verifying sha256** against constants in this file. Abort and delete the file on mismatch. Skip this module entirely if you chose the bundle-in-wheel route.
6. Training code: keep it working but out of the default import path (`wren.train`). `confirm_and_back_up_model` writes next to the module; change it to a `--workdir` directory so a pip-installed package never writes into site-packages.
7. Update `evaluation/evaluate.py` (it imports `Comms` and `main` from a model directory) to use `Analyser`, then re-run it and confirm 47.0% / 86.2% again.

---

## Phase 3. CLI (`src/wren/cli.py`)

`argparse` with subcommands, entry point `wren = "wren.cli:main"`. No new dependency.

| Command | Behaviour |
|---|---|
| `wren predict TEXT...` | Prints the star rating and label. `-` reads one review per line from stdin. `--proba` prints all 5 probabilities. `--json` prints JSON lines. |
| `wren eval --data reviews.jsonl` | Wraps `evaluation/evaluate.py` logic: exact and within-one accuracy. |
| `wren train ...` | Wraps the training loop. Flags replace the prompts: `--batch-size`, `--batches`, `--workdir`, `--yes` (skip the overwrite confirmation). |
| `wren setup ...` | Replaces running `InitDatabases.py` and `RealDataSetup.py` by hand (training only). |
| `wren fetch` | Downloads and verifies the artifacts up front (only if using `store.py`). |

Rules: non-zero exit code on error, results on stdout, progress and warnings on stderr (so pipes work). Remove every `input()` call.

---

## Phase 4. Packaging and release

1. `pyproject.toml`: setuptools backend, `requires-python = ">=3.9"` (the venv was 3.9 and it is the only version tested; test 3.12 too), `dependencies = ["numpy>=1.24", "scipy>=1.10"]` (check whether `scipy` is still imported anywhere; if not, drop it), `[project.scripts] wren = "wren.cli:main"`. Add a `demo` extra later if you do the Gradio option.
2. Add `LICENSE` and the embeddings attribution from step 1.1. Check the Amazon Reviews 2023 terms before releasing weights trained on them (the dataset is published for research use); state the result in MODEL.md.
3. Tests that must pass in CI (`pytest`): the existing gradient tests, a new `test_predict.py` (one fixed sentence returns a fixed rating; `predict_proba` rows sum to 1; a long review and an empty string do not crash), and the Phase 1.3 equivalence check.
4. Fresh-machine test, the real acceptance test: on a clean venv with no repo, `pip install .`, then `wren predict "Does exactly what it says"` works and `python -c "from wren import Analyser"` works. Time the first call and the second.
5. Rewrite the README quick-start and the "Files" table in MODEL.md. Tag `v1.0`, attach the release assets.

---

## Decisions for you

1. **Distribution:** download-on-first-use (more code) or bundle in the wheel (no code, ~50 MB wheel)?
2. **Vocabulary size:** accept the 37.5k set if Phase 1.4 shows no loss?
3. ~~Unknown embedding source~~ resolved: fastText wiki-news-300d-1M.
4. **Keep training in the package** (`wren train`), or move it to a separate `dev/` folder and ship inference only? Inference-only is a noticeably smaller job.
