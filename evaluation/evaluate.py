"""Honest evaluation of the trained sentiment model. Read only: no database is
modified, and the script checks that for you.

Reports, with 95% confidence intervals:
  1. the fresh test set built by build_test_set.py
  2. a slice of the training reviews (so the train/test gap is explicit)
  3. the original validation set, split into the 300 reviews that main.py used to
     pick its checkpoint and the other 1,700 that it did not
  4. leakage: exact and near-duplicate overlap between test/validation and training
  5. trivial baselines on the test set, and how many test words have no embedding

Usage:
    python evaluation/evaluate.py
"""

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import Counter

import numpy as np

from build_test_set import normalise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_CLASSES = 5
MODEL_FILES = ("neuron_weights.db", "convolution_layers.db")
ALL_FILES = MODEL_FILES + ("test_data.db", "word_embeddings.db")


# ---- pure maths (no model needed) -------------------------------------------------

def wilson(successes, n, z=1.96):
    """95% Wilson score interval for a proportion. Returns (low, high)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def two_proportion_z(k1, n1, k2, n2):
    """z statistic for 'these two accuracies differ'. |z| < 1.96 is within noise."""
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return (k1 / n1 - k2 / n2) / se if se else 0.0


def metrics(predictions, labels):
    """Accuracy figures, per-class recall, confusion matrix (rows = true star,
    columns = predicted star) and a collapse flag."""
    predictions = np.asarray(predictions, dtype=int)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)

    exact_k = int(np.sum(predictions == labels))
    within_k = int(np.sum(np.abs(predictions - labels) <= 1))

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(labels, predictions):
        confusion[true - 1, pred - 1] += 1
    support = confusion.sum(axis=1)
    recall = [float(confusion[i, i] / support[i]) if support[i] else None for i in range(NUM_CLASSES)]

    return {
        "n": n,
        "exact": exact_k / n if n else 0.0,
        "exact_ci": wilson(exact_k, n),
        "exact_k": exact_k,
        "within_one": within_k / n if n else 0.0,
        "within_one_ci": wilson(within_k, n),
        "recall": recall,
        "confusion": confusion.tolist(),
        "prediction_counts": np.bincount(predictions, minlength=NUM_CLASSES + 1)[1:].tolist(),
        "label_counts": np.bincount(labels, minlength=NUM_CLASSES + 1)[1:].tolist(),
        "collapsed": bool(n) and len(set(predictions.tolist())) == 1,
    }


def baselines(labels):
    """Uniform random (expected values) and the best constant prediction."""
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    stars = np.arange(1, NUM_CLASSES + 1)

    # Uniform random: each star equally likely, so the expected score is the mean
    # over the labels of (stars that would be right) / 5.
    random_within = float(np.mean([np.sum(np.abs(stars - y) <= 1) for y in labels]) / NUM_CLASSES)

    constants = {int(c): (float(np.mean(labels == c)), float(np.mean(np.abs(labels - c) <= 1)))
                 for c in stars}
    best = max(constants, key=lambda c: constants[c][0])
    return {"uniform_exact": 1 / NUM_CLASSES, "uniform_within_one": random_within,
            "best_constant": best, "best_constant_exact": constants[best][0],
            "best_constant_within_one": constants[best][1], "n": n}


# ---- near-duplicate detection ------------------------------------------------------

SHINGLE = 5


def shingle_set(text):
    """Hashes of every run of SHINGLE consecutive words."""
    tokens = normalise(text).split()
    if len(tokens) <= SHINGLE:
        return {hash(" ".join(tokens))}
    return {hash(" ".join(tokens[i:i + SHINGLE])) for i in range(len(tokens) - SHINGLE + 1)}


def best_overlap(queries, corpus):
    """For each query text, the highest Jaccard similarity of 5-word shingles with
    any text in `corpus`. The (small) queries are indexed and the (large) corpus is
    streamed past them, so memory stays small."""
    query_sets = [shingle_set(q) for q in queries]
    index = {}
    for i, shingles in enumerate(query_sets):
        for h in shingles:
            index.setdefault(h, []).append(i)

    best = [0.0] * len(queries)
    for text in corpus:
        candidate = shingle_set(text)
        shared = Counter()
        for h in candidate:
            for i in index.get(h, ()):
                shared[i] += 1
        for i, count in shared.items():
            jaccard = count / (len(query_sets[i]) + len(candidate) - count)
            if jaccard > best[i]:
                best[i] = jaccard
    return best


# ---- model and data access ----------------------------------------------------------

def file_fingerprint(directory):
    """sha256 of every database the evaluation touches."""
    out = {}
    for name in ALL_FILES:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            out[name] = digest.hexdigest()
    return out


def load_model(model_dir):
    """Imports the model code from model_dir and loads the stored weights."""
    sys.path.insert(0, model_dir)
    import Comms as com
    import main as model
    import WordVectorConversions as wvc
    from FilterBankLayer import FilterBankLayer
    from InitDatabases import FILTER_BANK_CONFIG
    from NeuralLayer import NeuralLayer

    conv = [FilterBankLayer(n, r, f) for n, r, f in FILTER_BANK_CONFIG]
    dense = [NeuralLayer(n) for n in range(1, 6)]
    out = NeuralLayer(6)
    for layer in conv:
        layer.fetchKernel()
    for layer in dense + [out]:
        layer.fetch_values()
    return com, model, wvc, (conv, dense, out)


def predict(model, layers, texts):
    conv, dense, out = layers
    predictions = []
    for start in range(0, len(texts), model.EVALUATION_CHUNK):
        chunk = texts[start:start + model.EVALUATION_CHUNK]
        probs = model.forward_pass(conv, dense, out, chunk)
        predictions.extend((np.argmax(probs, axis=1) + 1).tolist())
    return np.asarray(predictions)


def fetch_rows(com, ids):
    """[(rating, text)] for review ids, in chunks."""
    rows = []
    for start in range(0, len(ids), 500):
        rows.extend(com.fetch_batch(ids[start:start + 500]))
    return rows


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [int(r["rating"]) for r in records], [r["text"] for r in records]


def zero_vector_rate(wvc, texts, vocabulary):
    """Fraction of word tokens (as the model tokenises them) with no embedding."""
    tokens = [w for cleaned in wvc.format_entry_data(texts) for w in cleaned.split()]
    return sum(w not in vocabulary for w in tokens) / max(len(tokens), 1)


# ---- reporting -----------------------------------------------------------------------

def pct(x):
    return f"{100 * x:5.1f}%"


def show(title, m):
    lo, hi = m["exact_ci"]
    wlo, whi = m["within_one_ci"]
    print(f"\n{title} (n={m['n']})")
    print(f"  Exact accuracy   : {pct(m['exact'])}   95% CI [{pct(lo)}, {pct(hi)}]")
    print(f"  Within one star  : {pct(m['within_one'])}   95% CI [{pct(wlo)}, {pct(whi)}]")
    print("  Recall per star  : " + "  ".join(
        f"{i + 1}*: {pct(r) if r is not None else '  n/a'}" for i, r in enumerate(m["recall"])))
    print("  Predictions      : " + "  ".join(
        f"{i + 1}*: {c:>5}" for i, c in enumerate(m["prediction_counts"])))
    print("  Confusion (rows = true star, columns = predicted star):")
    for i, row in enumerate(m["confusion"]):
        print(f"    {i + 1}* | " + " ".join(f"{c:>5}" for c in row))
    if m["collapsed"]:
        print("  WARNING: every prediction is the same class (model has collapsed).")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=os.path.join(REPO, "sentiment"),
                        help="Directory holding the model code and the four .db files")
    parser.add_argument("--test-set", default=os.path.join(REPO, "Data", "eval_test.jsonl"))
    parser.add_argument("--train-slice", type=int, default=2000,
                        help="Training reviews to score for the train/test gap")
    parser.add_argument("--selection-size", type=int, default=300,
                        help="Leading validation reviews main.py used to pick its checkpoint")
    parser.add_argument("--near-dup-threshold", type=float, default=0.8,
                        help="Jaccard similarity of 5-word shingles that counts as a near duplicate")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the training slice")
    parser.add_argument("--out", default=os.path.join(REPO, "evaluation", "results.json"))
    args = parser.parse_args()

    before = file_fingerprint(args.model_dir)
    com, model, wvc, layers = load_model(args.model_dir)
    start = time.time()

    # Same split main.py trained with.
    train_ids, val_ids = com.split_ids(com.fetch_all_review_ids(), model.VALIDATION_SIZE, model.DATA_SEED)
    train_sample = random.Random(args.seed).sample(train_ids, min(args.train_slice, len(train_ids)))

    test_labels, test_texts = read_jsonl(args.test_set)
    val_rows = fetch_rows(com, val_ids)
    val_labels, val_texts = [r[0] for r in val_rows], [r[1] for r in val_rows]
    slice_rows = fetch_rows(com, train_sample)
    slice_labels, slice_texts = [r[0] for r in slice_rows], [r[1] for r in slice_rows]

    results = {"test_set": args.test_set, "model_dir": args.model_dir}
    groups = [
        ("test", "FRESH TEST SET (never seen, never used for selection)", test_labels, test_texts),
        ("train_slice", "TRAINING SLICE (reviews the model trained on)", slice_labels, slice_texts),
        ("validation_selection", f"VALIDATION, first {args.selection_size} "
         f"(used by main.py to pick the checkpoint)",
         val_labels[:args.selection_size], val_texts[:args.selection_size]),
        ("validation_rest", f"VALIDATION, remaining reviews (not used for selection)",
         val_labels[args.selection_size:], val_texts[args.selection_size:]),
    ]
    for key, title, labels, texts in groups:
        if not texts:
            continue
        results[key] = metrics(predict(model, layers, texts), labels)
        show(title, results[key])

    # Gap and selection inflation, stated in terms of noise.
    print("\n== Gaps ==")
    if "test" in results and "train_slice" in results:
        t, s = results["test"], results["train_slice"]
        z = two_proportion_z(s["exact_k"], s["n"], t["exact_k"], t["n"])
        results["train_test_gap"] = {"exact_gap": s["exact"] - t["exact"], "z": z}
        verdict = "outside noise" if abs(z) >= 1.96 else "within noise"
        print(f"  Train minus test exact accuracy: {100 * (s['exact'] - t['exact']):+.1f} points "
              f"(z={z:.2f}, {verdict})")
    if "validation_selection" in results and "validation_rest" in results:
        a, b = results["validation_selection"], results["validation_rest"]
        z = two_proportion_z(a["exact_k"], a["n"], b["exact_k"], b["n"])
        results["selection_inflation"] = {"exact_gap": a["exact"] - b["exact"], "z": z}
        verdict = "outside noise" if abs(z) >= 1.96 else "within noise"
        print(f"  Selection slice minus rest of validation: {100 * (a['exact'] - b['exact']):+.1f} points "
              f"(z={z:.2f}, {verdict})")

    # Leakage.
    print("\n== Leakage against the training reviews ==")
    train_texts_iter = (r[1] for start_i in range(0, len(train_ids), 500)
                        for r in com.fetch_batch(train_ids[start_i:start_i + 500]))
    train_normalised = {normalise(t) for t in train_texts_iter}
    leak = {}
    for name, texts in (("test", test_texts), ("validation", val_texts)):
        exact = sum(normalise(t) in train_normalised for t in texts)
        # Streams the training texts a second time; cheap next to the model passes.
        stream = (r[1] for start_i in range(0, len(train_ids), 500)
                  for r in com.fetch_batch(train_ids[start_i:start_i + 500]))
        near = sum(b >= args.near_dup_threshold for b in best_overlap(texts, stream))
        leak[name] = {"n": len(texts), "exact_duplicates": exact, "near_duplicates": near}
        print(f"  {name:<10}: {exact} exact and {near} near duplicates (Jaccard >= "
              f"{args.near_dup_threshold}) of {len(texts)} reviews")
    results["leakage"] = leak

    # Baselines and embedding coverage.
    base = baselines(test_labels)
    results["baselines"] = base
    print("\n== Baselines on the test set ==")
    print(f"  Uniform random      : exact {pct(base['uniform_exact'])}, "
          f"within one {pct(base['uniform_within_one'])}")
    print(f"  Always predict {base['best_constant']}*   : exact {pct(base['best_constant_exact'])}, "
          f"within one {pct(base['best_constant_within_one'])}")

    with sqlite3.connect(f"file:{os.path.join(args.model_dir, 'word_embeddings.db')}?mode=ro",
                         uri=True) as conn:
        vocabulary = {row[0] for row in conn.execute("SELECT word FROM embeddings")}
    results["zero_vector_rate"] = {
        "test": zero_vector_rate(wvc, test_texts, vocabulary),
        "train_slice": zero_vector_rate(wvc, slice_texts, vocabulary)}
    print(f"\n== Words with no embedding (read as a zero vector) ==\n"
          f"  test {pct(results['zero_vector_rate']['test'])}, "
          f"train slice {pct(results['zero_vector_rate']['train_slice'])}")
    print("  (The embedding table only holds words from the training corpus, so a higher\n"
          "   test rate is a real handicap the model also has when deployed.)")

    # Prove nothing was written, and tie the result to the exact checkpoint.
    after = file_fingerprint(args.model_dir)
    results["databases_unchanged"] = before == after
    results["checkpoint"] = {
        name: {"sha256": before[name],
               "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(
                   os.path.getmtime(os.path.join(args.model_dir, name))))}
        for name in MODEL_FILES if name in before}
    results["seconds"] = round(time.time() - start, 1)
    print(f"\nDatabases unchanged by this run: {results['databases_unchanged']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
