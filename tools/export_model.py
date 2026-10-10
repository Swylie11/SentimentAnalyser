"""Exports Wren 1.0 as two portable .npz files, replacing the SQLite databases.

  wren-1.0-weights.npz     filter bank kernels/biases and dense weights/biases, float64
  wren-1.0-embeddings.npz  `words` and float32 `vectors` (N x 300) for the model's vocabulary

The embeddings come straight from the fastText file the model was trained with
(wiki-news-300d-1M.vec), restricted to the words Wren can actually look up: every token
the model's own cleaner produces from the training reviews and the eval set. Words not
in the file stay zero vectors, exactly as in training. The 3.9 GB word_embeddings.db is
not read.

After writing, the script runs the eval reviews through both the databases and the new
files and fails unless the probabilities agree. Run from the repo root:

    python tools/export_model.py [--sentiment-dir DIR_WITH_THE_DB_FILES]
"""

import argparse
import ast
import hashlib
import json
import os
import sqlite3
import sys

import numpy as np

from wren import Analyser
from wren import text as wvc
from wren.config import FILTER_BANK_CONFIG, NUM_DENSE
from wren.layers.filter_bank import FilterBankLayer
from wren.layers.neural import NeuralLayer
from wren.train import comms as com
from wren.train import loop as model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_only(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def tokens_of(texts):
    """The tokens the model would look up for these reviews (first WORD_LIMIT words)."""
    vocab = set()
    for start in range(0, len(texts), 2000):
        for cleaned in wvc.format_entry_data(texts[start:start + 2000]):
            vocab.update(cleaned.split()[:wvc.WORD_LIMIT])
    return vocab


def read_eval(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r["text"] for r in rows], [int(r["rating"]) for r in rows]


def export_embeddings(vec_path, vocab, out_path):
    found = {}
    with open(vec_path, encoding="utf-8") as f:
        header = f.readline().split()
        assert int(header[1]) == wvc.EMBEDDING_DIM, f"expected 300d, header says {header}"
        for line in f:
            word, _, rest = line.partition(" ")
            if word in vocab:
                found[word] = np.array(rest.split(), dtype=np.float64)

    words = sorted(found)
    exact = np.stack([found[w] for w in words])
    vectors = exact.astype(np.float32)
    # The .vec has 4 decimals, so float32 should lose nothing that matters.
    worst = float(np.abs(vectors.astype(np.float64) - exact).max())
    assert worst < 1e-6, f"float32 rounding error {worst}"

    np.savez_compressed(out_path, words=np.array(words), vectors=vectors)
    print(f"embeddings: {len(words):,}/{len(vocab):,} vocabulary tokens found in {os.path.basename(vec_path)}"
          f" (the rest stay zero vectors); max float32 error {worst:.1e}")


def export_weights(sentiment_dir, out_path):
    arrays = {}
    with read_only(os.path.join(sentiment_dir, "convolution_layers.db")) as conn:
        for layer_num, _, _ in FILTER_BANK_CONFIG:
            kernel, bias = conn.execute(
                "SELECT kernel, bias FROM kernels WHERE LayerNum = ?", (layer_num,)).fetchone()
            arrays[f"conv{layer_num}_kernel"] = np.array(ast.literal_eval(kernel), dtype=np.float64)
            arrays[f"conv{layer_num}_bias"] = np.array(ast.literal_eval(bias), dtype=np.float64)
    with read_only(os.path.join(sentiment_dir, "neuron_weights.db")) as conn:
        for layer_num in range(1, NUM_DENSE + 1):
            weights, biases = conn.execute(
                "SELECT weights, biases FROM weights WHERE LayerNum = ?", (layer_num,)).fetchone()
            arrays[f"dense{layer_num}_weights"] = np.array(ast.literal_eval(weights), dtype=np.float64)
            arrays[f"dense{layer_num}_biases"] = np.array(ast.literal_eval(biases), dtype=np.float64)

    np.savez_compressed(out_path, **arrays)
    print(f"weights: {sum(a.size for a in arrays.values()):,} parameters in {len(arrays)} arrays")


def layers_from_database():
    conv = [FilterBankLayer(n, r, f) for n, r, f in FILTER_BANK_CONFIG]
    dense = [NeuralLayer(n) for n in range(1, NUM_DENSE + 1)]
    for layer in conv:
        layer.fetchKernel()
    for layer in dense:
        layer.fetch_values()
    return conv, dense[:-1], dense[-1]


def database_probabilities(texts):
    layers = layers_from_database()
    return np.concatenate([model.forward_pass(*layers, texts[i:i + 50]) for i in range(0, len(texts), 50)])


def verify(weights_path, embeddings_path, texts, labels):
    """Same reviews through the databases and through the exported files (via the shipped
    Analyser, which never touches a database); the outputs must match."""
    reference = database_probabilities(texts)
    exported = Analyser(weights_path, embeddings_path).predict_proba(texts)

    worst = float(np.abs(reference - exported).max())
    labels = np.asarray(labels)
    accuracy = float(np.mean(np.argmax(exported, axis=1) + 1 == labels))
    print(f"verify: {len(texts)} reviews, max probability difference {worst:.2e}, "
          f"exported exact accuracy {100 * accuracy:.1f}%")
    assert worst < 1e-5, "exported model disagrees with the databases"
    assert (np.argmax(reference, axis=1) == np.argmax(exported, axis=1)).all(), "predictions differ"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vec", default=os.path.join(REPO, "Data", "wiki-news-300d-1M.vec"))
    p.add_argument("--eval", default=os.path.join(REPO, "Data", "eval_test.jsonl"))
    p.add_argument("--sentiment-dir", default=os.path.join(REPO, "sentiment"))
    p.add_argument("--out", default=os.path.join(REPO, "models", "wren-1.0"))
    p.add_argument("--check", type=int, default=2000, help="eval reviews to verify on (0 skips)")
    args = p.parse_args()

    com.set_db_dir(args.sentiment_dir)
    eval_texts, eval_labels = read_eval(args.eval)
    with read_only(os.path.join(args.sentiment_dir, "test_data.db")) as conn:
        train_texts = [row[0] for row in conn.execute("SELECT text FROM test_dataset4")]
    vocab = tokens_of(train_texts) | tokens_of(eval_texts)
    print(f"{len(train_texts):,} training + {len(eval_texts):,} eval reviews -> {len(vocab):,} distinct tokens")

    os.makedirs(args.out, exist_ok=True)
    weights_path = os.path.join(args.out, "wren-1.0-weights.npz")
    embeddings_path = os.path.join(args.out, "wren-1.0-embeddings.npz")
    export_embeddings(args.vec, vocab, embeddings_path)
    export_weights(args.sentiment_dir, weights_path)

    if args.check:
        verify(weights_path, embeddings_path, eval_texts[:args.check], eval_labels[:args.check])

    for path in (weights_path, embeddings_path):
        print(f"{os.path.basename(path)}  {os.path.getsize(path):,} bytes  sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
