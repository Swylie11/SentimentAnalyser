"""Exports Wren 1.5 from its training databases as two portable .npz files.

  wren-1.5-weights.npz     filter bank kernels/biases and dense weights/biases, float64
  wren-1.5-embeddings.npz  `words` and float32 `vectors` (N x 300), copied from the work
                           directory's word_embeddings.db, so the model sees exactly the
                           vocabulary it was trained and tested with

Then it runs the 2,000 multi-category test reviews through the exported files with the
shipped Analyser and fails unless every prediction matches evaluation/mcnemar.json, which
was produced from the databases. Run from the repo root:

    python tools/export_wren15.py [--workdir wren-work-big]
"""

import argparse
import ast
import hashlib
import json
import os
import sqlite3
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from wren import Analyser  # noqa: E402


def read_only(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default=os.path.join(REPO, "wren-work-big"))
    p.add_argument("--out", default=os.path.join(REPO, "models", "wren-1.5"))
    args = p.parse_args()
    weights_path = os.path.join(args.out, "wren-1.5-weights.npz")
    embeddings_path = os.path.join(args.out, "wren-1.5-embeddings.npz")

    arrays = {}
    with read_only(os.path.join(args.workdir, "convolution_layers.db")) as conn:
        for n, kernel, bias in conn.execute("SELECT LayerNum, kernel, bias FROM kernels ORDER BY LayerNum"):
            arrays[f"conv{n}_kernel"] = np.array(ast.literal_eval(kernel), dtype=np.float64)
            arrays[f"conv{n}_bias"] = np.array(ast.literal_eval(bias), dtype=np.float64)
    with read_only(os.path.join(args.workdir, "neuron_weights.db")) as conn:
        for n, weights, biases in conn.execute("SELECT LayerNum, weights, biases FROM weights ORDER BY LayerNum"):
            arrays[f"dense{n}_weights"] = np.array(ast.literal_eval(weights), dtype=np.float64)
            arrays[f"dense{n}_biases"] = np.array(ast.literal_eval(biases), dtype=np.float64)
    np.savez_compressed(weights_path, **arrays)
    print(f"weights: {sum(a.size for a in arrays.values()):,} parameters in {len(arrays)} arrays")

    with read_only(os.path.join(args.workdir, "word_embeddings.db")) as conn:
        rows = sorted(conn.execute("SELECT word, embedding FROM embeddings"))
    exact = np.array([ast.literal_eval(e) for _, e in rows], dtype=np.float64)
    vectors = exact.astype(np.float32)
    worst = float(np.abs(vectors - exact).max())
    assert worst < 1e-6, f"float32 rounding error {worst}"  # the .vec has 4 decimals
    np.savez_compressed(embeddings_path, words=np.array([w for w, _ in rows]), vectors=vectors)
    print(f"embeddings: {len(rows):,} words, max float32 error {worst:.1e}")

    with open(os.path.join(REPO, "evaluation", "mcnemar.json")) as f:
        reference = json.load(f)
    with open(reference["test_set"], encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    predictions = Analyser(weights_path, embeddings_path).predict(texts)
    mismatches = sum(a != b for a, b in zip(predictions, reference["pred15"]))
    accuracy = np.mean(np.array(predictions) == np.array(reference["labels"]))
    print(f"verify: {len(texts)} reviews, {mismatches} predictions differ from the databases, "
          f"exact accuracy {100 * accuracy:.1f}%")
    assert mismatches == 0, "exported model disagrees with the databases"

    for path in (weights_path, embeddings_path):
        print(f"{os.path.basename(path)}  {os.path.getsize(path):,} bytes  sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
