"""Checks for build_test_set.py and evaluate.py. Needs no real data and never
touches the real databases: the end-to-end check builds a synthetic model in a
temporary copy of sentiment/.

    python evaluation/tests/test_evaluation.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(REPO, "evaluation")
sys.path.insert(0, EVAL)

import numpy as np  # noqa: E402

import evaluate as ev  # noqa: E402
from build_test_set import normalise, select_fresh  # noqa: E402


def check(name, condition):
    print(("pass  " if condition else "FAIL  ") + name)
    if not condition:
        check.failed += 1


check.failed = 0


def test_wilson():
    low, high = ev.wilson(50, 100)
    check("wilson(50/100) is [0.404, 0.596]", abs(low - 0.4038) < 1e-3 and abs(high - 0.5962) < 1e-3)
    check("wilson of an empty sample is (0, 0)", ev.wilson(0, 0) == (0.0, 0.0))


def test_metrics():
    labels = [1, 2, 3, 4, 5, 5]
    predictions = [1, 3, 3, 1, 5, 4]
    m = ev.metrics(predictions, labels)
    check("exact accuracy is 3/6", m["exact_k"] == 3)
    check("within-one accuracy is 5/6", abs(m["within_one"] - 5 / 6) < 1e-12)
    check("confusion rows are true stars", m["confusion"][1] == [0, 0, 1, 0, 0]
          and m["confusion"][4] == [0, 0, 0, 1, 1])
    check("recall of 5 star is 1/2", m["recall"][4] == 0.5)
    check("a constant predictor is flagged as collapsed", ev.metrics([4] * 6, labels)["collapsed"])
    check("a varied predictor is not", not m["collapsed"])


def test_baselines():
    b = ev.baselines([1, 1, 1, 5, 5])
    check("best constant is the commonest star", b["best_constant"] == 1 and b["best_constant_exact"] == 0.6)
    check("constant 1 is within one star of 1s and 2s only", abs(b["best_constant_within_one"] - 0.6) < 1e-12)


def test_near_duplicates():
    base = "the quick brown fox jumps over the lazy dog and keeps running far away"
    best = ev.best_overlap([base, "something completely different about software licences and updates"],
                           [base + " today", "an unrelated review of a spreadsheet application"])
    check("a near copy is found", best[0] > 0.8)
    check("an unrelated review is not", best[1] < 0.2)


def test_select_fresh():
    reservoirs = {1: ["Great, product!", "Awful thing", "awful   THING", "Fresh one", "Another new"],
                  2: ["Fine item", "Meh"]}
    known = {normalise("great product")}
    chosen, dropped = select_fresh(reservoirs, known, per_class=3, seed=0)
    kept = [normalise(t) for t in chosen[1]]
    check("a review already in training is excluded", "great product" not in kept)
    check("a repeat within the test set is excluded", kept.count("awful thing") <= 1)
    check("counts are capped at per_class", len(chosen[1]) == 3 and len(chosen[2]) == 2)
    check("drops are counted", dropped["in_training"] == 1 and dropped["repeated"] == 1)


def sha_all(directory):
    return ev.file_fingerprint(directory)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "sentiment")
        os.makedirs(model_dir)
        for name in os.listdir(os.path.join(REPO, "sentiment")):
            if name.endswith(".py"):
                shutil.copy(os.path.join(REPO, "sentiment", name), model_dir)

        run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp, capture_output=True, text=True)
        built = run(os.path.join(model_dir, "InitDatabases.py"), "--synthetic", "300")
        check("synthetic model builds", built.returncode == 0)

        # A test set drawn from the synthetic vocabulary.
        from_vocab = {1: "awful broken useless refund", 3: "fine okay average adequate",
                      5: "excellent brilliant perfect superb"}
        test_path = os.path.join(tmp, "test.jsonl")
        with open(test_path, "w") as f:
            for i in range(30):
                star = [1, 3, 5][i % 3]
                f.write(json.dumps({"rating": star, "text": f"{from_vocab[star]} the and it {i}"}) + "\n")

        fingerprint = sha_all(model_dir)
        out_path = os.path.join(tmp, "results.json")
        result = run(os.path.join(EVAL, "evaluate.py"), "--model-dir", model_dir, "--test-set", test_path,
                     "--train-slice", "50", "--selection-size", "30", "--out", out_path)
        check("evaluate.py runs", result.returncode == 0)
        if result.returncode != 0:
            print(result.stdout[-1500:], result.stderr[-1500:])
            return
        check("databases are byte-identical afterwards", sha_all(model_dir) == fingerprint)
        results = json.load(open(out_path))
        check("results report the databases unchanged", results["databases_unchanged"] is True)
        check("results cover test, train slice and validation",
              all(k in results for k in ("test", "train_slice", "validation_selection", "leakage")))
        check("test set size is read correctly", results["test"]["n"] == 30)


if __name__ == "__main__":
    for t in (test_wilson, test_metrics, test_baselines, test_near_duplicates, test_select_fresh,
              test_end_to_end):
        t()
    print("\nAll checks passed." if not check.failed else f"\n{check.failed} check(s) FAILED.")
    sys.exit(1 if check.failed else 0)
