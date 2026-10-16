"""McNemar test: Wren 1.0 vs Wren 1.5 on the same fresh test set (exact star match).

Also checks the test set for overlap with Wren 1.5's training reviews, because
the test set is checked against both models' training data.

    python evaluation/mcnemar.py                        # Wren 1.5 from wren-work-big
    python evaluation/mcnemar.py --wren15 wren-work-multi
"""

import argparse
import importlib.util
import json
import math
import os
import sqlite3
import sys

import numpy as np

from build_test_set import normalise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mcnemar(b, c):
    """Exact two-sided McNemar p-value (binomial on the discordant pairs) and the
    continuity-corrected chi-squared. b = only model A right, c = only model B right."""
    n = b + c
    if n == 0:
        return 1.0, 0.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail), (abs(b - c) - 1) ** 2 / n


def load_wren15(workdir):
    package = os.path.join(REPO, "src", "wren-1.5")
    spec = importlib.util.spec_from_file_location(
        "wren15", os.path.join(package, "__init__.py"), submodule_search_locations=[package])
    sys.modules["wren15"] = module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from wren15.train import comms, loop
    comms.set_db_dir(workdir)
    conv, dense, out = loop.build_layers()
    for layer in conv:
        layer.fetchKernel()
    for layer in dense + [out]:
        layer.fetch_values()
    return lambda texts: np.asarray(loop.predict_texts(conv, dense, out, texts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(REPO, "Data", "eval_test_multi.jsonl"))
    parser.add_argument("--wren15", default=os.path.join(REPO, "wren-work-big"))
    parser.add_argument("--out", default=os.path.join(REPO, "evaluation", "mcnemar.json"))
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    labels = np.array([int(r["rating"]) for r in records])
    texts = [r["text"] for r in records]

    trained = set()
    for workdir in (os.path.join(REPO, "sentiment"), args.wren15):
        with sqlite3.connect(f"file:{os.path.join(workdir, 'test_data.db')}?mode=ro", uri=True) as conn:
            trained |= {normalise(t) for (t,) in conn.execute("SELECT text FROM test_dataset4")}
    leaked = np.array([normalise(t) in trained for t in texts])

    sys.path.insert(0, os.path.join(REPO, "src"))
    from wren import Analyser  # Wren 1.0, released weights from ~/.cache/wren
    pred10 = np.asarray(Analyser().predict(texts))
    pred15 = load_wren15(args.wren15)(texts)

    def compare(mask):
        r10, r15 = pred10[mask] == labels[mask], pred15[mask] == labels[mask]
        b, c = int(np.sum(r10 & ~r15)), int(np.sum(~r10 & r15))
        p, chi2 = mcnemar(b, c)
        return {"n": int(mask.sum()), "wren10_exact": float(r10.mean()), "wren15_exact": float(r15.mean()),
                "both_right": int(np.sum(r10 & r15)), "only_wren10": b, "only_wren15": c,
                "both_wrong": int(np.sum(~r10 & ~r15)), "p_exact": p, "chi2_cc": chi2}

    result = {"test_set": args.data, "wren15_workdir": args.wren15, "leaked": int(leaked.sum()),
              "all": compare(np.ones(len(labels), bool)), "clean": compare(~leaked),
              "labels": labels.tolist(), "pred10": pred10.tolist(), "pred15": pred15.tolist(),
              "leaked_mask": leaked.tolist()}
    with open(args.out, "w") as f:
        json.dump(result, f)

    print(f"Test reviews also in either model's training data: {result['leaked']}")
    for name in ("all", "clean"):
        r = result[name]
        print(f"\n[{name}] n={r['n']}  Wren 1.0 {100 * r['wren10_exact']:.1f}%  Wren 1.5 {100 * r['wren15_exact']:.1f}%")
        print(f"  both right {r['both_right']}  only 1.0 {r['only_wren10']}  "
              f"only 1.5 {r['only_wren15']}  both wrong {r['both_wrong']}")
        print(f"  McNemar exact p = {r['p_exact']:.3g}   chi2 (cc) = {r['chi2_cc']:.2f}")


if __name__ == "__main__":
    assert mcnemar(0, 0) == (1.0, 0.0) and abs(mcnemar(5, 5)[0] - 1.0) < 1e-12
    assert abs(mcnemar(1, 9)[0] - 0.021484375) < 1e-12  # 2 * (1 + 10) / 1024
    main()
