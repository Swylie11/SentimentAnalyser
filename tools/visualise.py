"""Quick LinkedIn-ready charts for Wren 1.0. Needs matplotlib (pip install matplotlib).

  python tools/visualise.py confusion            # from evaluation/results.json, no model run
  python tools/visualise.py recall               # per-star accuracy vs chance
  python tools/visualise.py training             # validation accuracy over the 15,000-batch run
  python tools/visualise.py review "Great, but broke after a week"   # star probabilities
  python tools/visualise.py words "Great, but broke after a week"    # which words drive the score

PNGs land in ./charts (override with --out).
"""
import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, GREY = "#2f6fb0", "#c9ced6"
STARS = ["1★", "2★", "3★", "4★", "5★"]


def results():
    with open(os.path.join(REPO, "evaluation", "results.json")) as f:
        return json.load(f)["test"]


def analyser():
    from wren import Analyser
    return Analyser()


def save(fig, args, name):
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(path)


def confusion(args):
    t = results()
    m = np.array(t["confusion"], dtype=float)
    m /= m.sum(axis=1, keepdims=True)  # rows = true star, so each row sums to 100%
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center",
                    color="white" if m[i, j] > 0.5 else "black")
    ax.set_xticks(range(5), STARS)
    ax.set_yticks(range(5), STARS)
    ax.set_xlabel("Wren predicted")
    ax.set_ylabel("Actual rating")
    ax.set_title(f"Where Wren gets it right and wrong\n{t['exact']:.1%} exact, "
                 f"{t['within_one']:.1%} within one star ({t['n']:,} unseen reviews)")
    save(fig, args, "confusion")


def recall(args):
    t = results()
    r = np.array(t["recall"]) * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(STARS, r, color=BLUE)
    ax.bar_label(bars, fmt="%.0f%%")
    ax.axhline(20, color="grey", ls="--")
    ax.text(4.45, 21, "random guessing", ha="right", color="grey", fontsize=9)
    ax.set_ylabel("Reviews of this rating predicted exactly (%)")
    ax.set_title("Wren is best at the extremes, weakest at 2★")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args, "recall")


def training(args):
    # Per-batch loss was never logged; the 200-batch quick-validation checkpoints are the curve.
    pat = re.compile(r"\[batch (\d+)\] quick validation: ([\d.]+)% exact, ([\d.]+)% within one")
    with open(os.path.join(REPO, "evaluation", "logs", "train_run_phase4.log"), encoding="utf-8") as f:
        rows = [(int(b), float(e), float(w)) for b, e, w in pat.findall(f.read())]
    b, exact, within = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(b, within, color=GREY, lw=2, label="within one star")
    ax.plot(b, exact, color=BLUE, lw=2.5, label="exact star")
    ax.axhline(20, color="grey", ls="--", lw=1)
    ax.text(b[-1], 21.5, "chance (exact)", ha="right", color="grey", fontsize=9)
    ax.set_xlabel("Training batches (32 reviews each)")
    ax.set_ylabel("Accuracy on 300 held-out reviews (%)")
    ax.set_title("Learning to read: stuck at chance, then it clicks")
    ax.legend(loc="center right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args, "training")


def review(args):
    p = analyser().predict_proba(args.text)[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(STARS, p * 100, color=[BLUE if i == p.argmax() else GREY for i in range(5)])
    ax.bar_label(bars, fmt="%.0f%%")
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"“{args.text[:70]}{'…' if len(args.text) > 70 else ''}”", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args, "review")


def words(args):
    """Occlusion: drop each word and see how the expected star rating moves."""
    a = analyser()
    toks = args.text.split()
    variants = [args.text] + [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
    expected = a.predict_proba(variants) @ np.arange(1, 6)
    delta = expected[0] - expected[1:]  # positive = word pulls the rating up
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(toks)), 4))
    ax.bar(range(len(toks)), delta, color=[BLUE if d >= 0 else "#c0504d" for d in delta])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(toks)), toks, rotation=45, ha="right")
    ax.set_ylabel("Effect on expected stars")
    ax.set_title(f"What moves Wren's rating (overall {expected[0]:.1f}★)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args, "words")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="charts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for fn in (confusion, recall, training):
        sub.add_parser(fn.__name__).set_defaults(fn=fn)
    for fn in (review, words):
        s = sub.add_parser(fn.__name__)
        s.add_argument("text")
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)
