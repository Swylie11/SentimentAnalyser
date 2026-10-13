"""Quick LinkedIn-ready charts for Wren 1.0. Needs matplotlib (pip install matplotlib).

  python tools/visualise.py confusion            # from evaluation/results.json, no model run
  python tools/visualise.py recall               # per-star accuracy vs chance
  python tools/visualise.py training             # validation accuracy over the 15,000-batch run
  python tools/visualise.py seeds                # multi-seed run, from <workdir>/seed_runs.json
  python tools/visualise.py loss                 # animated training loss (MP4, or GIF without ffmpeg)
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


def seeds(args):
    """Multi-seed run: every seed's validation curve, the kept one highlighted."""
    path = args.file or os.path.join(os.environ.get("WREN_WORKDIR", "wren-work"), "seed_runs.json")
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    runs = log["runs"]
    final = np.array([r["val_exact"] for r in runs]) * 100
    std = final.std(ddof=1) if len(final) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(8, 4.2))  # 1600 x 840 at dpi 200: LinkedIn's 1.91:1
    curves = [np.array(r["curve"], dtype=float) for r in runs]
    n = min(len(c) for c in curves)
    if n:  # the spread between seeds at each checkpoint is the luck in one run
        x = curves[0][:n, 0]
        acc = np.stack([c[:n, 1] for c in curves]) * 100
        ax.fill_between(x, acc.min(axis=0), acc.max(axis=0), color=BLUE, alpha=0.08, lw=0)
    for r, c in sorted(zip(runs, curves), key=lambda rc: rc[0]["seed"] == log["champion_seed"]):
        kept = r["seed"] == log["champion_seed"]
        ax.plot(c[:, 0], c[:, 1] * 100, color=BLUE if kept else GREY, lw=2.5 if kept else 1.2, zorder=3 if kept else 2)
        ax.plot(c[-1, 0], c[-1, 1] * 100, "o", ms=6 if kept else 4, color=BLUE if kept else GREY, zorder=4)
        if kept:
            ax.annotate(f"seed {r['seed']}, kept", (c[-1, 0], c[-1, 1] * 100), xytext=(8, 0),
                        textcoords="offset points", va="center", color=BLUE, fontweight="bold")
    ax.axhline(20, color="grey", ls="--", lw=1)
    ax.text(ax.get_xlim()[1], 21.5, "chance", ha="right", color="grey", fontsize=9)
    ax.set_xlabel(f"Training batches ({log['batch_size']} reviews each)")
    ax.set_ylabel("Accuracy on 300 held-out reviews (%)")
    fig.suptitle(f"Same model, same data, {len(runs)} different random starts",
                 x=0.125, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(f"Final accuracy {final.mean():.1f} ± {std:.1f}% across seeds "
                 f"({log.get('validation_size', 2000):,} held-out reviews). "
                 f"Only the best run's weights are saved.", loc="left", fontsize=10, color="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.02)
    save(fig, args, "seeds")


def loss(args):
    """Animated training loss for one seed of a Wren 1.5 multi-seed run, for video.

    MP4 if ffmpeg is on PATH, otherwise GIF. Also writes the final frame as a PNG."""
    from matplotlib import animation
    path = args.file or os.path.join(os.environ.get("WREN_WORKDIR", "wren-work"), "seed_runs.json")
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    seed = log["champion_seed"] if args.seed is None else args.seed
    run = next(r for r in log["runs"] if r["seed"] == seed)
    if not run.get("loss_curve"):
        raise SystemExit(f"seed {seed} has no loss_curve: it was trained before loss logging was added")
    x, y = np.array(run["loss_curve"], dtype=float).T
    k = max(1, len(y) // 30)  # trailing mean, so the line the viewer follows isn't batch noise
    smooth = np.array([y[max(0, i - k + 1):i + 1].mean() for i in range(len(y))])
    chance = np.log(5)  # cross-entropy of guessing 20% for every star

    fig, ax = plt.subplots(figsize=(8, 4.5))  # 16:9, 1280 x 720 at dpi 160
    ax.set_xlim(0, x[-1] * 1.04)
    ax.set_ylim(min(y.min(), smooth.min()) * 0.95, max(y.max(), chance) * 1.05)
    ax.axhline(chance, color="grey", ls="--", lw=1)
    ax.text(x[-1] * 1.04, chance, "random guessing ", ha="right", va="bottom", color="grey", fontsize=9)
    raw_line, = ax.plot([], [], color=GREY, lw=1)
    line, = ax.plot([], [], color=BLUE, lw=2.5)
    head, = ax.plot([], [], "o", color=BLUE, ms=8)
    label = ax.annotate("", (0, 0), xytext=(10, 0), textcoords="offset points", va="center",
                        color="#333", fontweight="bold")
    ax.set_xlabel(f"Training batches ({log['batch_size']} reviews each)")
    ax.set_ylabel("Training loss (cross-entropy)")
    fig.suptitle("Wren 1.5 learning: loss falls as it reads more reviews",
                 x=0.125, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(f"Seed {seed}. Grey: mean loss every {int(x[0])} batches. "
                 f"Blue: smoothed. Lower is better.", loc="left", fontsize=10, color="#555")
    ax.spines[["top", "right"]].set_visible(False)

    draw_frames = args.fps * args.seconds

    def draw(frame):
        n = len(y) if frame >= draw_frames else max(1, round((frame + 1) / draw_frames * len(y)))
        raw_line.set_data(x[:n], y[:n])
        line.set_data(x[:n], smooth[:n])
        head.set_data([x[n - 1]], [smooth[n - 1]])
        label.xy = (x[n - 1], smooth[n - 1])
        label.set_text(f"{smooth[n - 1]:.2f}")
        return raw_line, line, head, label

    # Hold the finished graph for two seconds at the end of the clip.
    anim = animation.FuncAnimation(fig, draw, frames=draw_frames + 2 * args.fps, blit=True)
    writer = "ffmpeg" if animation.writers.is_available("ffmpeg") else "pillow"
    os.makedirs(args.out, exist_ok=True)
    video = os.path.join(args.out, f"loss_seed{seed}." + ("mp4" if writer == "ffmpeg" else "gif"))
    anim.save(video, writer=writer, fps=args.fps, dpi=160)
    print(video)
    save(fig, args, f"loss_seed{seed}")  # the last frame drawn is the finished graph


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
    s = sub.add_parser("seeds")
    s.add_argument("--file", help="seed_runs.json (default $WREN_WORKDIR or ./wren-work)")
    s.set_defaults(fn=seeds)
    s = sub.add_parser("loss")
    s.add_argument("--file", help="seed_runs.json (default $WREN_WORKDIR or ./wren-work)")
    s.add_argument("--seed", type=int, help="which run to plot (default: the kept champion)")
    s.add_argument("--seconds", type=int, default=10, help="length of the drawing animation")
    s.add_argument("--fps", type=int, default=30)
    s.set_defaults(fn=loss)
    for fn in (review, words):
        s = sub.add_parser(fn.__name__)
        s.add_argument("text")
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)
