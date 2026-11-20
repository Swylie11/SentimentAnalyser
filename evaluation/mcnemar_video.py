"""Animated McNemar result from evaluation/mcnemar.json -> docs/charts/mcnemar.mp4
(or .gif when no ffmpeg is found).

    python evaluation/mcnemar_video.py
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30

SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8a8980", "#33332f"
WREN10, WREN15 = "#3987e5", "#d95926"           # categorical slots 1 and 2, dark steps
BOTH_RIGHT, BOTH_WRONG = "#6b6a63", "#2e2e2b"   # neutral: agreement is not the story


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    with open(os.path.join(REPO, "evaluation", "mcnemar.json")) as f:
        r = json.load(f)
    s = r["all"]
    labels, p10, p15 = (np.array(r[k]) for k in ("labels", "pred10", "pred15"))
    r10, r15 = p10 == labels, p15 == labels

    # One square per review, sorted so each outcome forms a block: 50 x 40 waffle.
    group = np.select([r10 & r15, r10 & ~r15, ~r10 & r15], [0, 1, 2], 3)
    order = np.argsort(group, kind="stable")
    group = group[order]
    cols = 50
    xs, ys = np.arange(len(group)) % cols, -(np.arange(len(group)) // cols)
    colours = np.array([BOTH_RIGHT, WREN10, WREN15, BOTH_WRONG])
    reveal = np.random.default_rng(0).permutation(len(group))  # random fill-in order

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=SURFACE)
    title = fig.text(0.05, 0.92, "", fontsize=26, weight="bold", color=INK)
    sub = fig.text(0.05, 0.875, "", fontsize=14, color=INK2)
    foot = fig.text(0.05, 0.03, f"{s['n']:,} unseen reviews · 5 product categories · exact star rating",
                    fontsize=11, color=MUTED)

    # Scene 1: accuracy bars with 95% CIs.
    ax1 = fig.add_axes([0.17, 0.14, 0.78, 0.66], facecolor=SURFACE)
    # Scene 2: waffle of every review, then the discordant tug-of-war.
    ax2 = fig.add_axes([0.05, 0.12, 0.5, 0.7], facecolor=SURFACE)
    ax3 = fig.add_axes([0.6, 0.2, 0.36, 0.55], facecolor=SURFACE)
    for ax in (ax1, ax2, ax3):
        ax.set_visible(False)

    accs = [s["wren10_exact"], s["wren15_exact"]]
    cis = [wilson(a, s["n"]) for a in accs]

    def scene_bars(t):
        ax1.clear()
        ax1.set_visible(True)
        ax1.set_facecolor(SURFACE)
        g = ease(t / 1.6)
        for i, (a, ci, c, name) in enumerate(zip(accs, cis, (WREN10, WREN15), ("Wren 1.0", "Wren 1.5"))):
            ax1.barh(i, a * g * 100, height=0.5, color=c)
            if t > 1.6:
                k = ease((t - 1.6) / 0.6)
                ax1.plot([ci[0] * 100, ci[1] * 100], [i, i], color=INK, lw=2, alpha=k)
                ax1.plot([ci[0] * 100] * 2, [i - .12, i + .12], color=INK, lw=2, alpha=k)
                ax1.plot([ci[1] * 100] * 2, [i - .12, i + .12], color=INK, lw=2, alpha=k)
            ax1.text(-1, i, name, ha="right", va="center", fontsize=18, color=INK)
            ax1.text(a * g * 100 + 4.5 if t > 1.6 else a * g * 100 + 1, i, f"{a * g * 100:.1f}%",
                     va="center", fontsize=20, weight="bold", color=INK)
        ax1.axvline(20, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=0)
        ax1.text(20.3, 1.42, "random guessing", color=MUTED, fontsize=11)
        ax1.set_xlim(0, 65)
        ax1.set_ylim(1.6, -0.6)
        ax1.set_yticks([])
        ax1.tick_params(colors=MUTED, labelsize=11)
        ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        for side in ("top", "right", "left"):
            ax1.spines[side].set_visible(False)
        ax1.spines["bottom"].set_color(GRID)
        title.set_text("Wren 1.5 vs Wren 1.0")
        sub.set_text("Exact-star accuracy on the same test set, with 95% confidence intervals"
                     if t > 1.6 else "Exact-star accuracy on the same test set")

    def scene_waffle(t):
        ax1.set_visible(False)
        ax2.clear()
        ax2.set_visible(True)
        ax2.set_axis_off()
        shown = np.zeros(len(group), bool)
        shown[reveal[: int(len(group) * ease(t / 2.5))]] = True
        dim = 0.25 if t > 3.3 else 1.0
        alpha = np.where(np.isin(group, (1, 2)), 1.0, 1 - (1 - dim) * ease((t - 3.3) / 0.6))
        face = [(*plt.matplotlib.colors.to_rgb(colours[g]), a if v else 0)
                for g, a, v in zip(group, alpha, shown)]
        ax2.scatter(xs, ys, s=22, marker="s", c=face, linewidths=0)
        ax2.set_xlim(-1, cols)
        ax2.set_ylim(ys.min() - 1, 1)
        ax2.set_aspect("equal")

        ax3.clear()
        ax3.set_visible(True)
        ax3.set_axis_off()
        rows = [("Both right", s["both_right"], BOTH_RIGHT), ("Only Wren 1.0 right", s["only_wren10"], WREN10),
                ("Only Wren 1.5 right", s["only_wren15"], WREN15), ("Both wrong", s["both_wrong"], BOTH_WRONG)]
        for i, (name, count, c) in enumerate(rows):
            focus = i in (1, 2)
            a = 1.0 if focus or t < 3.3 else 0.35
            ax3.add_patch(plt.Rectangle((0, -i * 1.2 - .25), .5, .5, color=c, alpha=a))
            ax3.text(.9, -i * 1.2, name, va="center", fontsize=15, color=INK if focus else INK2, alpha=a)
            ax3.text(9.5, -i * 1.2, f"{round(count * ease(t / 2.5)):,}", va="center", ha="right",
                     fontsize=17, weight="bold", color=INK, alpha=a)
        if t > 4.2:
            k = ease((t - 4.2) / 0.8)
            ax3.text(0, -5.4, f"p = {s['p_exact']:.4f}", fontsize=40, weight="bold", color=INK, alpha=k)
            ax3.text(0, -6.3, "McNemar exact test, two-sided", fontsize=13, color=INK2, alpha=k)
        ax3.set_xlim(0, 10)
        ax3.set_ylim(-6.8, 0.8)

        title.set_text("Every review, both models")
        sub.set_text("Only the disagreements count: when exactly one model is right, which one is it?"
                     if t > 3.3 else "One square per test review")

    timeline = [(scene_bars, 4.5), (scene_waffle, 7.5)]
    frames = [(fn, i / FPS) for fn, length in timeline for i in range(int(length * FPS))]

    anim = animation.FuncAnimation(fig, lambda i: frames[i][0](frames[i][1]), frames=len(frames))
    out = os.path.join(REPO, "docs", "charts", "mcnemar")
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    if animation.FFMpegWriter.isAvailable():
        anim.save(out + ".mp4", writer=animation.FFMpegWriter(fps=FPS, bitrate=4000),
                  savefig_kwargs={"facecolor": SURFACE})
        print(out + ".mp4")
    else:
        anim.save(out + ".gif", writer=animation.PillowWriter(fps=FPS), savefig_kwargs={"facecolor": SURFACE})
        print(out + ".gif")
    frames[-1][0](frames[-1][1])
    fig.savefig(out + ".png", facecolor=SURFACE)


if __name__ == "__main__":
    assert ease(-1) == 0 and ease(2) == 1 and ease(.5) == .5
    main()
