"""Three square promo videos for the Wren 1.5 release -> docs/charts/promo/*.mp4

    python evaluation/mcnemar.py      # first: writes evaluation/mcnemar.json
    python tools/promo_videos.py      # needs matplotlib and ffmpeg (or imageio-ffmpeg)

Every number on screen comes from mcnemar.json, runs/wren-1.5/seed_runs.json or a
real forward pass of Wren 1.5 (cached in docs/charts/promo/reviews.json).
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))  # runs from a checkout without pip install
OUT = os.path.join(REPO, "docs", "charts", "promo")
WORKDIR = os.path.join(REPO, "runs", "wren-1.5")
FPS = 30

SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8a8980", "#33332f"
WREN10, WREN15, SEED1 = "#3987e5", "#d95926", "#9085e9"   # categorical slots 1, 2, 7 (dark steps)
BOTH_RIGHT, BOTH_WRONG = "#6b6a63", "#2e2e2b"
HEAT = colors.LinearSegmentedColormap.from_list("heat", ["#262624", "#7a3416", WREN15, "#ffb38a"])
STARS = {1: "Very negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Very positive"}
FOOTER = "Wren 1.5  ·  a text CNN written from scratch in NumPy  ·  github.com/Swylie11/cadello"


def ease(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def frame(title_text=""):
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()
    ax.text(5, 95.5, "WREN 1.5", fontsize=15, weight="bold", color=WREN15)
    ax.text(5, 2.5, FOOTER, fontsize=11, color=MUTED)
    title = ax.text(5, 89.5, title_text, fontsize=34, weight="bold", color=INK)
    return fig, ax, title


def save(fig, update, seconds, name):
    anim = animation.FuncAnimation(fig, update, frames=int(seconds * FPS))
    path = os.path.join(OUT, name)
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=6000, extra_args=["-pix_fmt", "yuv420p"]),
              savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    print(path)


# ---- data ---------------------------------------------------------------------------

def load_wren15():
    from cadello.wren import text
    from cadello.wren.train import comms, loop
    comms.set_db_dir(WORKDIR)
    layers = loop.build_layers()
    for layer in layers[0]:
        layer.fetchKernel()
    for layer in layers[1] + [layers[2]]:
        layer.fetch_values()
    return loop, text, layers


def pick_reviews():
    """Short test reviews Wren 1.5 rates correctly, one per star, with the real
    probabilities, the 300 pooled filter features and how hard the filters fired on
    each word (sum of each filter's max activation over the words in its window)."""
    cache = os.path.join(OUT, "reviews.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    loop, text, (conv, dense, out) = load_wren15()
    with open(os.path.join(REPO, "Data", "eval_test_multi.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    short = [r for r in records if 7 <= len(text.format_entry_data([r["text"]])[0].split()) <= 16
             and len(r["text"]) <= 95]
    picked = []
    for star in (5, 1, 3, 4):
        for r in short:
            if int(r["rating"]) != star:
                continue
            probs = loop.forward_pass(conv, dense, out, [r["text"]])[0]
            if int(np.argmax(probs)) + 1 != star:
                continue
            words = text.format_entry_data([r["text"]])[0].split()
            heat = np.zeros(len(words))
            for layer in conv:
                for f_index, pos in enumerate(layer.argmax_positions[0]):
                    heat[pos:pos + layer.region_size] += layer.output[0, f_index]
            features = np.concatenate([layer.output[0] for layer in conv])
            picked.append({"star": star, "words": words, "probs": probs.tolist(),
                           "heat": (heat / heat.max()).tolist(),
                           "features": (features / features.max()).tolist()})
            break
    with open(cache, "w") as f:
        json.dump(picked, f)
    return picked


# ---- video 1: the swarm -------------------------------------------------------------

def video_swarm(m):
    labels, p10, p15 = (np.array(m[k]) for k in ("labels", "pred10", "pred15"))
    s = m["all"]
    r10, r15 = p10 == labels, p15 == labels
    group = np.select([r10 & r15, r10 & ~r15, ~r10 & r15], [0, 1, 2], 3)
    n = len(group)
    rng = np.random.default_rng(1)
    palette = np.array([colors.to_rgb(c) for c in (BOTH_RIGHT, WREN10, WREN15, BOTH_WRONG)])

    # Positions: swirl -> shuffled grid -> grid sorted by outcome -> two towers.
    cell, cols = 1.4, 50
    def grid(slot):
        return np.c_[15.7 + (slot % cols) * cell, 76 - (slot // cols) * cell]
    radius = 6 + 28 * np.sqrt(rng.random(n))
    angle0 = rng.random(n) * 2 * np.pi
    shuffled = grid(rng.permutation(n))
    sorted_ = grid(np.argsort(np.argsort(group, kind="stable")))
    towers = sorted_.copy()
    for g, x0 in ((1, 22), (2, 64)):
        idx = np.flatnonzero(group == g)
        k = np.arange(len(idx))
        towers[idx] = np.c_[x0 + (k % 10) * cell, 13 + (k // 10) * cell]
    rest = (group == 0) | (group == 3)
    towers[rest] = sorted_[rest] + rng.normal(0, 30, (rest.sum(), 2))

    delay = rng.random(n) * 0.6
    fig, ax, title = frame()
    sub = ax.text(5, 85, "", fontsize=17, color=INK2)
    dots = ax.scatter(*sorted_.T, s=70, marker="s", linewidths=0)
    counts = [ax.text(0, 0, "", fontsize=30, weight="bold", color=INK, ha="center") for _ in range(2)]
    names = [ax.text(x, 8, name, fontsize=17, color=INK2, ha="center")
             for x, name in ((28.3, "only Wren 1.0 right"), (70.3, "only Wren 1.5 right"))]
    big = ax.text(50, 71, "", fontsize=64, weight="bold", color=INK, ha="center", va="center")
    small = ax.text(50, 63.5, "", fontsize=19, color=INK2, ha="center", va="center")

    def update(i):
        t = i / FPS
        if t < 2.2:
            spin = angle0 + t * (1.6 - radius / 40)
            pos = np.c_[50 + radius * np.cos(spin), 46 + radius * np.sin(spin) * 0.9]
            if t > 1.4:
                k = ease((t - 1.4) / 0.8 - delay)[:, None]
                pos = pos * (1 - k) + shuffled * k
            rgb, alpha = np.tile(colors.to_rgb(INK2), (n, 1)), np.full(n, 0.8)
            title.set_text("2,000 reviews it never saw")
            sub.set_text("Five product categories, one square each")
        elif t < 4.6:
            pos = shuffled
            k = ease((t - 2.4) / 1.2 - delay)[:, None]
            rgb = np.tile(colors.to_rgb(INK2), (n, 1)) * (1 - k) + palette[group] * k
            alpha = np.ones(n)
            title.set_text("Two models read every one")
            sub.set_text("Grey: both agree.  Blue: only Wren 1.0 got it.  Orange: only Wren 1.5 got it.")
        elif t < 6.4:
            k = ease((t - 4.6) / 1.4 - delay)[:, None]
            pos, rgb, alpha = shuffled * (1 - k) + sorted_ * k, palette[group], np.ones(n)
            title.set_text("Sort by who was right")
        else:
            k = ease((t - 6.4) / 1.6 - delay)[:, None]
            pos, rgb = sorted_ * (1 - k) + towers * k, palette[group]
            alpha = np.where(rest, 1 - k[:, 0], 1.0)
            title.set_text("Only the disagreements count")
            sub.set_text("McNemar's test: when exactly one model is right, which one is it?")
            for txt, g, x in zip(counts, (1, 2), (28.3, 70.3)):
                c = int(s["only_wren10"] if g == 1 else s["only_wren15"])
                height = 13 + np.ceil(c / 10) * cell
                txt.set_position((x, height + 1.5))
                txt.set_text(f"{round(c * ease((t - 7) / 1.5)):,}" if t > 7 else "")
        for txt in names:
            txt.set_alpha(float(ease((t - 7.4) / 0.6)))
        if t > 9:
            k = float(ease((t - 9) / 0.5))
            big.set_text(f"p = {s['p_exact']:.3f}")
            big.set_fontsize(64 + 60 * (1 - k))
            big.set_alpha(k)
            small.set_text(f"{100 * s['wren15_exact']:.1f}% vs {100 * s['wren10_exact']:.1f}% exact stars  ·  "
                           "a real improvement, not luck")
            small.set_alpha(float(ease((t - 9.5) / 0.6)))
        dots.set_offsets(pos)
        dots.set_facecolors(np.c_[rgb, alpha])

    save(fig, update, 12.5, "wren15_swarm.mp4")


# ---- video 2: how it reads ------------------------------------------------------------

def video_reading(reviews):
    fig, ax, title = frame("How Wren 1.5 reads a review")
    per = 3.6
    word_texts = [ax.text(0, 0, "", fontsize=28, family="DejaVu Sans Mono", color=INK, va="center")
                  for _ in range(20)]
    window_label = ax.text(5, 59, "", fontsize=15, color=INK2)
    feat_label = ax.text(5, 54.5, "300 phrase detectors, strongest response each", fontsize=15, color=INK2)
    fx, fy = np.meshgrid(np.arange(30), np.arange(10))
    feats = ax.scatter(6 + fx.ravel() * 2.2, 51.5 - fy.ravel() * 2.2, s=190, marker="s", linewidths=0)
    bars = [ax.add_patch(plt.Rectangle((28, 26 - j * 3.6), 0, 2.6, color=GRID)) for j in range(5)]
    bar_names = [ax.text(5, 27.3 - j * 3.6, f"{j + 1}★  {STARS[j + 1]}", fontsize=16, color=INK2, va="center")
                 for j in range(5)]
    pct = [ax.text(0, 27.3 - j * 3.6, "", fontsize=16, color=INK, va="center") for j in range(5)]
    verdict = ax.text(50, 6.5, "", fontsize=26, weight="bold", color=INK, ha="center")

    def layout(words):
        x, y, spots = 5, 80, []
        for w in words:
            width = 2.17 * (len(w) + 1)
            if x + width > 96:
                x, y = 5, y - 6.5
            spots.append((x, y))
            x += width
        return spots

    def update(i):
        t = i / FPS
        r = reviews[min(int(t // per), len(reviews) - 1)]
        lt = t - int(t // per) * per
        words, heat, probs = r["words"], np.array(r["heat"]), np.array(r["probs"])
        spots = layout(words)
        sweep = lt / 1.5 * (len(words) + 2)                      # window start position
        size = 3 + int(lt / 0.5) % 3                             # cycles 3, 4, 5 word windows
        for j, txt in enumerate(word_texts):
            if j >= len(words):
                txt.set_text("")
                continue
            txt.set_text(words[j])
            txt.set_position(spots[j])
            appear = float(ease(lt * 4 - j * 0.15))
            glow = heat[j] * float(ease((lt - 1.5) / 0.6))
            in_window = lt < 1.5 and int(sweep) <= j < int(sweep) + size
            txt.set_color(HEAT(0.35 + 0.65 * glow) if glow > 0 else INK)
            txt.set_alpha(appear)
            txt.set_bbox({"facecolor": "#3a3a36", "edgecolor": "none", "pad": 3} if in_window else
                          {"facecolor": "none", "edgecolor": "none"})
        window_label.set_text(f"Sliding {size}-word windows across the text" if lt < 1.5 else
                              "Brighter words = phrases the filters locked onto")
        shown = np.arange(300) < 300 * float(ease(lt / 1.6))
        feats.set_facecolors(np.where(shown[:, None], HEAT(np.array(r["features"])), colors.to_rgba(GRID)))
        g = float(ease((lt - 1.6) / 0.8))
        best = int(np.argmax(probs))
        for j in range(5):
            bars[j].set_width(60 * probs[j] * g)
            bars[j].set_color(WREN15 if j == best else "#5c5b55")
            pct[j].set_position((29 + 60 * probs[j] * g, 27.3 - j * 3.6))
            pct[j].set_text(f"{100 * probs[j] * g:.0f}%" if g > 0 else "")
            bar_names[j].set_color(INK if j == best and g > 0.5 else INK2)
        verdict.set_text(f"Wren 1.5: {best + 1}★   ·   the reviewer gave {r['star']}★  ✓" if lt > 2.5 else "")
        verdict.set_alpha(float(ease((lt - 2.5) / 0.4)))

    save(fig, update, per * len(reviews), "wren15_reading.mp4")


# ---- video 3: training and what changed -------------------------------------------------

def video_training(runs, m):
    fig, ax, title = frame("575,500 reviews. 45,000 batches.")
    sub = ax.text(5, 85, "Two seeds racing, trained from scratch on a CPU", fontsize=17, color=INK2)
    acc_ax = fig.add_axes([0.1, 0.5, 0.82, 0.3], facecolor=SURFACE)
    loss_ax = fig.add_axes([0.1, 0.12, 0.82, 0.28], facecolor=SURFACE)
    for a, ylabel in ((acc_ax, "Validation accuracy"), (loss_ax, "Training loss")):
        a.set_xlim(0, 45000)
        a.tick_params(colors=MUTED, labelsize=12)
        a.grid(axis="y", color=GRID, lw=1)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.set_ylabel(ylabel, color=INK2, fontsize=13)
        a.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    acc_ax.set_ylim(40, 58)
    acc_ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    loss_ax.set_ylim(0.9, 1.7)
    curves = [np.array(r["curve"]) for r in runs]
    losses = [np.array(r["loss_curve"]) for r in runs]
    acc_lines = [acc_ax.plot([], [], color=c, lw=2.5)[0] for c in (WREN15, SEED1)]
    heads = [acc_ax.scatter([], [], s=90, color=c, edgecolors=SURFACE, linewidths=2, zorder=3)
             for c in (WREN15, SEED1)]
    loss_lines = [loss_ax.plot([], [], color=c, lw=1.5, alpha=0.9)[0] for c in (WREN15, SEED1)]
    tags = [acc_ax.text(0, 0, "", fontsize=14, weight="bold", color=INK) for _ in runs]
    counter = ax.text(95, 81.3, "", fontsize=15, color=INK, ha="right")

    # Second act: recall per star, Wren 1.0 -> Wren 1.5, same test set.
    labels = np.array(m["labels"])
    recall = {k: [np.mean(np.array(m[k])[labels == s] == s) for s in range(1, 6)] for k in ("pred10", "pred15")}
    bar_ax = fig.add_axes([0.1, 0.12, 0.82, 0.62], facecolor=SURFACE)
    bar_ax.set_visible(False)

    def update(i):
        t = i / FPS
        if t < 7.5:
            x = 45000 * float(ease(t / 6.5))
            for j in range(len(runs)):
                c, l = curves[j], losses[j]
                cv, lv = c[c[:, 0] <= x], l[l[:, 0] <= x]
                acc_lines[j].set_data(cv[:, 0], cv[:, 1] * 100)
                loss_lines[j].set_data(lv[:, 0], lv[:, 1])
                if len(cv):
                    heads[j].set_offsets([[cv[-1, 0], cv[-1, 1] * 100]])
                    tags[j].set_position((cv[-1, 0] + 600, cv[-1, 1] * 100 + (1.2 if j == 0 else -2.4)))
                    tags[j].set_text(f"seed {runs[j]['seed']}  {cv[-1, 1] * 100:.1f}%")
            counter.set_text(f"batch {x:,.0f}   ·   {x * 32 / 1e6:.2f}M reviews read")
            return
        acc_ax.set_visible(False)
        loss_ax.set_visible(False)
        counter.set_text("")
        title.set_text("Biggest jump: 2-star reviews")
        sub.set_text("Recall per star rating on the same 2,000 unseen reviews")
        bar_ax.clear()
        bar_ax.set_visible(True)
        bar_ax.set_facecolor(SURFACE)
        k = float(ease((t - 7.6) / 1.2))
        k15 = float(ease((t - 8.6) / 1.4))
        xs = np.arange(5)
        bar_ax.bar(xs - 0.2, np.array(recall["pred10"]) * 100 * k, width=0.38, color=WREN10, label="Wren 1.0")
        bar_ax.bar(xs + 0.2, np.array(recall["pred15"]) * 100 * k15, width=0.38, color=WREN15, label="Wren 1.5")
        for x0, a, b in zip(xs, recall["pred10"], recall["pred15"]):
            if k15 > 0.95:
                d = (b - a) * 100
                bar_ax.text(x0 + 0.2, b * 100 + 1.5, f"{d:+.0f}", ha="center", fontsize=16, weight="bold",
                            color=INK)
        bar_ax.set_ylim(0, 90)
        bar_ax.tick_params(colors=MUTED, labelsize=12)
        bar_ax.set_xticks(xs, [f"{s}★" for s in range(1, 6)], color=INK, fontsize=18)
        bar_ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        bar_ax.grid(axis="y", color=GRID, lw=1)
        bar_ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            bar_ax.spines[side].set_visible(False)
        bar_ax.spines["bottom"].set_color(GRID)
        bar_ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=15, labelcolor=INK)

    save(fig, update, 13, "wren15_training.mp4")


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    with open(os.path.join(REPO, "evaluation", "mcnemar.json")) as f:
        m = json.load(f)
    with open(os.path.join(WORKDIR, "seed_runs.json")) as f:
        runs = json.load(f)["runs"]
    which = sys.argv[1:] or ["swarm", "reading", "training"]
    if "swarm" in which:
        video_swarm(m)
    if "reading" in which:
        video_reading(pick_reviews())
    if "training" in which:
        video_training(runs, m)


if __name__ == "__main__":
    assert ease(-1) == 0 and ease(2) == 1 and ease(0.5) == 0.5
    main()
