"""Training dashboard: a live progress bar while training, and `wren watch` beside it.

While training, the terminal shows a banner, then one progress bar per seed with a
spinner that keeps turning during validation. Notable events print above the bars.
Without a terminal (piped through tee, or under the tests) it prints plain lines
instead. Every message is also appended to a log file, if one is given.

Training also rewrites <workdir>/status.json as it goes. `wren watch` renders that file
as a live stats panel in a second window. The watcher only reads, so it can't affect
training. For runs from before status.json existed, it reads the training log instead.
"""

import glob
import json
import os
import re
import sys
import time

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeRemainingColumn)
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

STATUS_FILE = "status.json"

# The watcher calls a run stale when status.json hasn't changed for this long. Training
# writes it every LOSS_EVERY batches, about every 30 seconds.
STALE_AFTER = 120

BANNER = r"""
 __        ______  _____ _   _     _   ____
 \ \      / /  _ \| ____| \ | |   / | | ___|
  \ \ /\ / /| |_) |  _| |  \| |   | | |___ \
   \ V  V / |  _ <| |___| |\  |   | |_ ___) |
    \_/\_/  |_| \_\_____|_| \_|   |_(_)____/
"""


def _can_print(characters):
    try:
        characters.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# A real Windows console is UTF-8, but output through a pipe or an old code page may be
# cp1252, which has no block or braille characters. Fall back to ASCII there.
FANCY = _can_print("▁▂▃▄▅▆▇█⠋·")
SPARK = "▁▂▃▄▅▆▇█" if FANCY else " .:-=+*#"
SPINNER = "dots" if FANCY else "line"
DOT = "·" if FANCY else "-"


class Dashboard:
    """ What training shows and records. Dashboard() with no arguments prints plain
    lines and writes no files, which is what run_once gets when called directly. """

    def __init__(self, workdir=None, log_path=None, live=None):
        self.console = Console()
        self.live = self.console.is_terminal if live is None else live
        self.log_path = log_path
        self.status_path = os.path.join(workdir, STATUS_FILE) if workdir else None
        self.status = {"model": "Wren 1.5", "started": time.time(), "phase": "starting",
                       "runs": [], "loss": [], "quick": []}
        self.progress = None
        self.task = None

    # Messages

    def say(self, message="", style=None):
        """ Always shown, and logged. """
        if self.live:
            self.console.print(message, style=style, markup=False, highlight=False, soft_wrap=True)
        else:
            print(message, flush=True)
        self._log(message)

    def note(self, message):
        """ Routine detail such as the loss every 50 batches: logged, but shown only in
        plain mode, where there's no live bar already showing it. """
        if not self.live:
            print(message, flush=True)
        self._log(message)

    def _log(self, message):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def banner(self, config, watch_command):
        self.status["config"] = config
        summary = "   ".join(f"{k} {v}" for k, v in config.items())
        if self.live:
            self.console.print(Panel(Text(BANNER.strip("\n"), style="bold cyan"),
                                     border_style="magenta", expand=False))
            self.console.print(summary, style="magenta", highlight=False)
            # soft_wrap: a hard line break in the middle would break copy and paste.
            self.console.print(Text.assemble(("Live stats in another window:  ", "dim"),
                                             (watch_command, "bold green")), soft_wrap=True)
        else:
            print(f"Wren 1.5 training: {summary}", flush=True)
            print(f"Live stats in another window:  {watch_command}", flush=True)
        self._log(f"Wren 1.5 training: {summary}")
        self.write_status()

    # Progress

    def start(self, seeds, batches, batch_size):
        self.status.update(seeds=list(seeds), batches=batches, batch_size=batch_size,
                           started=time.time(), phase="training")
        self.write_status()

    def start_seed(self, seed):
        seeds = self.status.get("seeds") or [seed]
        position = seeds.index(seed) + 1 if seed in seeds else 1
        self.status.update(seed=seed, batch=0, loss=[], quick=[], phase="training")
        self.write_status()
        if not self.live:
            return
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(SPINNER, style="magenta"),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(complete_style="cyan", finished_style="green"),
                TaskProgressColumn(), MofNCompleteColumn(),
                TextColumn("loss [yellow]{task.fields[loss]}"),
                TextColumn("[dim]{task.fields[phase]}"), TimeRemainingColumn(),
                console=self.console)
            self.progress.start()
        label = f"Seed {seed} ({position}/{len(seeds)})" if seed is not None else "Training"
        self.task = self.progress.add_task(label, total=self.status.get("batches"), loss="-", phase="")

    def batch(self, number):
        self.status["batch"] = number
        if self.task is not None:
            self.progress.update(self.task, completed=number)

    def loss(self, number, value):
        self.status["loss"].append([number, value])
        self.note(f"  [batch {number}] training loss: {value:.4f}")
        if self.task is not None:
            self.progress.update(self.task, loss=f"{value:.3f}")
        self.write_status()

    def validating(self, what):
        """ Marks the start of an evaluation, so the bar says why it has stopped moving. """
        self.status["phase"] = "validating"
        self.write_status()
        if self.task is not None:
            self.progress.update(self.task, phase=f"validating {what}")

    def quick(self, number, exact, within_one):
        best = max((q[1] for q in self.status["quick"]), default=-1.0)
        self.status["quick"].append([number, exact, within_one])
        self.status["phase"] = "training"
        message = (f"  [batch {number}] quick validation: "
                   f"{100 * exact:.2f}% exact, {100 * within_one:.2f}% within one")
        if exact > best and self.live:
            self.say(f"Seed {self.status.get('seed')}: new best quick validation "
                     f"{100 * exact:.2f}% at batch {number:,}", style="green")
        self.note(message)
        if self.task is not None:
            self.progress.update(self.task, phase="")
        self.write_status()

    def seed_done(self, record, champion_seed, champion_exact):
        self.status["runs"].append({k: record[k] for k in
                                    ("seed", "val_exact", "val_within_one", "seconds", "saved")})
        self.status.update(champion_seed=champion_seed, champion_exact=champion_exact)
        if self.task is not None:
            self.progress.update(self.task, phase=f"done: {100 * record['val_exact']:.2f}%")
        self.write_status()

    def finish(self, phase="finished"):
        """ Stops the bars. phase is "stopped" if training crashed or was interrupted. """
        self.status["phase"] = phase
        self.write_status()
        if self.progress is not None:
            self.progress.stop()
            self.progress = self.task = None

    def write_status(self):
        if not self.status_path:
            return
        self.status["updated"] = time.time()
        temporary = self.status_path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(self.status, f)
            os.replace(temporary, self.status_path)
        except OSError:
            # Windows refuses the replace while the watcher has the file open. Skipping
            # one update is harmless, and the dashboard must never stop training.
            pass


# The watcher

def read_status(workdir):
    """ status.json if training wrote one, otherwise the newest training log, else None. """
    path = os.path.join(workdir, STATUS_FILE)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None  # caught mid-replace; the next poll gets it
    logs = sorted(glob.glob(os.path.join(workdir, "train_*.log")))
    return parse_log(logs[-1]) if logs else None


def parse_log(path):
    """ Rebuilds a status dict from a training log, for runs that predate status.json. """
    status = {"model": "Wren 1.5", "source": os.path.basename(path), "phase": "starting",
              "runs": [], "loss": [], "quick": [], "seed": None, "batch": 0}
    stamp = re.search(r"(\d{8}-\d{6})", os.path.basename(path))
    status["started"] = (time.mktime(time.strptime(stamp.group(1), "%Y%m%d-%H%M%S"))
                         if stamp else os.path.getmtime(path))
    status["updated"] = os.path.getmtime(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    in_validation = False
    run = {}
    for line in lines:
        if m := re.match(r"== Training: seeds ([-\d ]+), (\d+) batches of (\d+)", line):
            status.update(seeds=[int(s) for s in m[1].split()], batches=int(m[2]),
                          batch_size=int(m[3]), phase="training")
        elif m := re.match(r"=== Seed (-?\d+) ===", line):
            run = {"seed": int(m[1]), "saved": False}
            status.update(seed=run["seed"], batch=0, loss=[], quick=[], phase="training")
        elif m := re.match(r"Batch: (\d+)", line):
            status["batch"] = int(m[1])
        elif m := re.search(r"\[batch (\d+)\] training loss: ([\d.]+)", line):
            status["loss"].append([int(m[1]), float(m[2])])
        elif m := re.search(r"\[batch (\d+)\] quick validation: ([\d.]+)% exact, ([\d.]+)% within", line):
            status["quick"].append([int(m[1]), float(m[2]) / 100, float(m[3]) / 100])
        elif m := re.match(r"Total time elapsed: ([\d.]+)s", line):
            run["seconds"] = float(m[1])
            status["phase"] = "validating"
        elif line.startswith("Validation ("):
            in_validation = True
        elif in_validation and (m := re.match(r"\s+Accuracy\s+: ([\d.]+)%", line)):
            run["val_exact"] = float(m[1]) / 100
        elif in_validation and (m := re.match(r"\s+Within one star\s+: ([\d.]+)%", line)):
            run["val_within_one"] = float(m[1]) / 100
            status["runs"].append(run)
            in_validation = False
            status["phase"] = "training"
        elif "New champion" in line and status["runs"]:
            status["runs"][-1]["saved"] = True
            status.update(champion_seed=status["runs"][-1]["seed"],
                          champion_exact=status["runs"][-1]["val_exact"])
        elif line.startswith("Multi-seed summary"):
            status["phase"] = "finished"
    return status


def sparkline(values, width=40):
    values = np.asarray(values[-width:], dtype=float)
    if len(values) == 0:
        return ""
    low, high = values.min(), values.max()
    scaled = np.zeros(len(values), dtype=int) if high == low else \
        ((values - low) / (high - low) * (len(SPARK) - 1)).round().astype(int)
    return "".join(SPARK[i] for i in scaled)


def duration(seconds):
    minutes = int(seconds) // 60
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def render(status, now=None):
    """ The stats panel for one status dict. Pure, so it can be tested without a run. """
    now = time.time() if now is None else now
    if not status:
        return Panel("Waiting for training to start (no status.json or train_*.log yet).",
                     title="WREN 1.5", border_style="yellow")

    seeds = status.get("seeds") or [status.get("seed")]
    batches = status.get("batches") or 0
    runs = status.get("runs", [])
    batch = status.get("batch", 0)
    phase = status.get("phase", "?")
    age = now - status.get("updated", now)

    if phase == "finished":
        state = Text("FINISHED", style="bold green")
    elif phase == "stopped":
        state = Text("STOPPED: training crashed or was interrupted, see the log", style="bold red")
    elif age > STALE_AFTER:
        state = Text(f"STALE: no update for {duration(age)}, training may have stopped", style="bold red")
    else:
        state = Text(phase.upper(), style="bold cyan")
    elapsed = status.get("updated", now) - status.get("started", now)
    running = phase not in ("finished", "stopped")
    done = len(runs) * batches + (batch if running else 0)
    header = Text.assemble("status ", state, f"   updated {int(age)}s ago   "
                           f"started {time.strftime('%H:%M', time.localtime(status.get('started', now)))}, "
                           f"running {duration(elapsed)}")
    if done and batches and running:
        remaining = (len(seeds) * batches - done) * elapsed / done
        header.append(f"   ETA {time.strftime('%H:%M', time.localtime(now + remaining))}", style="bold")

    parts = [header]

    if running:
        current = Table.grid(padding=(0, 2))
        current.add_column(style="bold")
        current.add_column()
        position = seeds.index(status.get("seed")) + 1 if status.get("seed") in seeds else len(runs) + 1
        current.add_row("Seed", f"{status.get('seed')}  ({position} of {len(seeds)})")
        if batches:
            current.add_row("Progress", ProgressBar(total=batches, completed=batch, width=40,
                                                    complete_style="cyan"))
            current.add_row("", f"batch {batch:,} / {batches:,} ({100 * batch / batches:.0f}%)")
        losses = [l for _, l in status.get("loss", [])]
        if losses:
            current.add_row("Loss", Text.assemble(
                (f"now {losses[-1]:.3f}", "bold yellow"),
                f"   start {losses[0]:.3f}   best {min(losses):.3f}   "))
            current.add_row("", Text(sparkline(losses), style="yellow"))
        quick = status.get("quick", [])
        if quick:
            best = max(quick, key=lambda q: q[1])
            current.add_row("Quick val", f"last {100 * quick[-1][1]:.2f}% exact, "
                                         f"{100 * quick[-1][2]:.2f}% within one (batch {quick[-1][0]:,})")
            current.add_row("", Text(f"best {100 * best[1]:.2f}% (batch {best[0]:,})", style="green"))
        parts.append(Panel(current, title="Current run", border_style="cyan"))

    if runs:
        table = Table(box=None, header_style="bold magenta")
        for column in ("Seed", "Exact", "Within one", "Time", "Saved"):
            table.add_column(column, justify="right")
        for r in runs:
            kept = r.get("seed") == status.get("champion_seed")
            table.add_row(str(r.get("seed")), f"{100 * r.get('val_exact', 0):.2f}%",
                          f"{100 * r.get('val_within_one', 0):.2f}%",
                          duration(r["seconds"]) if "seconds" in r else "?", "yes" if r.get("saved") else "",
                          style="bold green" if kept else None)
        exact = np.array([100 * r.get("val_exact", 0) for r in runs])
        spread = f"{exact.mean():.1f} ± {exact.std(ddof=1):.1f}%" if len(exact) > 1 \
            else f"{exact.mean():.1f}% (std needs 2+ seeds)"
        summary = Text.assemble("Champion  ", (f"seed {status.get('champion_seed')}, "
                                               f"{100 * (status.get('champion_exact') or 0):.2f}% exact",
                                               "bold green"), f"     Mean so far  {spread}")
        parts.append(Panel(Group(table, Text(""), summary),
                           title=f"Finished seeds ({len(runs)} of {len(seeds)})", border_style="green"))

    config = status.get("config")
    if config:
        parts.append(Text("   ".join(f"{k} {v}" for k, v in config.items()), style="dim"))
    if status.get("source"):
        parts.append(Text(f"read from {status['source']} (this run predates status.json)", style="dim"))

    return Panel(Group(*parts), title=f"[bold cyan]WREN 1.5[/] {DOT} live training stats",
                 border_style="magenta")


def watch(workdir, interval=2.0):
    """ Redraws the stats panel every interval seconds until Ctrl+C. Read-only. """
    console = Console()
    try:
        with Live(render(read_status(workdir)), console=console, refresh_per_second=4) as live:
            while True:
                time.sleep(interval)
                live.update(render(read_status(workdir)))
    except KeyboardInterrupt:
        pass


def demo(interval=0.5):
    """ The watch panel fed by a fake three-seed run in a temporary folder, to show it off
    without training. Never reads or writes a real workdir. """
    import tempfile
    import threading

    seeds, batches = [0, 1, 2], 2000
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workdir:
        dash = Dashboard(workdir, live=False)
        dash.status["config"] = {"seeds": "0 1 2", "batches": f"{batches:,} x 32", "demo": "fake data"}
        dash.start(seeds, batches, 32)

        def fake_training():
            champion = (None, -1.0)
            for seed in seeds:
                dash.start_seed(seed)
                for number in range(50, batches + 1, 50):
                    time.sleep(interval / 4)
                    dash.batch(number)
                    dash.status["loss"].append([number, 1.6 * np.exp(-number / 800) + 0.9
                                                + rng.normal(0, 0.03)])
                    if number % 500 == 0:
                        dash.status["quick"].append([number, 0.35 + 0.1 * number / batches + rng.normal(0, 0.01),
                                                     0.75 + 0.1 * number / batches])
                    dash.write_status()
                exact = 0.45 + rng.normal(0, 0.01)
                if exact > champion[1]:
                    champion = (seed, exact)
                dash.seed_done({"seed": seed, "val_exact": exact, "val_within_one": exact + 0.35,
                                "seconds": batches * interval / 200, "saved": champion[0] == seed}, *champion)
            dash.finish()
            while True:  # Windows skips a write while the watcher reads, so keep the last one coming
                time.sleep(interval)
                dash.write_status()

        threading.Thread(target=fake_training, daemon=True).start()
        watch(workdir, interval)
