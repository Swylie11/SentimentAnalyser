"""Training dashboard checks for Wren 1.5: status.json, the log, and `wren watch`.

Trains two tiny seeds on synthetic data in a temporary folder, then checks what the
watcher would see, both from status.json and from the log alone:

    python tests/test_dashboard.py
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import time

# src/wren-1.5 isn't a valid package name, so load it by path as `wren15`.
HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(HERE, "..", "src", "wren-1.5")
spec = importlib.util.spec_from_file_location(
    "wren15", os.path.join(PACKAGE_DIR, "__init__.py"), submodule_search_locations=[PACKAGE_DIR])
sys.modules["wren15"] = module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from rich.console import Console  # noqa: E402

from wren15.train import comms as com, dashboard, init_databases, loop  # noqa: E402


def text_of(renderable):
    console = Console(file=io.StringIO(), width=120)
    console.print(renderable)
    return console.file.getvalue()


def main():
    # Small intervals so a few batches produce loss records and quick validations.
    loop.LOSS_EVERY, loop.VALIDATION_EVERY = 2, 4

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workdir:
        com.set_db_dir(workdir)
        init_databases.setup(synthetic=40)
        log_path = os.path.join(workdir, "train_20260101-000000.log")
        loop.train(4, 8, assume_yes=True, seeds=[0, 1], log_path=log_path)

        with open(os.path.join(workdir, dashboard.STATUS_FILE), encoding="utf-8") as f:
            status = json.load(f)
        assert status["phase"] == "finished", status["phase"]
        assert [r["seed"] for r in status["runs"]] == [0, 1]
        assert status["champion_seed"] in (0, 1)
        assert len(status["loss"]) == 4 and len(status["quick"]) == 2, "seed 1's curves are missing"
        print("pass  status.json records both seeds, the champion and the curves")

        with open(log_path, encoding="utf-8") as f:
            log = f.read()
        for expected in ("=== Seed 0 ===", "training loss:", "quick validation:", "Multi-seed summary"):
            assert expected in log, f"log is missing {expected!r}"
        print("pass  the log gets every message")

        # The log alone must give the watcher the same answer, for runs without status.json.
        parsed = dashboard.parse_log(log_path)
        assert [r["seed"] for r in parsed["runs"]] == [0, 1]
        assert parsed["champion_seed"] == status["champion_seed"]
        assert parsed["phase"] == "finished"
        print("pass  parse_log agrees with status.json")

        assert "FINISHED" in text_of(dashboard.render(status))
        mid_run = dict(status, phase="training", runs=status["runs"][:1], seed=1, batch=6)
        shown = text_of(dashboard.render(mid_run, now=status["updated"] + 5))
        assert "TRAINING" in shown and "Current run" in shown and "ETA" in shown
        stale = text_of(dashboard.render(mid_run, now=status["updated"] + 3600))
        assert "STALE" in stale
        assert "Waiting" in text_of(dashboard.render(None))
        assert "STOPPED" in text_of(dashboard.render(dict(status, phase="stopped")))
        print("pass  render shows running, stale, stopped, finished and waiting states")


if __name__ == "__main__":
    main()
