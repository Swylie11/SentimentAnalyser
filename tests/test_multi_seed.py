"""Multi-seed training checks for Wren 1.5 (`wren train --seeds`).

A seed must change the weight initialisation and dropout masks, and nothing about
the data. The champion rule must save a run only when it beats every earlier one.

Builds its own small synthetic databases in a temporary folder, so it needs no
setup and leaves the work directory alone:

    python tests/test_multi_seed.py
"""

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

# src/wren-1.5 isn't a valid package name, so load it by path as `wren15`.
HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(HERE, "..", "src", "wren-1.5")
spec = importlib.util.spec_from_file_location(
    "wren15", os.path.join(PACKAGE_DIR, "__init__.py"), submodule_search_locations=[PACKAGE_DIR])
sys.modules["wren15"] = module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from wren15.train import comms as com, init_databases, loop  # noqa: E402

BATCH_SIZE = 4
NUM_BATCHES = 3


def weights(layers):
    conv_layers, dense_layers, output_layer = layers
    return [np.array(layer.kernel, dtype=float) for layer in conv_layers] + \
           [np.array(layer.weights, dtype=float) for layer in dense_layers + [output_layer]]


def train_seed(seed, train_ids, validation_ids):
    layers = loop.build_layers()
    loop.run_once(seed, layers, train_ids, validation_ids, BATCH_SIZE, NUM_BATCHES, checkpoint=False)
    return weights(layers)


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workdir:
        com.set_db_dir(workdir)
        init_databases.setup(synthetic=40)
        train_ids, validation_ids = com.split_ids(com.fetch_all_review_ids(), 10, loop.DATA_SEED)

        first, again, other = (train_seed(s, train_ids, validation_ids) for s in (0, 0, 1))
        assert all(np.array_equal(a, b) for a, b in zip(first, again)), "seed 0 is not reproducible"
        print("pass  two runs with seed 0 give identical weights")
        assert not any(np.array_equal(a, b) for a, b in zip(first, other)), "seeds 0 and 1 match"
        print("pass  seeds 0 and 1 give different weights")

        batches = []
        for seed in (0, 1):
            np.random.seed(seed)
            batches.append(list(loop.training_batches(train_ids, BATCH_SIZE, NUM_BATCHES, loop.DATA_SEED)))
        assert batches[0] == batches[1], "the seed changed the batch order"
        print("pass  seeds 0 and 1 see the same batches")

        # Champion rule: scores 0.4, 0.6, 0.5 should save after seeds 0 and 1 only.
        scores = iter([0.4, 0.6, 0.5])
        saves = []
        loop.evaluate = lambda *args: (next(scores), 0.9, np.array([3, 3]), np.array([3, 4]))
        loop.save_parameters = lambda *args: saves.append(1)
        loop.confirm_and_back_up_model = lambda assume_yes: True
        loop.train(BATCH_SIZE, NUM_BATCHES, assume_yes=True, seeds=[0, 1, 2])

        with open(os.path.join(workdir, loop.SEED_RUNS_FILE), encoding="utf-8") as f:
            results = json.load(f)
        assert len(saves) == 2, f"save_parameters called {len(saves)} times, expected 2"
        assert results["champion_seed"] == 1, results["champion_seed"]
        assert [r["saved"] for r in results["runs"]] == [True, True, False]
        print("pass  champion rule saves only when beaten, champion is seed 1")

        com.set_db_dir(HERE)  # Closes the cached connections so the folder can be removed
    return 0


if __name__ == "__main__":
    sys.exit(main())
