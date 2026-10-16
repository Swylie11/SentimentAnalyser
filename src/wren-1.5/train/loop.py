"""Trains a new Wren model. Run it with `wren train`.

Architecture: embeddings -> parallel filter bank branches (Kim 2014 style: several
region sizes, each a bank of filters sliding along the word axis only, ReLU, global
max-pool over word position) -> concatenate -> the hidden ReLU layers in HIDDEN_WIDTHS -> softmax over
five star ratings.
"""

import json
import os
import shutil
import time

import numpy as np

from . import comms as com, init_databases
from .dashboard import Dashboard
from ..config import FILTER_BANK_CONFIG, HIDDEN_WIDTHS, NUM_CLASSES
from ..layers.filter_bank import FilterBankLayer
from ..layers.neural import NeuralLayer
from ..model import forward_pass as _forward_pass

# AdamW. Under SGD + momentum (lr 0.003, momentum 0.9) the three-seed round sat at
# chance loss (ln 5) for 3,000 to 9,000 of its 15,000 batches before learning, and the
# loss was still falling when the runs ended. AdamW scales each weight's step by its
# own gradient size, which is what gets a five-layer ReLU stack off that plateau.
# 1e-3 and 0.01 are the usual AdamW starting points, not yet tuned for Wren.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01


def cosine_learning_rate(batch_index, num_batches):
    """ LEARNING_RATE at the first batch, falling along a half cosine to 0 at the last.

    At a constant rate the quick validation checks late in a run swung by up to 9
    points between checks, so picking the best checkpoint was partly luck. Shrinking
    the steps lets the run settle instead of bouncing around the minimum. """
    return LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * batch_index / num_batches))

# Reviews held out and never trained on, so the reported accuracy means something.
VALIDATION_SIZE = 2000

# Seeded so the train/validation split and the batch order are reproducible.
DATA_SEED = 0

# Reviews per forward pass during evaluation. Each review is a 200x300 embedding
# matrix, so evaluating thousands at once would not fit in memory.
EVALUATION_CHUNK = 50

# Weights are held in memory during a run and written to SQLite this often, rather
# than round-tripping every parameter through JSON on every single batch.
CHECKPOINT_EVERY = 25

# How often, in batches, to check progress against a small validation slice during
# training. Watching this curve is what tells us when a run has plateaued instead of
# having to guess a batch count in advance.
VALIDATION_EVERY = 500

# A fixed slice of validation_ids (not a fresh random draw each time) so successive
# quick checks are comparable to each other. Smaller than the full VALIDATION_SIZE
# since this runs many times per training run. At 300 one check was about +-2.9
# points, too noisy to pick the best checkpoint on; 1000 is about +-1.6. Each check
# costs roughly as much as 10 training batches, so VALIDATION_EVERY is 500 to keep
# checking at about 2% of training time.
QUICK_VALIDATION_SIZE = 1000

# How often, in batches, to record the mean training loss over the batches since the
# last record. Free to compute (the loss is already worked out every batch), so it is
# much finer than VALIDATION_EVERY. Plotted by `tools/visualise.py loss`.
LOSS_EVERY = 50

# Where training's output goes. train() swaps in one that draws the live progress bar,
# logs to a file and writes status.json for `wren watch`. run_once called on its own, as
# the tests do, gets this plain one.
dash = Dashboard()



def one_hot_ratings(star_ratings):
    """ Converts a list of 1-5 star ratings to a (batch, 5) one hot matrix. """
    encoded = np.zeros((len(star_ratings), NUM_CLASSES))
    encoded[np.arange(len(star_ratings)), np.asarray(star_ratings, dtype=int) - 1] = 1
    return encoded


def forward_pass(conv_layers, dense_layers, output_layer, texts, training=False):
    """ Runs a batch of review texts through the network, reading embeddings from the database. """
    return _forward_pass(conv_layers, dense_layers, output_layer, texts, com.fetch_embedding, training)


def backward_pass(conv_layers, dense_layers, output_layer, one_hot):
    """ Propagates the loss gradient back through the whole network.

    Each calculate_derivatives call returns the gradient with respect to that
    layer's inputs, which is the gradient of the layer below's output, so it has to
    be carried down the stack rather than discarded.

    Assumes the forward pass ran with training=True, so each dense layer holds the
    dropout mask it applied. """
    gradient = output_layer.calculate_derivatives(output_layer.combined_derivative(one_hot))
    for layer in reversed(dense_layers):
        gradient = layer.calculate_derivatives(layer.droput_relu_backward(gradient))

    # Split the concatenated gradient back into each branch's slice of features, in
    # the same order forward_pass concatenated them in. Embeddings are not currently
    # trained, so branches don't need their input gradient back.
    start = 0
    for layer in conv_layers:
        end = start + layer.num_filters
        layer.backpropagate(gradient[:, start:end])
        start = end


def class_distribution(values):
    """ Formats the count of each star rating, so a model that has collapsed onto a
    single class is visible rather than hidden behind an accuracy figure. """
    values = np.asarray(values, dtype=int)
    total = max(len(values), 1)
    parts = []
    for star in range(1, NUM_CLASSES + 1):
        count = int(np.sum(values == star))
        parts.append(f"{star}*: {count:>5} ({100 * count / total:4.1f}%)")
    return "  ".join(parts)


def predict_texts(conv_layers, dense_layers, output_layer, texts):
    """ 1-5 star predictions for review texts, a chunk at a time to bound memory. """
    predictions = []
    for start in range(0, len(texts), EVALUATION_CHUNK):
        softmax_output = forward_pass(conv_layers, dense_layers, output_layer,
                                      texts[start:start + EVALUATION_CHUNK])
        predictions.extend((np.argmax(softmax_output, axis=1) + 1).tolist())
    return predictions


def evaluate(conv_layers, dense_layers, output_layer, review_ids):
    """ Runs the network over a set of reviews without training on them.

    Returns exact accuracy, accuracy within one star, the predictions and the
    true labels. Adjacent class errors are the normal failure mode for star
    prediction, so both figures are worth seeing. """
    predictions = []
    labels = []

    for start in range(0, len(review_ids), EVALUATION_CHUNK):
        rows = com.fetch_batch(review_ids[start:start + EVALUATION_CHUNK])
        labels.extend(int(row[0]) for row in rows)
        predictions.extend(predict_texts(conv_layers, dense_layers, output_layer, [row[1] for row in rows]))

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    exact = float(np.mean(predictions == labels)) if len(labels) else 0.0
    within_one = float(np.mean(np.abs(predictions - labels) <= 1)) if len(labels) else 0.0

    return exact, within_one, predictions, labels


def training_batches(train_ids, reviews_per_batch, num_batches, seed):
    """ Yields batches of review ids drawn from a seeded shuffle.

    The previous code walked the ids in order, so if the table has any ordering by
    rating the model would see one class at a time and could not learn. """
    rng = np.random.default_rng(seed)
    order = rng.permutation(train_ids)
    position = 0

    for _ in range(num_batches):
        if position + reviews_per_batch > len(order):
            order = rng.permutation(train_ids)  # Reshuffle once the epoch is exhausted
            position = 0
        yield order[position:position + reviews_per_batch].tolist()
        position += reviews_per_batch


MODEL_DATABASES = ("neuron_weights.db", "convolution_layers.db")

# Written by a multi-seed run, next to the databases it describes.
SEED_RUNS_FILE = "seed_runs.json"


def confirm_and_back_up_model(assume_yes=False):
    """ Confirms before training overwrites the stored model, and keeps a copy.

    Training reinitialises every weight and kernel, so starting it destroys whatever
    model was loaded. Rather than warning about that in the README, ask first and
    write a timestamped backup of the databases that are about to be overwritten. """
    here = com.DB_DIR
    existing = [name for name in MODEL_DATABASES if os.path.exists(os.path.join(here, name))]

    if existing:
        print("\nTraining will overwrite the current weights and kernels in "
              + ", ".join(existing) + ".")
        if not assume_yes and input(
                "A timestamped backup will be written first. Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled. Nothing was changed.")
            return False

        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in existing:
            source = os.path.join(here, name)
            backup = os.path.join(here, f"{name[:-3]}.{stamp}.db")
            shutil.copy2(source, backup)
            print(f"  backed up {name} -> {os.path.basename(backup)}")

    return True


def save_parameters(conv_layers, all_layers):
    """ Writes the current parameters to the database. """
    for layer in conv_layers:
        layer.save()
    for layer in all_layers:
        layer.save()


def snapshot_model(conv_layers, all_layers):
    """ Copies every trainable array in memory, so training can restore the
    best-scoring checkpoint seen during a run rather than whatever the final batch
    happened to leave behind. """
    return {
        "conv": [(np.array(layer.kernel, dtype=float).copy(), np.array(layer.bias, dtype=float).copy())
                for layer in conv_layers],
        "dense": [(np.array(layer.weights, dtype=float).copy(), np.array(layer.biases, dtype=float).copy())
                 for layer in all_layers],
    }


def restore_model(conv_layers, all_layers, snapshot):
    """ Restores a snapshot taken by snapshot_model. """
    for layer, (kernel, bias) in zip(conv_layers, snapshot["conv"]):
        layer.kernel = kernel
        layer.bias = bias
    for layer, (weights, biases) in zip(all_layers, snapshot["dense"]):
        layer.weights = weights
        layer.biases = biases


def build_layers():
    """ The network's layers, with no values loaded yet. """
    conv_layers = [FilterBankLayer(layer_num, region_size, num_filters)
                  for layer_num, region_size, num_filters in FILTER_BANK_CONFIG]
    dense_layers = [NeuralLayer(n) for n in range(1, len(HIDDEN_WIDTHS) + 1)]
    output_layer = NeuralLayer(len(HIDDEN_WIDTHS) + 1)
    return conv_layers, dense_layers, output_layer


def run_once(seed, layers, train_ids, validation_ids, reviews_per_batch, num_batches, checkpoint):
    """ Trains the layers from a fresh start and evaluates them on validation_ids.

    The seed only drives the global numpy RNG, which is used for weight
    initialisation and dropout masks. The data split and batch order have their own
    generators seeded from DATA_SEED, so every seed sees the same batches. With
    checkpoint off nothing is written to the database.

    Returns the run record, and the validation predictions and labels. """
    conv_layers, dense_layers, output_layer = layers
    all_layers = list(dense_layers) + [output_layer]
    quick_validation_ids = validation_ids[:QUICK_VALIDATION_SIZE]

    if seed is not None:
        np.random.seed(seed)
        dash.say(f"\n=== Seed {seed} ===", style="bold cyan")
    dash.start_seed(seed)

    # Fresh values in memory. They are kept there for the whole run and only
    # written to the database at checkpoints, if checkpoint is on.
    for layer in conv_layers:
        layer.initialize_values()
    for layer in all_layers:
        layer.initialise_values()
    if checkpoint:
        save_parameters(conv_layers, all_layers)

    start = time.time()

    total_loss = 0.0
    correct_outputs = 0
    seen = 0
    train_predictions = []
    train_labels = []
    best_quick_accuracy = -1.0
    best_snapshot = None
    curve = []
    loss_curve = []
    window_loss = 0.0

    for batch_index, batch_ids in enumerate(
            training_batches(train_ids, reviews_per_batch, num_batches, DATA_SEED)):

        dash.batch(batch_index + 1)

        # One query for the whole batch, rather than one per review.
        rows = com.fetch_batch(batch_ids)
        texts = [row[1] for row in rows]
        star_ratings = [int(row[0]) for row in rows]
        one_hot = one_hot_ratings(star_ratings)

        softmax_output = forward_pass(conv_layers, dense_layers, output_layer, texts, training=True)
        predictions = np.argmax(softmax_output, axis=1) + 1

        output_layer.ccel_calculation(one_hot)
        backward_pass(conv_layers, dense_layers, output_layer, one_hot)
        total_loss += output_layer.averageLoss
        window_loss += output_layer.averageLoss
        if (batch_index + 1) % LOSS_EVERY == 0:
            loss_curve.append([batch_index + 1, window_loss / LOSS_EVERY])
            dash.loss(batch_index + 1, window_loss / LOSS_EVERY)
            window_loss = 0.0

        correct_outputs += int(np.sum(predictions == np.asarray(star_ratings)))
        seen += len(texts)
        train_predictions.extend(predictions.tolist())
        train_labels.extend(star_ratings)

        # Updating values
        learning_rate = cosine_learning_rate(batch_index, num_batches)
        for layer in all_layers:
            layer.adjust_values(learning_rate, WEIGHT_DECAY)
        for layer in conv_layers:
            layer.adjust_kernel_values(learning_rate, WEIGHT_DECAY)

        if checkpoint and (batch_index + 1) % CHECKPOINT_EVERY == 0:
            save_parameters(conv_layers, all_layers)

        if (batch_index + 1) % VALIDATION_EVERY == 0:
            dash.validating(f"{len(quick_validation_ids)} reviews")
            quick_exact, quick_within_one, _, _ = evaluate(
                conv_layers, dense_layers, output_layer, quick_validation_ids)
            dash.quick(batch_index + 1, quick_exact, quick_within_one)
            curve.append([batch_index + 1, quick_exact, quick_within_one])
            if quick_exact > best_quick_accuracy:
                best_quick_accuracy = quick_exact
                best_snapshot = snapshot_model(conv_layers, all_layers)

    if best_snapshot is not None:
        restore_model(conv_layers, all_layers, best_snapshot)
        dash.say(f"\nRestored best checkpoint seen during training "
             f"(quick validation accuracy {100 * best_quick_accuracy:.2f}%).")

    elapsed = time.time() - start
    dash.say(f"\nTotal time elapsed: {elapsed:.2f}s")

    dash.say(f"\nTraining (seen during training, not a fair measure)")
    dash.say(f"  Average loss     : {total_loss / max(num_batches, 1):.4f}")
    dash.say(f"  Accuracy         : {100 * correct_outputs / max(seen, 1):.2f}%")
    dash.say(f"  True labels      : {class_distribution(train_labels)}")
    dash.say(f"  Predictions      : {class_distribution(train_predictions)}")

    dash.validating(f"{len(validation_ids):,} reviews")
    exact, within_one, predictions, labels = evaluate(
        conv_layers, dense_layers, output_layer, validation_ids)

    dash.say(f"\nValidation ({len(labels)} held out reviews, never trained on)")
    dash.say(f"  Accuracy         : {100 * exact:.2f}%")
    dash.say(f"  Within one star  : {100 * within_one:.2f}%")

    record = {"seed": seed, "curve": curve, "loss_curve": loss_curve, "val_exact": exact, "val_within_one": within_one,
              "best_quick": best_quick_accuracy if best_snapshot is not None else None,
              "seconds": round(elapsed, 1), "saved": False}
    return record, predictions, labels


def print_distributions(predictions, labels):
    dash.say(f"  True labels      : {class_distribution(labels)}")
    dash.say(f"  Predictions      : {class_distribution(predictions)}")
    if len(set(predictions.tolist())) == 1:
        dash.say("\n  Warning: every prediction is the same class. The model has "
              "collapsed onto one output rather than learning to separate them.")


def write_results(results):
    """ Rewrites seed_runs.json next to the databases, so finished runs survive a crash. """
    with open(os.path.join(com.DB_DIR, SEED_RUNS_FILE), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


def print_summary(results, predictions, labels):
    runs = results["runs"]
    dash.say(f"\nMulti-seed summary ({len(runs)} runs, {results['batches']} batches each, "
          f"data seed {results['data_seed']})\n")
    dash.say("  Seed   Exact    Within one   Best quick   Time      Saved")
    dash.say("  ----   ------   ----------   ----------   -------   -----")
    for r in runs:
        best_quick = f"{100 * r['best_quick']:.2f}%" if r["best_quick"] is not None else "n/a"
        minutes, seconds = divmod(int(round(r["seconds"])), 60)
        time_taken = f"{minutes}m {seconds:02d}s"
        dash.say(f"  {r['seed']:>4}   {100 * r['val_exact']:5.2f}%   {100 * r['val_within_one']:9.2f}%   "
              f"{best_quick:>10}   {time_taken:>7}   {'yes' if r['saved'] else ''}".rstrip())

    exact = 100 * np.array([r["val_exact"] for r in runs])
    within_one = 100 * np.array([r["val_within_one"] for r in runs])
    if len(runs) > 1:
        # Sample std: these seeds are a sample of all possible seeds.
        spread = (f"{exact.mean():.1f} ± {exact.std(ddof=1):.1f}%   "
                  f"{within_one.mean():.1f} ± {within_one.std(ddof=1):.1f}%")
    else:
        spread = f"{exact.mean():.1f}%, std n/a   {within_one.mean():.1f}%, std n/a"
    champion = next(r for r in runs if r["seed"] == results["champion_seed"])
    dash.say(f"\n  Mean ± std     {spread}")
    dash.say(f"  Kept           seed {champion['seed']} ({100 * champion['val_exact']:.2f}% exact) -> "
          + ", ".join(MODEL_DATABASES))
    print_distributions(predictions, labels)


def short_path(path):
    """ Relative to the current folder when it's inside it, so the printed watch command
    fits on one line. Different drives have no relative path on Windows. """
    try:
        relative = os.path.relpath(path)
    except ValueError:
        return path
    return path if relative.startswith("..") else relative


def wilson(successes, n, z=1.96):
    """ 95% Wilson score interval for a proportion, as evaluation/evaluate.py reports it. """
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def test_evaluation(layers, test_path):
    """ Scores the saved model on a fresh test set, a .jsonl of {rating, text} lines
    such as Data/eval_test.jsonl, the same set Wren 1.0's 47.0% was measured on.

    Reads the weights back from the databases rather than using what is in memory:
    after a multi-seed run memory holds the last seed, not the champion that was saved. """
    conv_layers, dense_layers, output_layer = layers
    for layer in conv_layers:
        layer.fetchKernel()
    for layer in list(dense_layers) + [output_layer]:
        layer.fetch_values()

    with open(test_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    labels = np.array([int(r["rating"]) for r in records])
    dash.say(f"\nEvaluating the saved model on {test_path} ({len(labels):,} reviews)...")
    predictions = np.asarray(predict_texts(conv_layers, dense_layers, output_layer,
                                           [r["text"] for r in records]))

    exact_k = int(np.sum(predictions == labels))
    within_k = int(np.sum(np.abs(predictions - labels) <= 1))
    return {"test_set": test_path, "n": len(labels),
            "exact": exact_k / len(labels), "exact_ci": wilson(exact_k, len(labels)),
            "within_one": within_k / len(labels), "within_one_ci": wilson(within_k, len(labels)),
            "recall": [float(np.mean(predictions[labels == s] == s)) if np.any(labels == s) else None
                       for s in range(1, NUM_CLASSES + 1)],
            "prediction_counts": [int(np.sum(predictions == s)) for s in range(1, NUM_CLASSES + 1)]}


def print_test(test, previous_best):
    lo, hi = test["exact_ci"]
    dash.say(f"\nFresh test set ({test['n']:,} reviews, never trained on or used for selection)",
             style="bold")
    dash.say(f"  Exact accuracy   : {100 * test['exact']:.1f}%   95% CI {100 * lo:.1f} to {100 * hi:.1f}")
    dash.say(f"  Within one star  : {100 * test['within_one']:.1f}%")
    dash.say("  Recall per star  : " + "  ".join(
        f"{s}*: {100 * r:.0f}%" for s, r in enumerate(test["recall"], 1) if r is not None))
    if previous_best:
        dash.say(f"  Previous best    : {100 * previous_best['test']['exact']:.1f}% "
                 f"({previous_best['finished']}, {previous_best['workdir']})")


def read_history(history_path):
    if not os.path.exists(history_path):
        return []
    with open(history_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_history(history_path, results, test):
    """ Adds one line for this training run to the history file. Unlike seed_runs.json,
    which each run overwrites in its own work directory, this file keeps every run, so
    tonight's result can be compared with last night's. """
    record = {"finished": time.strftime("%Y-%m-%d %H:%M"), "workdir": com.DB_DIR,
              **{k: v for k, v in results.items() if k != "runs"},
              "val_exact": {str(r["seed"]): r["val_exact"] for r in results["runs"]},
              "test": test}
    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def train(reviews_per_batch, num_batches, assume_yes=False, seeds=None, log_path=None,
          test_path=None, history_path=None):
    """Trains a new model on the databases in comms.DB_DIR, replacing the stored weights.

    With seeds, trains once per seed and saves only a run that beats every earlier
    one on the full validation set (the champion). The mean and std across seeds are
    the fair figure for the architecture; the champion's own score is optimistic.

    log_path, if given, gets every message appended to it, without the progress bar.

    test_path, if given and present, scores the saved model on that fresh test set
    once training ends. The scores go into seed_runs.json and, with history_path, a
    line is appended there too. """
    global dash
    layers = build_layers()

    if not confirm_and_back_up_model(assume_yes):
        return
    # The dense layers read their shape from the database, so reseed it to match
    # HIDDEN_WIDTHS. Safe: the old model was just backed up and is about to be replaced.
    init_databases.seed_layers(init_databases.flattened_width())

    dash = Dashboard(com.DB_DIR, log_path)
    dash.banner({"seeds": " ".join(map(str, seeds)) if seeds else "none",
                 "batches": f"{num_batches:,} x {reviews_per_batch}", "lr": f"{LEARNING_RATE}, cosine to 0",
                 "optimiser": f"AdamW, weight decay {WEIGHT_DECAY}",
                 "dropout": f"features {FilterBankLayer.dropout_rate}, hidden {NeuralLayer.dropout_rate}",
                 "hidden": HIDDEN_WIDTHS},
                f'python tools/wren15.py watch --workdir "{short_path(com.DB_DIR)}"')
    dash.start(seeds or [None], num_batches, reviews_per_batch)
    try:
        results = _train(layers, reviews_per_batch, num_batches, seeds)
    finally:
        if dash.status["phase"] != "finished":
            dash.finish("stopped")  # crashed or Ctrl+C: put the terminal back, tell the watcher

    if not test_path:
        return
    if not os.path.exists(test_path):
        dash.say(f"\nNo test set at {test_path}, so the saved model was not evaluated.")
        return
    test = test_evaluation(layers, test_path)
    results["test"] = test
    write_results(results)
    if history_path:
        # Only runs scored on this same test set are comparable.
        same = os.path.normcase(os.path.abspath(test_path))
        earlier = [r for r in read_history(history_path) if r.get("test")
                   and os.path.normcase(os.path.abspath(r["test"]["test_set"])) == same]
        print_test(test, max(earlier, key=lambda r: r["test"]["exact"], default=None))
        append_history(history_path, results, test)
        dash.say(f"  Saved to {SEED_RUNS_FILE} and {history_path}")
    else:
        print_test(test, None)


def _train(layers, reviews_per_batch, num_batches, seeds):
    conv_layers, dense_layers, output_layer = layers
    all_layers = list(dense_layers) + [output_layer]

    train_ids, validation_ids = com.split_ids(
        com.fetch_all_review_ids(), VALIDATION_SIZE, DATA_SEED)
    dash.say(f"\n{len(train_ids)} training reviews, {len(validation_ids)} held out for validation.")

    # With one run there is no champion to protect, so it checkpoints as before.
    checkpoint = seeds is None or len(seeds) == 1

    results = {"batches": num_batches, "batch_size": reviews_per_batch, "data_seed": DATA_SEED,
               "dropout_rate": NeuralLayer.dropout_rate, "feature_dropout_rate": FilterBankLayer.dropout_rate,
               "hidden_widths": HIDDEN_WIDTHS, "learning_rate": LEARNING_RATE, "lr_schedule": "cosine",
               "optimiser": "adamw", "weight_decay": WEIGHT_DECAY,
               "validation_size": len(validation_ids), "runs": [], "champion_seed": None}

    if seeds is None:
        record, predictions, labels = run_once(None, layers, train_ids, validation_ids,
                                               reviews_per_batch, num_batches, checkpoint)
        save_parameters(conv_layers, all_layers)
        record["saved"] = True
        results["runs"].append(record)
        dash.seed_done(record, None, record["val_exact"])
        dash.finish()
        print_distributions(predictions, labels)
        return results

    champion_exact = -1.0
    champion_predictions = None

    for seed in seeds:
        record, predictions, labels = run_once(seed, layers, train_ids, validation_ids,
                                               reviews_per_batch, num_batches, checkpoint)
        # Strictly better only, so a tie keeps the earlier seed and the files on
        # disk change only when a run actually beats the champion.
        if record["val_exact"] > champion_exact:
            save_parameters(conv_layers, all_layers)
            champion_exact = record["val_exact"]
            champion_predictions = predictions
            results["champion_seed"] = seed
            record["saved"] = True
            dash.say(f"  New champion, saved to {', '.join(MODEL_DATABASES)}.", style="bold green")
        results["runs"].append(record)
        write_results(results)
        dash.seed_done(record, results["champion_seed"], champion_exact)

    dash.finish()
    print_summary(results, champion_predictions, labels)
    return results
