"""Trains or runs the sentiment analyser.

Architecture: embeddings -> parallel filter bank branches (Kim 2014 style: several
region sizes, each a bank of filters sliding along the word axis only, ReLU, global
max-pool over word position) -> concatenate -> five dense ReLU layers -> softmax over
five star ratings.
"""

import os
import shutil
import time

import numpy as np

import Comms as com
import WordVectorConversions as wvc
from FilterBankLayer import FilterBankLayer
from InitDatabases import FILTER_BANK_CONFIG
from NeuralLayer import NeuralLayer

# 0.01 was too low for this depth: the overfit test reached 100% on two random
# seeds out of three and stalled at 46% on the third. At 0.03 it reaches 100% on
# every seed tried.
LEARNING_RATE = 0.03

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
VALIDATION_EVERY = 200

# A fixed slice of validation_ids (not a fresh random draw each time) so successive
# quick checks are comparable to each other. Deliberately smaller than the full
# VALIDATION_SIZE since this runs many times per training run.
QUICK_VALIDATION_SIZE = 300

NUM_CLASSES = 5

SENTIMENT_LABELS = {1: "Very negative", 2: "Negative", 3: "Neutral",
                    4: "Positive", 5: "Very positive"}


def one_hot_ratings(star_ratings):
    """ Converts a list of 1-5 star ratings to a (batch, 5) one hot matrix. """
    encoded = np.zeros((len(star_ratings), NUM_CLASSES))
    encoded[np.arange(len(star_ratings)), np.asarray(star_ratings, dtype=int) - 1] = 1
    return encoded


def forward_pass(conv_layers, dense_layers, output_layer, texts):
    """ Runs a batch of review texts through the whole network.

    Returns the softmax output, shape (len(texts), 5). """
    conv_input = wvc.pad_matrix(wvc.return_vector_matrix_jsonl(wvc.format_entry_data(texts)))
    conv_input = np.asarray(conv_input, dtype=float)

    # Parallel filter bank branches over the same embedding input, concatenated into
    # one feature vector -- not a stacked pipeline like the old two ConvLayers.
    activations = np.concatenate([layer.convPass(conv_input) for layer in conv_layers], axis=1)

    for layer in dense_layers:
        activations = layer.batch_layer_output(activations)

    return output_layer.softmax(activations)[1]


def backward_pass(conv_layers, dense_layers, output_layer, one_hot):
    """ Propagates the loss gradient back through the whole network.

    Each calculate_derivatives call returns the gradient with respect to that
    layer's inputs, which is the gradient of the layer below's output, so it has to
    be carried down the stack rather than discarded. """
    gradient = output_layer.calculate_derivatives(output_layer.combined_derivative(one_hot))
    for layer in reversed(dense_layers):
        gradient = layer.calculate_derivatives(layer.relu_backward(gradient))

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


def evaluate(conv_layers, dense_layers, output_layer, review_ids):
    """ Runs the network over a set of reviews without training on them.

    Returns exact accuracy, accuracy within one star, the predictions and the
    true labels. Adjacent class errors are the normal failure mode for star
    prediction, so both figures are worth seeing. """
    predictions = []
    labels = []

    for start in range(0, len(review_ids), EVALUATION_CHUNK):
        chunk = review_ids[start:start + EVALUATION_CHUNK]
        rows = com.fetch_batch(chunk)

        texts = [row[1] for row in rows]
        labels.extend(int(row[0]) for row in rows)

        softmax_output = forward_pass(conv_layers, dense_layers, output_layer, texts)
        predictions.extend((np.argmax(softmax_output, axis=1) + 1).tolist())

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


def confirm_and_back_up_model():
    """ Confirms before training overwrites the stored model, and keeps a copy.

    Training reinitialises every weight and kernel, so starting it destroys whatever
    model was loaded. Rather than warning about that in the README, ask first and
    write a timestamped backup of the databases that are about to be overwritten. """
    here = os.path.dirname(os.path.abspath(__file__))
    existing = [name for name in MODEL_DATABASES if os.path.exists(os.path.join(here, name))]

    if existing:
        print("\nTraining will overwrite the current weights and kernels in "
              + ", ".join(existing) + ".")
        answer = input("A timestamped backup will be written first. Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
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


def main():
    mode = int(input("To train a new model, press 1. To test the currently loaded model, press 2.\n"
                     "To load a model, input valid database files into the current folder titled:\n"
                     "'convolution_layers', and 'neuron_weights'.\n"
                     "WARNING: If training a new model, the current neuron weights and kernel files will be overridden.\n"))

    conv_layers = [FilterBankLayer(layer_num, region_size, num_filters)
                  for layer_num, region_size, num_filters in FILTER_BANK_CONFIG]
    dense_layers = [NeuralLayer(n) for n in range(1, 6)]
    output_layer = NeuralLayer(6)
    all_layers = list(dense_layers) + [output_layer]

    if mode == 1:
        if not confirm_and_back_up_model():
            return

        # Override the value matrices with freshly drawn values. If a new model is
        # not being trained, the currently loaded values are used.
        for layer in conv_layers:
            layer.initialize_values()
        for layer in all_layers:
            layer.initialise_values()

        reviews_per_batch = int(input("Enter number of reviews per batch: "))
        num_batches = int(input("Enter number of batches to run: "))

        train_ids, validation_ids = com.split_ids(
            com.fetch_all_review_ids(), VALIDATION_SIZE, DATA_SEED)
        print(f"\n{len(train_ids)} training reviews, {len(validation_ids)} held out for validation.")
        quick_validation_ids = validation_ids[:QUICK_VALIDATION_SIZE]
    else:
        input_sentence = [str(input("Enter data to be tested: "))]

    start = time.time()

    if mode == 1:
        total_loss = 0.0
        correct_outputs = 0
        seen = 0
        train_predictions = []
        train_labels = []
        best_quick_accuracy = -1.0
        best_snapshot = None

        # Fetch the currently loaded values once. They are then kept in memory for
        # the whole run and checkpointed to the database at intervals.
        for layer in conv_layers:
            layer.fetchKernel()
        for layer in all_layers:
            layer.fetch_values()

        for batch_index, batch_ids in enumerate(
                training_batches(train_ids, reviews_per_batch, num_batches, DATA_SEED)):

            print(f'Batch: {batch_index + 1}')

            # One query for the whole batch, rather than one per review.
            rows = com.fetch_batch(batch_ids)
            texts = [row[1] for row in rows]
            star_ratings = [int(row[0]) for row in rows]
            one_hot = one_hot_ratings(star_ratings)

            softmax_output = forward_pass(conv_layers, dense_layers, output_layer, texts)
            predictions = np.argmax(softmax_output, axis=1) + 1

            output_layer.ccel_calculation(one_hot)
            backward_pass(conv_layers, dense_layers, output_layer, one_hot)
            total_loss += output_layer.averageLoss

            correct_outputs += int(np.sum(predictions == np.asarray(star_ratings)))
            seen += len(texts)
            train_predictions.extend(predictions.tolist())
            train_labels.extend(star_ratings)

            # Updating values
            for layer in all_layers:
                layer.adjust_values(LEARNING_RATE)
            for layer in conv_layers:
                layer.adjust_kernel_values(LEARNING_RATE)

            if (batch_index + 1) % CHECKPOINT_EVERY == 0:
                save_parameters(conv_layers, all_layers)

            if (batch_index + 1) % VALIDATION_EVERY == 0:
                quick_exact, quick_within_one, _, _ = evaluate(
                    conv_layers, dense_layers, output_layer, quick_validation_ids)
                print(f"  [batch {batch_index + 1}] quick validation: "
                     f"{100 * quick_exact:.2f}% exact, {100 * quick_within_one:.2f}% within one")
                if quick_exact > best_quick_accuracy:
                    best_quick_accuracy = quick_exact
                    best_snapshot = snapshot_model(conv_layers, all_layers)

        if best_snapshot is not None:
            restore_model(conv_layers, all_layers, best_snapshot)
            print(f"\nRestored best checkpoint seen during training "
                 f"(quick validation accuracy {100 * best_quick_accuracy:.2f}%).")

        save_parameters(conv_layers, all_layers)

        elapsed = time.time() - start
        print(f"\nTotal time elapsed: {elapsed:.2f}s")

        print(f"\nTraining (seen during training, not a fair measure)")
        print(f"  Average loss     : {total_loss / max(num_batches, 1):.4f}")
        print(f"  Accuracy         : {100 * correct_outputs / max(seen, 1):.2f}%")
        print(f"  True labels      : {class_distribution(train_labels)}")
        print(f"  Predictions      : {class_distribution(train_predictions)}")

        exact, within_one, predictions, labels = evaluate(
            conv_layers, dense_layers, output_layer, validation_ids)

        print(f"\nValidation ({len(labels)} held out reviews, never trained on)")
        print(f"  Accuracy         : {100 * exact:.2f}%")
        print(f"  Within one star  : {100 * within_one:.2f}%")
        print(f"  True labels      : {class_distribution(labels)}")
        print(f"  Predictions      : {class_distribution(predictions)}")

        if len(set(predictions.tolist())) == 1:
            print("\n  Warning: every prediction is the same class. The model has "
                  "collapsed onto one output rather than learning to separate them.")
    else:
        for layer in conv_layers:
            layer.fetchKernel()
        for layer in all_layers:
            layer.fetch_values()

        softmax_output = forward_pass(conv_layers, dense_layers, output_layer, input_sentence)
        for prediction in np.argmax(softmax_output, axis=1) + 1:
            print(SENTIMENT_LABELS[int(prediction)])

        print(f"Total time elapsed: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
