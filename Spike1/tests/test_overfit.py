"""Overfit test: the network must be able to memorise a small sample.

A correctly wired network with no regularisation must reach near-100% training
accuracy on a couple of hundred examples. If it cannot, something in the graph is
still broken and there is no point spending hours training on the full dataset.

This is the cheap end-to-end counterpart to the gradient checks: those prove each
layer's derivatives in isolation, this proves the whole assembled network can
actually descend its loss.

Unlike the gradient checks, this one needs the databases. Build them first:

    python InitDatabases.py --synthetic 400

Then run:

    python tests/test_overfit.py                  # 200 reviews, 60 epochs
    python tests/test_overfit.py --reviews 40 --epochs 40    # quicker
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Comms as com  # noqa: E402
import WordVectorConversions as wvc  # noqa: E402
from ConvolutionLayer import ConvLayer  # noqa: E402
from NeuralLayer import NeuralLayer  # noqa: E402

NUM_CLASSES = 5

# A correctly wired network should memorise the sample outright. Anything below
# this and the graph is still wrong somewhere.
REQUIRED_ACCURACY = 0.95


def build_network(seed):
    # initialise_values draws from the global numpy RNG, so seeding here makes the
    # whole run reproducible. Without it the result swings on initialisation luck.
    np.random.seed(seed)

    conv_layers = (ConvLayer(1, 5), ConvLayer(2, 3))
    dense_layers = [NeuralLayer(n) for n in range(1, 6)]
    output_layer = NeuralLayer(6)
    all_layers = dense_layers + [output_layer]

    for layer in conv_layers:
        layer.initialize_values()
    for layer in all_layers:
        layer.initialise_values()

    for layer in conv_layers:
        layer.fetchKernel()
    for layer in all_layers:
        layer.fetch_values()

    return conv_layers, dense_layers, output_layer, all_layers


def train_step(conv_layers, dense_layers, output_layer, all_layers,
               embeddings, one_hot, stars, learning_rate):
    """One forward pass, one backward pass and one parameter update."""
    conv1, conv2 = conv_layers

    conv_output = conv1.convPass(conv1.reflectMatrix(embeddings))
    conv_output = conv2.convPass(conv2.reflectMatrix(conv_output))

    activations = np.asarray(conv_output, dtype=float).reshape(len(stars), -1)
    for layer in dense_layers:
        activations = layer.batch_layer_output(activations)
    softmax_output = output_layer.softmax(activations)[1]

    output_layer.ccel_calculation(one_hot)
    loss = output_layer.averageLoss
    correct = int(np.sum(np.argmax(softmax_output, axis=1) + 1 == stars))

    gradient = output_layer.calculate_derivatives(output_layer.combined_derivative(one_hot))
    for layer in reversed(dense_layers):
        gradient = layer.calculate_derivatives(layer.relu_backward(gradient))

    gradient = np.asarray(gradient).reshape(np.asarray(conv2.output, dtype=float).shape)
    gradient = conv2.backpropagate(gradient, False)
    conv1.backpropagate(gradient, False)

    for layer in all_layers:
        layer.adjust_values(learning_rate)
    for layer in conv_layers:
        layer.adjust_kernel_values(learning_rate)

    return loss, correct


def run_overfit(num_reviews, epochs, batch_size, learning_rate, seed=0, quiet=False):
    conv_layers, dense_layers, output_layer, all_layers = build_network(seed)

    review_ids = com.fetch_all_review_ids()[:num_reviews]
    if len(review_ids) < num_reviews:
        raise AssertionError(
            f"only {len(review_ids)} reviews in the database, needed {num_reviews}. "
            "Run InitDatabases.py with a larger --synthetic count, or import real data.")

    rows = com.fetch_batch(review_ids)
    texts = [row[1] for row in rows]
    stars = np.array([int(row[0]) for row in rows])

    # Embed once. The embeddings are fixed, only the network is trained.
    embeddings = np.asarray(
        wvc.pad_matrix(wvc.return_vector_matrix_jsonl(wvc.format_entry_data(texts))),
        dtype=float)

    one_hot = np.zeros((num_reviews, NUM_CLASSES))
    one_hot[np.arange(num_reviews), stars - 1] = 1

    accuracy = 0.0
    started = time.time()

    for epoch in range(epochs):
        order = np.random.default_rng(epoch).permutation(num_reviews)
        total_loss = 0.0
        correct = 0

        for start in range(0, num_reviews, batch_size):
            selection = order[start:start + batch_size]
            loss, batch_correct = train_step(
                conv_layers, dense_layers, output_layer, all_layers,
                embeddings[selection], one_hot[selection], stars[selection],
                learning_rate)
            total_loss += loss * len(selection)
            correct += batch_correct

        accuracy = correct / num_reviews
        if not quiet and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch + 1:>4}  loss {total_loss / num_reviews:.4f}  "
                  f"train accuracy {100 * accuracy:6.2f}%")

    if not quiet:
        print(f"    finished in {time.time() - started:.1f}s")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Overfitting {args.reviews} reviews for {args.epochs} epochs "
          f"(batch {args.batch_size}, learning rate {args.learning_rate}, "
          f"seed {args.seed}):")

    try:
        accuracy = run_overfit(args.reviews, args.epochs, args.batch_size,
                               args.learning_rate, args.seed)
    except FileNotFoundError as error:
        print(f"FAIL  {error}")
        return 1

    if accuracy < REQUIRED_ACCURACY:
        print(f"\nFAIL  reached only {100 * accuracy:.2f}% training accuracy, needed "
              f"{100 * REQUIRED_ACCURACY:.0f}%.\n"
              "      A correctly wired network must be able to memorise this sample.\n"
              "      Run tests/test_gradients.py and tests/test_conv_gradients.py first.")
        return 1

    print(f"\npass  overfit {args.reviews} reviews to {100 * accuracy:.2f}% training accuracy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
