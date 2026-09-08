"""Numerical gradient check for the dense stack.

This is the test that distinguishes "the gradients look right" from "the gradients
are right". It builds a small network by hand, runs one backward pass, and compares
every analytic weight and bias gradient against a central finite difference of the
loss. If the chain rule is broken anywhere between the loss and a parameter, the
relative error explodes and this fails.

The layers are constructed directly rather than through fetch_values(), so the test
needs no database and no data.

Run it with:

    python tests/test_gradients.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NeuralLayer import NeuralLayer  # noqa: E402

# A wrong gradient shows up as a relative error many orders of magnitude above this.
# Central differences at eps=1e-5 in float64 bottom out around 1e-9.
TOLERANCE = 1e-5
EPSILON = 1e-5

# 8 input features -> two hidden layers of 4 units -> 5 classes
LAYER_WIDTHS = [8, 4, 4, 5]
BATCH_SIZE = 3


def build_network(seed=0):
    """Creates the layer stack with random weights, bypassing the database."""
    rng = np.random.default_rng(seed)
    layers = []

    for index in range(1, len(LAYER_WIDTHS)):
        n_in = LAYER_WIDTHS[index - 1]
        n_out = LAYER_WIDTHS[index]

        layer = NeuralLayer(index)
        # Stored orientation is (n_out, n_in), matching what batch_layer_output expects.
        layer.weights = rng.normal(0, 0.5, (n_out, n_in))
        layer.biases = rng.normal(0, 0.5, n_out)
        reset_accumulators(layer)
        layers.append(layer)

    return layers


def reset_accumulators(layer):
    """Zeroes the running gradient totals, as fetch_values does in the real run."""
    n_out, n_in = np.asarray(layer.weights).shape
    layer.avdweights = np.zeros((n_in, n_out))
    layer.avdbiases = np.zeros((1, n_out))


def forward(layers, inputs):
    """Runs the ReLU stack followed by the softmax output layer."""
    activations = inputs
    for layer in layers[:-1]:
        activations = layer.batch_layer_output(activations)
    layers[-1].softmax(activations)
    return layers[-1].softmax_output


def loss_of(layers, inputs, one_hot):
    """Mean categorical cross entropy loss over the batch."""
    forward(layers, inputs)
    layers[-1].ccel_calculation(one_hot)
    return layers[-1].averageLoss


def backward(layers, one_hot):
    """One backward pass, leaving the gradients in each layer's accumulators."""
    output_layer = layers[-1]
    grad = output_layer.calculate_derivatives(output_layer.combined_derivative(one_hot))
    for layer in reversed(layers[:-1]):
        grad = layer.calculate_derivatives(layer.relu_backward(grad))
    return grad


def relative_error(analytic, numerical):
    return abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-8)


def check_gradients():
    rng = np.random.default_rng(1)
    layers = build_network()

    inputs = rng.normal(0, 1.0, (BATCH_SIZE, LAYER_WIDTHS[0]))
    labels = rng.integers(0, LAYER_WIDTHS[-1], BATCH_SIZE)
    one_hot = np.zeros((BATCH_SIZE, LAYER_WIDTHS[-1]))
    one_hot[np.arange(BATCH_SIZE), labels] = 1

    # Analytic gradients from one backward pass.
    for layer in layers:
        reset_accumulators(layer)
    loss_of(layers, inputs, one_hot)
    backward(layers, one_hot)

    # avdweights is stored transposed relative to weights, so index [i, o] is the
    # gradient of weights[o, i].
    analytic_weights = [np.array(layer.avdweights) for layer in layers]
    analytic_biases = [np.array(layer.avdbiases).reshape(-1) for layer in layers]

    worst = 0.0
    worst_where = None
    checked = 0
    failures = []

    for layer_index, layer in enumerate(layers):
        n_out, n_in = np.asarray(layer.weights).shape

        for o in range(n_out):
            for i in range(n_in):
                original = layer.weights[o, i]

                layer.weights[o, i] = original + EPSILON
                loss_plus = loss_of(layers, inputs, one_hot)

                layer.weights[o, i] = original - EPSILON
                loss_minus = loss_of(layers, inputs, one_hot)

                layer.weights[o, i] = original

                numerical = (loss_plus - loss_minus) / (2 * EPSILON)
                analytic = analytic_weights[layer_index][i, o]
                error = relative_error(analytic, numerical)
                checked += 1

                if error > worst:
                    worst, worst_where = error, f"layer {layer_index + 1} weight[{o}, {i}]"
                if error > TOLERANCE:
                    failures.append((f"layer {layer_index + 1} weight[{o}, {i}]",
                                     analytic, numerical, error))

        for o in range(n_out):
            original = layer.biases[o]

            layer.biases[o] = original + EPSILON
            loss_plus = loss_of(layers, inputs, one_hot)

            layer.biases[o] = original - EPSILON
            loss_minus = loss_of(layers, inputs, one_hot)

            layer.biases[o] = original

            numerical = (loss_plus - loss_minus) / (2 * EPSILON)
            analytic = analytic_biases[layer_index][o]
            error = relative_error(analytic, numerical)
            checked += 1

            if error > worst:
                worst, worst_where = error, f"layer {layer_index + 1} bias[{o}]"
            if error > TOLERANCE:
                failures.append((f"layer {layer_index + 1} bias[{o}]",
                                 analytic, numerical, error))

    return checked, worst, worst_where, failures


def check_input_gradient():
    """Checks the gradient handed back to the convolution layers below the stack."""
    rng = np.random.default_rng(2)
    layers = build_network(seed=5)

    inputs = rng.normal(0, 1.0, (BATCH_SIZE, LAYER_WIDTHS[0]))
    labels = rng.integers(0, LAYER_WIDTHS[-1], BATCH_SIZE)
    one_hot = np.zeros((BATCH_SIZE, LAYER_WIDTHS[-1]))
    one_hot[np.arange(BATCH_SIZE), labels] = 1

    for layer in layers:
        reset_accumulators(layer)
    loss_of(layers, inputs, one_hot)
    analytic = np.array(backward(layers, one_hot))

    assert analytic.shape == inputs.shape, (
        f"input gradient shape {analytic.shape} does not match inputs {inputs.shape}; "
        "the batch axis must survive the backward pass")

    worst = 0.0
    for b in range(BATCH_SIZE):
        for f in range(LAYER_WIDTHS[0]):
            original = inputs[b, f]

            inputs[b, f] = original + EPSILON
            loss_plus = loss_of(layers, inputs, one_hot)

            inputs[b, f] = original - EPSILON
            loss_minus = loss_of(layers, inputs, one_hot)

            inputs[b, f] = original

            numerical = (loss_plus - loss_minus) / (2 * EPSILON)
            worst = max(worst, relative_error(analytic[b, f], numerical))

    return worst


def test_relu_backward_does_not_mutate_state():
    """relu_backward must not write through to the stored forward activations."""
    layer = NeuralLayer(1)
    layer.weights = np.array([[1.0, -1.0], [-1.0, 1.0]])
    layer.biases = np.array([0.0, 0.0])
    reset_accumulators(layer)

    layer.batch_layer_output(np.array([[1.0, 2.0], [3.0, 1.0]]))
    before_output = np.array(layer.output)
    before_layer_output = np.array(layer.layer_output)

    layer.relu_backward(np.ones_like(before_output))

    assert np.array_equal(layer.output, before_output), "relu_backward mutated self.output"
    assert np.array_equal(layer.layer_output, before_layer_output), \
        "relu_backward mutated self.layer_output"


def test_relu_backward_gates_on_preactivation():
    """Units whose pre-activation was negative must receive no gradient."""
    layer = NeuralLayer(1)
    layer.weights = np.array([[1.0, 0.0], [-1.0, 0.0]])
    layer.biases = np.array([0.0, 0.0])
    reset_accumulators(layer)

    layer.batch_layer_output(np.array([[2.0, 0.0]]))
    # Pre-activations are [2, -2], so the second unit is clamped.
    gradient = layer.relu_backward(np.array([[1.0, 1.0]]))

    assert gradient[0, 0] == 1.0, "gradient blocked on an active unit"
    assert gradient[0, 1] == 0.0, "gradient passed through a clamped unit"


def test_relu_derivative_is_removed():
    """The old broken method must fail loudly rather than return garbage."""
    layer = NeuralLayer(1)
    try:
        layer.ReLU_derivative()
    except NotImplementedError:
        return
    raise AssertionError("ReLU_derivative should raise NotImplementedError")


def test_gradients():
    checked, worst, worst_where, failures = check_gradients()

    if failures:
        lines = [f"  {name}: analytic={a:.10f} numerical={n:.10f} rel_err={e:.3e}"
                 for name, a, n, e in failures[:20]]
        raise AssertionError(
            f"{len(failures)} of {checked} parameter gradients exceed {TOLERANCE}:\n"
            + "\n".join(lines))

    print(f"  {checked} parameter gradients checked, worst relative error "
          f"{worst:.3e} at {worst_where}")


def test_input_gradient():
    worst = check_input_gradient()
    if worst > TOLERANCE:
        raise AssertionError(
            f"input gradient relative error {worst:.3e} exceeds {TOLERANCE}")
    print(f"  input gradient checked, worst relative error {worst:.3e}")


def main():
    tests = [
        ("relu_backward gates on the pre-activation", test_relu_backward_gates_on_preactivation),
        ("relu_backward leaves stored state alone", test_relu_backward_does_not_mutate_state),
        ("ReLU_derivative raises", test_relu_derivative_is_removed),
        ("parameter gradients match finite differences", test_gradients),
        ("input gradient matches finite differences", test_input_gradient),
    ]

    failed = 0
    for name, test in tests:
        try:
            test()
        except AssertionError as error:
            failed += 1
            print(f"FAIL  {name}\n{error}")
        else:
            print(f"pass  {name}")

    if failed:
        print(f"\n{failed} of {len(tests)} checks failed.")
        return 1

    print(f"\nAll {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
