"""Numerical gradient check for FilterBankLayer.

Same idea as test_conv_gradients.py, applied to the new Kim (2014) style filter bank
that replaced the two-stage single-channel ConvLayer stack: build a small layer, take
a random linear functional of its output as the loss, and compare the analytic kernel,
bias and input gradients against central finite differences.

The layer is constructed directly rather than through fetchKernel(), so the test needs
no database.

Run it with:

    python tests/test_filterbank_gradients.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sentiment"))

from FilterBankLayer import FilterBankLayer  # noqa: E402

TOLERANCE = 1e-5
EPSILON = 1e-6

# (region_size, num_filters, sequence_length, embedding_dim)
CONFIGURATIONS = [
    (3, 4, 10, 6),
    (4, 3, 12, 6),
    (5, 2, 20, 6),
    (3, 5, 9, 8),
]

BATCH_SIZE = 3


def build_layer(region_size, num_filters, embedding_dim, seed):
    """Creates a FilterBankLayer with a random kernel, bypassing the database."""
    rng = np.random.default_rng(seed)
    layer = FilterBankLayer(1, region_size, num_filters, embedding_dim)
    layer.kernel = rng.normal(0, 0.5, (num_filters, region_size, embedding_dim))
    layer.bias = rng.normal(0, 0.5, (num_filters,))
    layer.kernel_derivatives = np.zeros_like(layer.kernel)
    layer.bias_derivatives = np.zeros_like(layer.bias)
    return layer


def relative_error(analytic, numerical):
    return abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-8)


def check_configuration(region_size, num_filters, seq_len, embedding_dim, seed=0):
    rng = np.random.default_rng(seed)
    layer = build_layer(region_size, num_filters, embedding_dim, seed)
    inputs = rng.normal(0, 1.0, (BATCH_SIZE, seq_len, embedding_dim))

    output = layer.convPass(inputs)
    # Loss is a random linear functional of the output, so d(loss)/d(output) is weights.
    weights = rng.normal(0, 1.0, output.shape)

    layer.kernel_derivatives = np.zeros_like(layer.kernel)
    layer.bias_derivatives = np.zeros_like(layer.bias)
    input_gradient = layer.backpropagate(weights, need_input_grad=True)
    kernel_gradient = np.array(layer.kernel_derivatives)
    bias_gradient = np.array(layer.bias_derivatives)

    assert input_gradient.shape == inputs.shape, (
        f"input gradient shape {input_gradient.shape} does not match the layer input "
        f"{inputs.shape}")

    def forward():
        return layer.convPass(inputs)

    worst_kernel = 0.0
    for f in range(num_filters):
        for i in range(region_size):
            for j in range(embedding_dim):
                original = layer.kernel[f, i, j]

                layer.kernel[f, i, j] = original + EPSILON
                loss_plus = np.sum(weights * forward())

                layer.kernel[f, i, j] = original - EPSILON
                loss_minus = np.sum(weights * forward())

                layer.kernel[f, i, j] = original

                numerical = (loss_plus - loss_minus) / (2 * EPSILON)
                worst_kernel = max(worst_kernel,
                                   relative_error(kernel_gradient[f, i, j], numerical))

    worst_bias = 0.0
    for f in range(num_filters):
        original = layer.bias[f]

        layer.bias[f] = original + EPSILON
        loss_plus = np.sum(weights * forward())

        layer.bias[f] = original - EPSILON
        loss_minus = np.sum(weights * forward())

        layer.bias[f] = original

        numerical = (loss_plus - loss_minus) / (2 * EPSILON)
        worst_bias = max(worst_bias, relative_error(bias_gradient[f], numerical))

    # Probe the input gradient at the argmax position of each (batch, filter) pair,
    # and at a handful of arbitrary non-argmax positions where it should be exactly 0.
    worst_input = 0.0
    probes = []
    for b in range(BATCH_SIZE):
        for f in range(min(num_filters, 3)):
            start = int(layer.argmax_positions[b, f])
            probes.append((b, start, 0))  # inside the winning window
    for b in range(BATCH_SIZE):
        probes.append((b, seq_len - 1, embedding_dim - 1))  # likely outside any argmax window

    for batch_index, row, column in probes:
        original = inputs[batch_index, row, column]

        inputs[batch_index, row, column] = original + EPSILON
        loss_plus = np.sum(weights * forward())

        inputs[batch_index, row, column] = original - EPSILON
        loss_minus = np.sum(weights * forward())

        inputs[batch_index, row, column] = original

        numerical = (loss_plus - loss_minus) / (2 * EPSILON)
        worst_input = max(worst_input,
                          relative_error(input_gradient[batch_index, row, column], numerical))

    return worst_kernel, worst_bias, worst_input, output.shape


def test_filterbank_gradients():
    failures = []
    worst_overall = 0.0

    for region_size, num_filters, seq_len, embedding_dim in CONFIGURATIONS:
        worst_kernel, worst_bias, worst_input, output_shape = check_configuration(
            region_size, num_filters, seq_len, embedding_dim)
        worst_overall = max(worst_overall, worst_kernel, worst_bias, worst_input)

        label = (f"region {region_size}, {num_filters} filters, "
                f"seq_len {seq_len}, dim {embedding_dim}")
        if worst_kernel > TOLERANCE:
            failures.append(f"{label}: kernel gradient rel_err {worst_kernel:.3e}")
        if worst_bias > TOLERANCE:
            failures.append(f"{label}: bias gradient rel_err {worst_bias:.3e}")
        if worst_input > TOLERANCE:
            failures.append(f"{label}: input gradient rel_err {worst_input:.3e}")

    if failures:
        raise AssertionError("filter bank gradients exceed tolerance:\n  "
                             + "\n  ".join(failures))

    print(f"  {len(CONFIGURATIONS)} configurations checked, worst relative error "
          f"{worst_overall:.3e}")


def test_relu_is_applied():
    """convPass must clamp negatives; output must never be negative."""
    layer = build_layer(3, 4, 6, seed=6)
    rng = np.random.default_rng(6)
    inputs = rng.normal(0, 1.0, (2, 10, 6))

    output = layer.convPass(inputs)

    assert np.all(output >= 0), "convPass output contains negative values, ReLU not applied"


def test_padding_rows_are_never_selected_over_real_content():
    """A window built entirely from constant padding should not win the max-pool
    against a window that contains a strong, non-constant signal."""
    layer = build_layer(3, 4, 6, seed=9)
    rng = np.random.default_rng(9)

    inputs = np.full((1, 20, 6), 0.001)  # all padding
    inputs[0, 5:8, :] = rng.normal(5.0, 1.0, (3, 6))  # one strong real window at position 5

    layer.convPass(inputs)

    # Every filter's winning window should overlap the real content at position 5,
    # not sit entirely inside the padding.
    for f in range(4):
        start = int(layer.argmax_positions[0, f])
        overlaps_real_content = not (start + layer.region_size <= 5 or start > 7)
        assert overlaps_real_content, (
            f"filter {f} selected a pure-padding window at position {start} instead of "
            f"the real content at position 5")


def main():
    tests = [
        ("ReLU is applied", test_relu_is_applied),
        ("padding rows are never selected over real content", test_padding_rows_are_never_selected_over_real_content),
        ("kernel, bias and input gradients match finite differences", test_filterbank_gradients),
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
