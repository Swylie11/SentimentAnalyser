"""Numerical gradient check for the convolution layer.

Same idea as test_gradients.py, applied to ConvLayer: build a small layer, take a
random linear functional of its output as the loss, and compare the analytic kernel
and input gradients against central finite differences.

This test was written after the dense-stack check passed and it immediately found
three defects: convPass emitted duplicated rows and columns at its trailing edges,
the backward pass rotated the kernel although the forward pass is a cross
correlation, and the backward pass assumed output position i came from input
position i*stride, which the duplicated positions violated. Relative errors were
around 1.5, including sign flips at interior positions.

The layer is constructed directly rather than through fetchKernel(), so the test
needs no database.

Run it with:

    python tests/test_conv_gradients.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConvolutionLayer import ConvLayer  # noqa: E402

TOLERANCE = 1e-5
EPSILON = 1e-6

# (stride, kernel size, input height, input width)
CONFIGURATIONS = [
    (1, 3, 8, 9),
    (2, 3, 8, 9),
    (3, 3, 9, 11),
    (5, 5, 12, 14),
    (5, 5, 20, 24),   # The strides main.py actually uses
    (3, 3, 20, 24),
]

BATCH_SIZE = 2


def build_layer(stride, kernel_size, seed):
    """Creates a ConvLayer with a random kernel, bypassing the database."""
    rng = np.random.default_rng(seed)
    layer = ConvLayer(1, stride)
    layer.kernel = rng.normal(0, 0.5, (kernel_size, kernel_size))
    layer.filter_derivatives = np.zeros_like(layer.kernel)
    return layer


def forward(layer, inputs):
    return np.asarray(layer.convPass(layer.reflectMatrix(inputs)), dtype=float)


def relative_error(analytic, numerical):
    return abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-8)


def check_configuration(stride, kernel_size, height, width, seed=0):
    rng = np.random.default_rng(seed)
    layer = build_layer(stride, kernel_size, seed)
    inputs = rng.normal(0, 1.0, (BATCH_SIZE, height, width))

    output = forward(layer, inputs)
    # Loss is a random linear functional of the output, so d(loss)/d(output) is weights.
    weights = rng.normal(0, 1.0, output.shape)

    layer.filter_derivatives = np.zeros_like(layer.kernel)
    input_gradient = layer.backpropagate(weights, False)
    kernel_gradient = np.array(layer.filter_derivatives)

    assert input_gradient.shape == inputs.shape, (
        f"input gradient shape {input_gradient.shape} does not match the layer input "
        f"{inputs.shape}; the padding gradient has not been folded back")

    worst_kernel = 0.0
    for row in range(kernel_size):
        for column in range(kernel_size):
            original = layer.kernel[row, column]

            layer.kernel[row, column] = original + EPSILON
            loss_plus = np.sum(weights * forward(layer, inputs))

            layer.kernel[row, column] = original - EPSILON
            loss_minus = np.sum(weights * forward(layer, inputs))

            layer.kernel[row, column] = original

            numerical = (loss_plus - loss_minus) / (2 * EPSILON)
            worst_kernel = max(worst_kernel,
                               relative_error(kernel_gradient[row, column], numerical))

    # Probe the input gradient at corners, edges and interior, where the padding
    # reflection and the stride clamping are most likely to disagree.
    probes = [(0, 0, 0), (0, 0, width // 2), (0, height // 2, 0),
              (0, height // 2, width // 2), (1, height - 1, width - 1),
              (1, height - 1, width // 2), (1, height // 2, width - 1)]

    worst_input = 0.0
    for batch_index, row, column in probes:
        original = inputs[batch_index, row, column]

        inputs[batch_index, row, column] = original + EPSILON
        loss_plus = np.sum(weights * forward(layer, inputs))

        inputs[batch_index, row, column] = original - EPSILON
        loss_minus = np.sum(weights * forward(layer, inputs))

        inputs[batch_index, row, column] = original

        numerical = (loss_plus - loss_minus) / (2 * EPSILON)
        worst_input = max(worst_input,
                          relative_error(input_gradient[batch_index, row, column], numerical))

    return worst_kernel, worst_input, output.shape


def test_conv_gradients():
    failures = []
    worst_overall = 0.0

    for stride, kernel_size, height, width in CONFIGURATIONS:
        worst_kernel, worst_input, output_shape = check_configuration(
            stride, kernel_size, height, width)
        worst_overall = max(worst_overall, worst_kernel, worst_input)

        label = f"stride {stride}, kernel {kernel_size}x{kernel_size}, input {height}x{width}"
        if worst_kernel > TOLERANCE:
            failures.append(f"{label}: kernel gradient rel_err {worst_kernel:.3e}")
        if worst_input > TOLERANCE:
            failures.append(f"{label}: input gradient rel_err {worst_input:.3e}")

    if failures:
        raise AssertionError("convolution gradients exceed tolerance:\n  "
                             + "\n  ".join(failures))

    print(f"  {len(CONFIGURATIONS)} configurations checked, worst relative error "
          f"{worst_overall:.3e}")


def test_output_grid_has_no_duplicates():
    """A stride of 1 must give a 'same' sized output with no repeated edge rows."""
    rng = np.random.default_rng(4)
    layer = build_layer(stride=1, kernel_size=3, seed=4)
    inputs = rng.normal(0, 1.0, (1, 6, 7))

    output = forward(layer, inputs)

    assert output.shape == (1, 6, 7), (
        f"stride 1 with a 3x3 kernel and reflection padding should preserve the input "
        f"size, got {output.shape}")

    rows = output[0]
    assert not np.allclose(rows[-1], rows[-2]), \
        "trailing rows are identical, the output grid is repeating its last position"


def test_relu_is_applied():
    """convPass must clamp negatives, and keep the pre-activation for the backward pass."""
    layer = build_layer(stride=1, kernel_size=3, seed=6)
    rng = np.random.default_rng(6)
    inputs = rng.normal(0, 1.0, (1, 6, 7))

    output = forward(layer, inputs)

    assert np.all(output >= 0), "convPass output contains negative values, ReLU not applied"
    assert layer.pre_activation is not None, "pre-activation not stored for the backward pass"
    assert np.any(np.asarray(layer.pre_activation) < 0), \
        "test is not exercising the clamp: no pre-activation was negative"
    assert np.allclose(output, np.maximum(0, layer.pre_activation)), \
        "output is not the ReLU of the stored pre-activation"


def test_reflect_matrix_does_not_mutate_its_argument():
    """reflectMatrix must build a new array, not pad the caller's data in place."""
    layer = build_layer(stride=1, kernel_size=3, seed=8)
    original = [[[1.0, 2.0], [3.0, 4.0]]]
    before = [[row[:] for row in matrix] for matrix in original]

    layer.reflectMatrix(original)

    assert original == before, "reflectMatrix modified the list it was given"


def main():
    tests = [
        ("stride 1 output grid has no duplicated edges", test_output_grid_has_no_duplicates),
        ("ReLU is applied and the pre-activation kept", test_relu_is_applied),
        ("reflectMatrix leaves its argument alone", test_reflect_matrix_does_not_mutate_its_argument),
        ("convolution gradients match finite differences", test_conv_gradients),
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
