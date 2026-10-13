"""SGD + momentum update rule for Wren 1.5, on both layer types.

With a constant gradient g, two steps must move a parameter by -lr*g and then
-lr*g*(1 + momentum), and momentum=0 must be exactly plain SGD. Needs no databases:

    python tests/test_momentum.py
"""

import importlib.util
import os
import sys

import numpy as np

# src/wren-1.5 isn't a valid package name, so load it by path as `wren15`.
HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(HERE, "..", "src", "wren-1.5")
spec = importlib.util.spec_from_file_location(
    "wren15", os.path.join(PACKAGE_DIR, "__init__.py"), submodule_search_locations=[PACKAGE_DIR])
sys.modules["wren15"] = module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from wren15.layers.filter_bank import FilterBankLayer  # noqa: E402
from wren15.layers.neural import NeuralLayer  # noqa: E402

LR = 0.1
GRAD = 0.5


def dense_steps(momentum):
    """ Weight [0, 0] of a dense layer after each of two steps with a constant gradient. """
    layer = NeuralLayer(1)
    layer.weights = np.zeros((3, 2))  # (n_out, n_in)
    layer.biases = np.zeros(3)
    layer.reset_gradients()
    seen = []
    for _ in range(2):
        layer.avdweights = np.full_like(layer.avdweights, GRAD)
        layer.avdbiases = np.full_like(layer.avdbiases, GRAD)
        layer.adjust_values(LR, momentum)
        assert np.allclose(layer.weights, layer.weights[0, 0]) and np.allclose(layer.biases, layer.weights[0, 0])
        assert not layer.avdweights.any(), "gradients not cleared after an update"
        seen.append(layer.weights[0, 0])
    return seen


def conv_steps(momentum):
    layer = FilterBankLayer(1, region_size=2, num_filters=3, embedding_dim=4)
    layer.initialize_values()
    layer.kernel = np.zeros_like(layer.kernel)
    layer.bias = np.zeros_like(layer.bias)
    seen = []
    for _ in range(2):
        layer.kernel_derivatives = np.full_like(layer.kernel, GRAD)
        layer.bias_derivatives = np.full_like(layer.bias, GRAD)
        layer.adjust_kernel_values(LR, momentum)
        assert np.allclose(layer.kernel, layer.kernel.flat[0]) and np.allclose(layer.bias, layer.kernel.flat[0])
        seen.append(layer.kernel.flat[0])
    return seen


def main():
    step = LR * GRAD
    for name, steps in (("dense", dense_steps), ("filter bank", conv_steps)):
        assert np.allclose(steps(0.0), [-step, -2 * step]), f"{name}: momentum=0 is not plain SGD"
        assert np.allclose(steps(0.9), [-step, -step - 1.9 * step]), f"{name}: momentum step is wrong"
        print(f"pass  {name} momentum update")


if __name__ == "__main__":
    main()
