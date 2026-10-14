"""AdamW update rule for Wren 1.5, on both layer types.

With a constant gradient, Adam's bias-corrected step is exactly lr in the direction
against the gradient, whatever the gradient's size. Weight decay then pulls weights
(not biases) toward zero by lr * decay * weight. Needs no databases:

    python tests/test_adamw.py
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
DECAY = 0.5


def dense_steps(grad, decay, start=0.0):
    """ (weight, bias) of a dense layer after each of two steps with a constant gradient. """
    layer = NeuralLayer(1)
    layer.weights = np.full((3, 2), start)  # (n_out, n_in)
    layer.biases = np.full(3, start)
    layer.reset_gradients()
    seen = []
    for _ in range(2):
        layer.avdweights = np.full_like(layer.avdweights, grad)
        layer.avdbiases = np.full_like(layer.avdbiases, grad)
        layer.adjust_values(LR, decay)
        assert layer.weights.shape == (3, 2) and layer.biases.shape == (3,)
        assert not layer.avdweights.any(), "gradients not cleared after an update"
        seen.append((layer.weights[0, 0], layer.biases[0]))
    return seen


def conv_steps(grad, decay, start=0.0):
    layer = FilterBankLayer(1, region_size=2, num_filters=3, embedding_dim=4)
    layer.initialize_values()
    layer.kernel = np.full_like(layer.kernel, start)
    layer.bias = np.full_like(layer.bias, start)
    seen = []
    for _ in range(2):
        layer.kernel_derivatives = np.full_like(layer.kernel, grad)
        layer.bias_derivatives = np.full_like(layer.bias, grad)
        layer.adjust_kernel_values(LR, decay)
        assert not layer.kernel_derivatives.any(), "gradients not cleared after an update"
        seen.append((layer.kernel.flat[0], layer.bias[0]))
    return seen


def main():
    for name, steps in (("dense", dense_steps), ("filter bank", conv_steps)):
        # Step size is lr whether the gradient is large or tiny: the point of Adam.
        for grad in (0.5, 1e-4):
            assert np.allclose(steps(grad, 0.0), [(-LR, -LR), (-2 * LR, -2 * LR)], atol=1e-4),                 f"{name}: Adam step is not lr for gradient {grad}"
        # Decoupled weight decay: with no gradient, weights shrink by lr*decay, biases don't.
        assert np.allclose(steps(0.0, DECAY, start=1.0)[0], (1 - LR * DECAY, 1.0)),             f"{name}: weight decay is wrong"
        print(f"pass  {name} AdamW update")


if __name__ == "__main__":
    main()
