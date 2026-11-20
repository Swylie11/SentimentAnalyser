"""Dropout on Wren 1.5's filter bank features: the backward pass must route the
gradient through the same mask the forward pass used. Needs no databases:

    python tests/test_feature_dropout.py
"""

import numpy as np

from cadello.wren.layers.filter_bank import FilterBankLayer

EPSILON = 1e-6
MASK_SEED = 7


def main():
    rng = np.random.default_rng(0)
    layer = FilterBankLayer(1, 3, 8, embedding_dim=5)
    layer.kernel = rng.normal(0, 0.5, (8, 3, 5))
    layer.bias = rng.normal(0, 0.5, 8)
    layer.kernel_derivatives = np.zeros_like(layer.kernel)
    layer.bias_derivatives = np.zeros_like(layer.bias)
    inputs = rng.normal(0, 1, (4, 10, 5))
    upstream = rng.normal(0, 1, (4, 8))

    def loss():
        np.random.seed(MASK_SEED)  # Same mask on every call
        return float(np.sum(upstream * layer.convPass(inputs, training=True)))

    loss()
    mask = layer.dropout_mask
    assert 0 < np.mean(mask == 0) < 1, "expected some features dropped and some kept"
    assert np.allclose(mask[mask > 0], 1 / (1 - layer.dropout_rate)), "kept features not rescaled"

    layer.backpropagate(upstream)
    worst = 0.0
    for index in np.ndindex(layer.kernel.shape):
        original = layer.kernel[index]
        layer.kernel[index] = original + EPSILON
        plus = loss()
        layer.kernel[index] = original - EPSILON
        minus = loss()
        layer.kernel[index] = original
        numerical = (plus - minus) / (2 * EPSILON)
        analytic = layer.kernel_derivatives[index]
        worst = max(worst, abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-8))
    assert worst < 1e-5, f"kernel gradient through dropout off by {worst:.2e}"

    # An evaluation pass applies no dropout and clears the mask, so a later
    # backpropagate can't reuse a stale one.
    output = layer.convPass(inputs)
    assert layer.dropout_mask is None
    assert np.all(output == layer.output)

    print(f"Feature dropout OK (worst kernel gradient error {worst:.2e})")


if __name__ == "__main__":
    main()
