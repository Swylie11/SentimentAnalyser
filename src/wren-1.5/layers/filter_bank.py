import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..train import comms as com


class FilterBankLayer:
    """ One region-size branch of a Kim (2014) style text CNN filter bank.

    Replaces the old two-stage single-channel ConvLayer stack. Each filter has shape
    (region_size, embedding_dim): its width always equals the FULL embedding
    dimension, so it only ever slides along the word axis and never mixes unrelated
    embedding dimensions the way striding across that axis did. A filter's
    cross-correlation over a review is ReLU'd then globally max-pooled over word
    position, so the branch emits one fixed-size feature per filter regardless of
    review length -- padding rows (constant 0.001) produce near-zero activations and
    essentially never win the max, so no explicit masking is needed.

    main.py runs several of these in parallel over the same embedding input (one per
    region size) and concatenates their outputs, rather than stacking them. """

    def __init__(self, layerNum, region_size, num_filters, embedding_dim=300):
        self.layerNum = layerNum
        self.region_size = region_size
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim

        self.kernel = None            # (num_filters, region_size, embedding_dim)
        self.bias = None              # (num_filters,)
        self.kernel_derivatives = None
        self.bias_derivatives = None
        self.kernel_velocity = None   # Momentum velocities, same shapes as kernel and bias
        self.bias_velocity = None

        self.inputs = None            # (batch, T, embedding_dim), kept for the backward gather
        self.argmax_positions = None  # (batch, num_filters) int, the winning word position per filter
        self.output = None            # (batch, num_filters)

    def fetchKernel(self):
        kernel, bias = com.fetch_kernel(self.layerNum)
        self.kernel = np.array(kernel, dtype=float)
        self.bias = np.array(bias, dtype=float)
        self.kernel_derivatives = np.zeros_like(self.kernel)
        self.bias_derivatives = np.zeros_like(self.bias)
        self.kernel_velocity = np.zeros_like(self.kernel)
        self.bias_velocity = np.zeros_like(self.bias)

    def initialize_values(self):
        """ Draws fresh kernels in memory only, without touching the database, so a
        multi-seed run can't overwrite its saved champion. Call save() to store them. """
        # He initialisation: the fan in of a filter is every cell of the window it
        # sees, i.e. region_size * embedding_dim.
        fan_in = self.region_size * self.embedding_dim
        standard_dev = np.sqrt(2 / fan_in)
        self.kernel = np.random.normal(0, standard_dev, (self.num_filters, self.region_size, self.embedding_dim))
        self.bias = np.full(self.num_filters, 0.01)

        self.kernel_derivatives = np.zeros_like(self.kernel)
        self.bias_derivatives = np.zeros_like(self.bias)
        self.kernel_velocity = np.zeros_like(self.kernel)
        self.bias_velocity = np.zeros_like(self.bias)

    def _windows(self, batch):
        """ (batch, T, D) -> (batch, positions, region_size, embedding_dim) sliding windows. """
        r, D = self.region_size, self.embedding_dim
        return sliding_window_view(batch, (r, D), axis=(1, 2))[:, :, 0, :, :]

    def convPass(self, batch_input):
        """ Correlates every filter against every word window, applies ReLU, then
        globally max-pools over word position. Returns (batch, num_filters). """
        batch = np.asarray(batch_input, dtype=float)
        self.inputs = batch

        r, D, F = self.region_size, self.embedding_dim, self.num_filters
        windows = self._windows(batch)                      # (batch, positions, r, D)
        batch_size, positions = windows.shape[0], windows.shape[1]

        flat_windows = windows.reshape(batch_size * positions, r * D)
        flat_kernel = self.kernel.reshape(F, r * D)
        pre_activation = (flat_windows @ flat_kernel.T).reshape(batch_size, positions, F)
        pre_activation = pre_activation + self.bias

        activated = np.maximum(0, pre_activation)           # (batch, positions, F)
        self.argmax_positions = np.argmax(activated, axis=1)  # (batch, F)

        batch_indices = np.arange(batch_size)[:, None]
        filter_indices = np.arange(F)[None, :]
        self.output = activated[batch_indices, self.argmax_positions, filter_indices]
        return self.output

    def backpropagate(self, dvalues, need_input_grad=False):
        """ dvalues: gradient wrt this branch's pooled output, shape (batch, num_filters).

        Accumulates kernel and bias gradients into kernel_derivatives/bias_derivatives,
        applied by adjust_kernel_values. Returns the gradient wrt this branch's input
        only if need_input_grad is set -- embeddings are not currently trained, so
        main.py's training loop never needs it and skips the extra work; the gradient
        check exercises it to verify the routing is correct. """
        dvalues = np.asarray(dvalues, dtype=float)
        batch_size = self.inputs.shape[0]
        r, D, F = self.region_size, self.embedding_dim, self.num_filters

        # Max-pool routes the gradient to the winning position only; ReLU gates on
        # whether that position was clamped. output IS the post-ReLU value at the
        # winning position, so gating on output>0 is equivalent to gating on the
        # pre-activation there (same identity NeuralLayer.relu_backward relies on).
        gate = dvalues * (self.output > 0)                  # (batch, F)

        batch_indices = np.arange(batch_size)
        windows = self._windows(self.inputs)                # (batch, positions, r, D)
        winning_windows = windows[batch_indices[:, None], self.argmax_positions]  # (batch, F, r, D)

        self.kernel_derivatives = self.kernel_derivatives + np.einsum(
            'bf,bfrd->frd', gate, winning_windows)
        self.bias_derivatives = self.bias_derivatives + gate.sum(axis=0)

        if not need_input_grad:
            return None

        dinput = np.zeros_like(self.inputs)
        contribution = np.einsum('bf,frd->bfrd', gate, self.kernel)  # (batch, F, r, D)
        for b in range(batch_size):
            for f in range(F):
                start = self.argmax_positions[b, f]
                dinput[b, start:start + r, :] += contribution[b, f]
        return dinput

    def adjust_kernel_values(self, learning_rate, momentum):
        """ SGD + momentum on the accumulated gradients (same rule as
        NeuralLayer.adjust_values), then clears them for the next batch. """
        self.kernel_velocity = momentum * self.kernel_velocity - learning_rate * self.kernel_derivatives
        self.bias_velocity = momentum * self.bias_velocity - learning_rate * self.bias_derivatives
        self.kernel = self.kernel + self.kernel_velocity
        self.bias = self.bias + self.bias_velocity

        self.kernel_derivatives = np.zeros_like(self.kernel)
        self.bias_derivatives = np.zeros_like(self.bias)

    def save(self):
        com.update_kernel(np.asarray(self.kernel, dtype=float).tolist(), self.layerNum,
                          np.asarray(self.bias, dtype=float).tolist())
