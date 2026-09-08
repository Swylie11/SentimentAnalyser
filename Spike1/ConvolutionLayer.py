import Comms as com
import numpy as np
from scipy.signal import convolve2d, correlate2d


class ConvLayer:
    def __init__(self, layerNum, stepSize):
        self.layerNum = layerNum
        self.stepSize = stepSize
        self.output = None
        self.kernel = None
        self.inputs = None
        self.reflected_input = None
        self.filter_derivatives = None
        self.pre_activation = None  # Convolution output before the ReLU

    def fetchKernel(self):
        # load kernel from storage and ensure numpy arrays for internal use
        self.kernel = com.fetch_kernel(self.layerNum)
        self.kernel = np.array(self.kernel, dtype=float)
        self.filter_derivatives = np.zeros_like(self.kernel, dtype=float)

    def initialize_values(self):
        self.fetchKernel()

        # Initializing kernel using the He method
        self.kernel = self.he_normal_kernel(self.kernel).tolist()

        # Updating the database with the new kernel
        com.update_kernel(self.kernel, self.layerNum)

    @staticmethod
    def he_normal_kernel(kernel):
        """ He initialisation for the kernel, matching the ReLU now applied in convPass.

        The fan in of a convolution unit is every cell of the window it sees, so it is
        the number of kernel elements rather than either of its two sides. """
        kernel_height = len(kernel)
        kernel_width = len(kernel[0])
        standard_dev = np.sqrt(2 / (kernel_height * kernel_width))

        # Returning normal dist of correct shape, the output size will be 5x5 or 3x3
        return np.random.normal(0, standard_dev, (kernel_height, kernel_width))

    def reflectMatrix(self, inputBatch):
        """ Pads every matrix in the batch by half the kernel width on each side,
        mirroring the border values.

        This builds and returns a new array. The previous version inserted into the
        caller's lists in place, so calling it on convLayer1.output silently rewrote
        layer 1's stored forward output with a padded copy after the fact, which in
        turn hid a shape mismatch in the backward pass. """
        buffer = len(self.kernel) // 2
        padded = np.pad(np.asarray(inputBatch, dtype=float),
                        ((0, 0), (buffer, buffer), (buffer, buffer)),
                        mode="symmetric")
        self.reflected_input = padded
        return padded

    @staticmethod
    def reflection_source_indices(padded_length, buffer):
        """ Maps each index of a symmetrically padded axis back to the index it copied. """
        original_length = padded_length - 2 * buffer
        sources = np.empty(padded_length, dtype=int)
        for position in range(padded_length):
            if position < buffer:
                sources[position] = buffer - 1 - position
            elif position < buffer + original_length:
                sources[position] = position - buffer
            else:
                sources[position] = original_length - 1 - (position - buffer - original_length)
        return sources

    def fold_reflection_gradient(self, gradient):
        """ Converts a gradient with respect to the padded input into one with respect
        to the unpadded input.

        The padding duplicates border values, so a padded cell's gradient belongs to
        whichever original cell it was mirrored from and has to be added back there
        rather than discarded. """
        gradient = np.asarray(gradient, dtype=float)
        buffer = len(self.kernel) // 2
        if buffer == 0:
            return gradient

        batch_size, padded_height, padded_width = gradient.shape
        row_sources = self.reflection_source_indices(padded_height, buffer)
        column_sources = self.reflection_source_indices(padded_width, buffer)

        folded_rows = np.zeros((batch_size, padded_height - 2 * buffer, padded_width))
        for position in range(padded_height):
            folded_rows[:, row_sources[position], :] += gradient[:, position, :]

        folded = np.zeros((batch_size, padded_height - 2 * buffer, padded_width - 2 * buffer))
        for position in range(padded_width):
            folded[:, :, column_sources[position]] += folded_rows[:, :, position]

        return folded

    def convPass(self, BatchInput):
        """ Cross-correlates every matrix in the batch with the layer kernel, taking
        every stepSize-th position, then applies ReLU.

        The previous hand rolled loop emitted duplicated rows and columns at the
        bottom and right edges: its bounds guard could never take the break branch,
        so once the window ran past the end it clamped to the last valid position and
        re-computed it for every remaining step. An 8x9 input with a 3x3 kernel and
        stride 1 came out as 8x9 (the padded size) instead of 6x9, with the last three
        rows identical. The backward pass then assumed output row i came from input
        row i*stepSize, which those duplicates violate.

        This is the same valid-correlate-then-stride formulation benchmark.py uses. """
        self.inputs = BatchInput

        batch = np.asarray(BatchInput, dtype=float)
        kernel = np.asarray(self.kernel, dtype=float)

        outputs = []
        for t in range(len(batch)):  # For each matrix in the batch input
            correlated = correlate2d(batch[t], kernel, mode='valid')
            outputs.append(correlated[::self.stepSize, ::self.stepSize])

        # Without an activation here the two convolution layers compose to a single
        # linear map and the second one adds nothing. The pre-activation is kept
        # because the backward pass needs it to know which units were clamped.
        self.pre_activation = np.stack(outputs)
        self.output = np.maximum(0, self.pre_activation)
        return self.output

    def backpropagate(self, dvalues, flattened):
        stepS = self.stepSize

        # reshape dvalues if flattened
        if flattened:
            dvalues = np.array(dvalues).reshape(np.array(self.output).shape)

        dvalues = np.array(dvalues, dtype=float)        # expected shape: (batch, out_h, out_w)

        # Gradient of the ReLU applied in convPass. Units whose pre-activation was
        # negative were clamped and pass no gradient.
        if self.pre_activation is not None:
            dvalues = dvalues * (np.asarray(self.pre_activation, dtype=float) > 0)

        kernel = np.array(self.kernel, dtype=float)     # (k_h, k_w)

        # ensure filter_derivatives exists
        if self.filter_derivatives is None:
            self.filter_derivatives = np.zeros_like(kernel, dtype=float)
        else:
            self.filter_derivatives = np.array(self.filter_derivatives, dtype=float)

        batch_size = len(self.inputs)
        k_h, k_w = kernel.shape

        filter_grad = np.zeros_like(kernel, dtype=float)
        per_input_grads = []

        for b in range(batch_size):
            inp = np.array(self.inputs[b], dtype=float)   # original input for sample b
            dout = dvalues[b]                             # (out_h, out_w)
            out_h, out_w = dout.shape

            # accumulate filter gradients
            for i in range(out_h):
                for j in range(out_w):
                    i_in = i * stepS
                    j_in = j * stepS
                    patch = inp[i_in:i_in + k_h, j_in:j_in + k_w]
                    if patch.shape == (k_h, k_w):
                        filter_grad += dout[i, j] * patch
                    else:
                        pad_patch = np.zeros_like(kernel, dtype=float)
                        h, w = patch.shape
                        pad_patch[:h, :w] = patch
                        filter_grad += dout[i, j] * pad_patch

            # upsample dout by stride (zeros inserted between positions)
            up_h = out_h * stepS - (stepS - 1)
            up_w = out_w * stepS - (stepS - 1)
            dout_up = np.zeros((up_h, up_w), dtype=float)
            for i in range(out_h):
                for j in range(out_w):
                    dout_up[i * stepS, j * stepS] = dout[i, j]

            # Gradient wrt input. The forward pass is a cross-correlation, so this is a
            # full convolution with the kernel as it stands: convolve2d already flips
            # its second argument. Rotating the kernel as well flipped it back and
            # produced gradients with the wrong orientation, including sign errors at
            # interior positions.
            grad_in_full = convolve2d(dout_up, kernel, mode='full')

            # crop/trim to original input size
            inp_h, inp_w = inp.shape
            grad_cropped = grad_in_full[:inp_h, :inp_w]
            if grad_cropped.shape != (inp_h, inp_w):
                tmp = np.zeros((inp_h, inp_w), dtype=float)
                h = min(grad_cropped.shape[0], inp_h)
                w = min(grad_cropped.shape[1], inp_w)
                tmp[:h, :w] = grad_cropped[:h, :w]
                grad_cropped = tmp

            per_input_grads.append(grad_cropped)

        # accumulate filter derivatives
        self.filter_derivatives = self.filter_derivatives + filter_grad

        # Return the gradient with respect to the unpadded input, which is what the
        # layer below actually produced.
        return self.fold_reflection_gradient(np.stack(per_input_grads))

    def adjust_kernel_values(self, learning_rate):
        """ Applies the accumulated kernel gradients.

        As in NeuralLayer.adjust_values there is no batch divisor: the gradient
        arriving from the dense stack has already been normalised by batch size. """
        kernel_arr = np.array(self.kernel, dtype=float)
        filt_deriv = np.array(self.filter_derivatives, dtype=float)

        kernel_arr = kernel_arr - (learning_rate * filt_deriv)

        self.kernel = kernel_arr

        # reset accumulated derivatives after the update
        self.filter_derivatives = np.zeros_like(kernel_arr, dtype=float)

    def save(self):
        """ Writes the kernel to the database.

        Kept separate from adjust_kernel_values so a run can update in memory every
        batch and only round-trip through SQLite at checkpoints. """
        com.update_kernel(np.asarray(self.kernel, dtype=float).tolist(), self.layerNum)
