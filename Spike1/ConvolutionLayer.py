import json
from types import SimpleNamespace
import Comms as com
import numpy as np
from scipy.signal import correlate2d


class ConvLayer:
    def __init__(self, layerNum, stepSize):
        self.layerNum = layerNum
        self.stepSize = stepSize
        self.output = None
        self.kernel = None
        self.inputs = None
        self.reflected_input = None
        self.filter_derivatives = None
        self.input_derivatives = None

    def fetchKernel(self):
        # load kernel from storage and ensure numpy arrays for internal use
        self.kernel = com.fetch_kernel(self.layerNum)
        self.kernel = np.array(self.kernel, dtype=float)
        self.filter_derivatives = np.zeros_like(self.kernel, dtype=float)
        # input_derivatives will be set after a forward pass when input shape is known
        self.input_derivatives = None

    def initialize_values(self):
        self.fetchKernel()

        # Initializing kernel using glorot method
        self.kernel = self.glorot_normal_kernel(self.kernel).tolist()

        # Updating the database with the new kernel
        com.update_kernel(self.kernel, self.layerNum)

    @staticmethod
    def glorot_normal_kernel(kernel):
        neurons_in = len(kernel)
        neurons_out = len(kernel[0])
        standard_dev = np.sqrt(2 / (neurons_in + neurons_out))  # Standard dev calculation

        # Returning normal dist of correct shape, the ouptut size will be 5x5 or 3x3
        return np.random.normal(0, standard_dev, (neurons_in, neurons_out))

    def reflectMatrix(self, inputBatch):
        """ This functions takes input of a matrix and a kernel and extends
        the input matrix in accordance with the size of the kernel"""
        output = []
        for m in range(len(inputBatch)):
            inputMatrix = inputBatch[m]
            for i in range(len(inputMatrix)):  # For every row in the input matrix
                buffer = len(self.kernel)//2  # How many rows to be added
                newArr = inputMatrix[i][:buffer]  # First half of the matrix
                for n in range(len(newArr)):
                    inputMatrix[i].insert(0, newArr[n])  # Adds the items in reverse order to the matrix
                newArr = inputMatrix[i][(len(inputMatrix[i])-buffer):]  # Second half of the matrix
                newArr.reverse()
                for n in range(len(newArr)):
                    inputMatrix[i].insert(len(inputMatrix[i]), newArr[n])
            buffer = len(self.kernel)//2
            newArr = inputMatrix[:buffer]  # First len(self.kernel)-1 rows in the input matrix
            for i in range(len(newArr)):
                inputMatrix.insert(0, newArr[i])  # Adding the new matrix to the start of the input matrix
            newArr = inputMatrix[(len(inputMatrix)-buffer):]
            newArr.reverse()
            for i in range(len(newArr)):
                inputMatrix.insert(len(inputMatrix), newArr[i])  # Adding the new matrix to the end of the input matrix
            output.append(inputMatrix)  # Adds the reflected input matrix to the final output
        self.reflected_input = output
        return output

    def convPass(self, BatchInput):
        """ This functions performs a convolution of a batch of input matrices
        with a single kernel using a specified step size for the layer"""
        totalOutput = []
        self.inputs = BatchInput
        for t in range(len(BatchInput)):  # For each matrix in the batch input
            matrix = BatchInput[t]
            padding = len(self.kernel) // 2  # Finds what number the indexes need to be incremented by
            output = []
            for r in range(padding, len(matrix) + padding, self.stepSize):
                # Starting at the padded value, up to the length of the matrix
                output1 = []
                currentTotal = 0
                if r > (len(matrix) - padding - 1) and (r - (len(matrix) - padding - 1) != r):
                    #  If r is indexed further than the padding allows and this multiplication hasn't already happened
                    #  Set r to the furthest possible value
                    r = len(matrix) - padding - 1
                elif r > (len(matrix) - padding - 1):
                    break
                for i in range(padding, len(matrix[0]) + padding, self.stepSize):
                    currentTotal = 0
                    # For as many sets of column multiplications are to be completed
                    if i > (len(matrix[0]) - padding - 1) and (i - (len(matrix[0]) - padding - 1) != i):
                        #  If i is indexed further than the padding and this multiplication hasn't already happened
                        i = len(matrix[0]) - padding - 1  # set i to the furthest possible value
                    elif i > (len(matrix[0]) - padding - 1):
                        # if i is greater than the furthest possible value stop the algorithm
                        break
                    for n in range(len(self.kernel)):  # For each list in the self.kernel
                        for k in range(len(self.kernel[n])):  # For each item (k) in the nth list of the self.kernel
                            if 0 <= n+r-padding < len(matrix) and 0 <= k+i-padding < len(matrix[0]):
                                product = self.kernel[n][k] * matrix[n+r-padding][k+i-padding]
                                currentTotal += product
                    # Adding all the outputs to the correct matrices ready for output
                    output1.append(currentTotal)
                output.append(output1)
            totalOutput.append(output)
            self.output = totalOutput
        return totalOutput

    @staticmethod
    def calculate_kernel_derivatives(inputs, dvalues, original_kernel, step_size):
        """ This method uses a convolutional layers inputs, output derivatives, kernel
        and step size to calculate the derivatives of the kernel values"""
        filter_derivatives = np.zeros_like(original_kernel).tolist()  # Making an initial matrix to perform operations
        for i in range(0, len(inputs), step_size):
            # Starting at 0, up until the number of rows, incrementing by step_size each time
            for j in range(0, len(inputs[0]), step_size):
                # Starting at 0, up until the number of columns, incrementing by step_size each time
                if i+step_size < len(inputs) and j+step_size < len(inputs[0]):  # Checking indexes haven't gone too high
                    # Collecting the current patch and putting those values into a matrix
                    inputs = np.array(inputs)
                    patch = np.array(inputs[i:i+step_size, j:j+step_size]).tolist()
                    # Adding to filter derivatives the  current matrix multiplied with its respective dvalue
                    filter_derivatives = np.add(np.array(filter_derivatives), np.multiply(dvalues[i//step_size][j//step_size], patch))
        return filter_derivatives

    @staticmethod
    def spread_matrix(inputs, dvalues, stride):
        """ Spreads out the values of an input matrix based on a given stride length """
        new_values = np.zeros_like(inputs)
        dvalues = np.array(dvalues)
        for i in range(0, len(dvalues), stride):
            for j in range(0, len(dvalues[0]), stride):
                if i+stride-1 <= len(new_values) and j+stride-1 <= len(new_values[0]):
                    new_values[i+(stride-1), j+(stride-1)] = dvalues[i//stride][j//stride]
        return new_values

    def backpropagate(self, dvalues, flattened):
        import numpy as np
        from scipy.signal import convolve2d

        stepS = self.stepSize

        # reshape dvalues if flattened
        if flattened:
            dvalues = np.array(dvalues).reshape(np.array(self.output).shape)

        dvalues = np.array(dvalues, dtype=float)        # expected shape: (batch, out_h, out_w)
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

            # gradient wrt input: full convolution of upsampled dout with rotated kernel
            grad_in_full = convolve2d(dout_up, np.rot90(kernel, 2), mode='full')

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

        # accumulate filter derivatives and store input derivatives (sum across batch)
        self.filter_derivatives = self.filter_derivatives + filter_grad
        summed_input_derivs = np.sum(np.stack(per_input_grads), axis=0)
        if self.input_derivatives is None:
            self.input_derivatives = summed_input_derivs
        else:
            self.input_derivatives = np.array(self.input_derivatives, dtype=float) + summed_input_derivs

        # return per-sample input gradients as (batch, H, W)
        return np.stack(per_input_grads)
    

    def adjust_kernel_values(self, batch_size, learning_rate=0.01):
        import numpy as np
        if batch_size <= 0:
            return
        kernel_arr = np.array(self.kernel, dtype=float)
        filt_deriv = np.array(self.filter_derivatives, dtype=float)
        update = (learning_rate * (filt_deriv / batch_size))
        kernel_arr = kernel_arr - update
        self.kernel = kernel_arr.tolist()
        com.update_kernel(self.kernel, self.layerNum)
        # reset accumulated derivatives after the update
        self.filter_derivatives = np.zeros_like(kernel_arr, dtype=float)
