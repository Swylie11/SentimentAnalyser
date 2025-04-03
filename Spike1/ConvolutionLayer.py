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

    # Obsolete
    def fetch_Kernel(self, file):
        changed = False
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # For each line in the file (jsonl format)
                    data = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
                    if data.layerNum == self.layerNum:
                        # if the collected layer number is ours
                        self.kernel = data.kernel
                        changed = True
                except:
                    print("Invalid layer number")

    def fetchKernel(self):
        self.kernel = com.fetch_kernel(self.layerNum)

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
        stepS = self.stepSize
        self.stepSize = 1

        if flattened:
            dvalues = np.array(dvalues).reshape(np.array(self.output).shape)

        # Creating a temporary copy so that the kernel isn't lost.
        kernelTemp = self.kernel
        self.kernel = dvalues.tolist()[0]

        # 180 degree rotation and preparation to become a 'batch'
        rotated_kernel = np.rot90(np.array(kernelTemp), 2).tolist()

        # Filter derivs calculation
        filter_derivatives = self.calculate_kernel_derivatives(self.inputs[0], self.kernel, kernelTemp, stepS)

        # Calculating input derivatives
        spread_dvalues = self.spread_matrix(self.inputs[0], self.kernel, stepS)
        input_derivatives = correlate2d(spread_dvalues.tolist(), rotated_kernel, mode='valid')

        # Adjusting the kernels according to the kernel derivatives
        new_kernel = np.subtract(np.array(kernelTemp), np.multiply(0.01, filter_derivatives)).tolist()

        # Updating the database with the new kernel made in the prev. statement.
        com.update_kernel(new_kernel, self.layerNum)

        # returning the kernel and step size back to normal
        self.kernel = new_kernel
        self.stepSize = stepS

        output = np.array([np.array(input_derivatives).tolist()])

        # Returning the input derivatives for use in the next backpropagation layer
        return output
