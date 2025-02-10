import json
from types import SimpleNamespace
import Comms as com


class ConvLayer:
    def __init__(self, layerNum, stepSize):
        self.layerNum = layerNum
        self.stepSize = stepSize
        self.kernel = None

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
        return output

    def convPass(self, BatchInput):
        """ This functions performs a convolution of a batch of input matrices
        with a single kernel using a specified step size for the layer"""
        totalOutput = []
        for t in range(len(BatchInput)):  # For each matrix in the batch input
            matrix = BatchInput[t]
            padding = len(self.kernel) // 2  # Finds what number the indexes need to be incremented by
            output = []
            for r in range(padding, len(matrix) + padding, self. stepSize):
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
                            product = self.kernel[n][k] * matrix[n+r-padding][k+i-padding]
                            currentTotal += product
                    # Adding all the outputs to the correct matrices ready for output
                    output1.append(currentTotal)
                output.append(output1)
            totalOutput.append(output)
        return totalOutput
