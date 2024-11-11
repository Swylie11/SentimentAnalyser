import json
from types import SimpleNamespace


class ConvLayer:
    def __init__(self, layerNum, stepSize):
        self.layerNum = layerNum
        self.stepSize = stepSize

    def fetchKernel(self, file):
        with open(file) as f:
            for line in f:
                # For each line in the file
                data = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
                if data.layerNum == self.layerNum:
                    # If the selected layer number is the same as entered for this class
                    kernel = data.values
                    return kernel
                else:  # If the entered layer number does not exist in the file
                    print('Layer number does not exist')
                    return None

    def reflectMatrix(self, inputMatrix, kernel):
        """ This functions takes input of a matrix and a kernel and extends
        the input matrix in accordance with the size of the kernel"""
        for i in range(len(inputMatrix)):  # For every row in the input matrix
            buffer = len(kernel)//2  # How many rows to be added
            newArr = inputMatrix[i][:buffer]  # First half of the matrix
            for n in range(len(newArr)):
                inputMatrix[i].insert(0, newArr[n])  # Adds the items in the new array in reverse order to the matrix
            newArr = inputMatrix[i][(len(inputMatrix[i])-buffer):]  # Second half of the matrix
            newArr.reverse()
            for n in range(len(newArr)):
                inputMatrix[i].insert(len(inputMatrix[i]), newArr[n])
        buffer = len(kernel)//2
        newArr = inputMatrix[:buffer]  # First len(kernel)-1 rows in the input matrix
        for i in range(len(newArr)):
            inputMatrix.insert(0, newArr[i])  # Adding the new matrix to the start of the input matrix
        newArr = inputMatrix[(len(inputMatrix)-buffer):]
        newArr.reverse()
        for i in range(len(newArr)):
            inputMatrix.insert(len(inputMatrix), newArr[i])  # Adding the new matrix to the end of the input matrix
        return inputMatrix  # Returns the reflected input matrix

    def convPass(self, inputMatrix, kernel):
        """ This functions performs a convolution of the input matrix with the kernel """
        pass
