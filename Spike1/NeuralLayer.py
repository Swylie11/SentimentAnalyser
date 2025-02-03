import numpy as np
import json
from types import SimpleNamespace
import Comms as com


class NeuralLayer:
    """ One layer of a neural network """
    def __init__(self, layerNum):
        """This layer takes a matrix of inputs and
        collects the weights and biases externally"""
        self.layerNum = layerNum
        self.inputs = None
        self.softmax_output = None
        self.softmax_copy = None
        self.output = None
        self.layer_output = None
        self.weights = None
        self.biases = None
        self.drelu_di = None

    def fetch_values2(self, file):
        changed = False
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                # For each line in the file (jsonl format)
                neuron = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
                if neuron.layerNum == self.layerNum:
                    # if the collected layer number is ours
                    self.weights = neuron.weights
                    self.biases = neuron.biases
                    changed = True
            if not changed:
                print("Invalid layer number")

    def fetch_values(self):
        """ Calls the SQL query in Comms.py and fetches the weights and biases """
        values = com.fetch_layer(self.layerNum)
        self.weights = values[0]
        self.biases = values[1]

    def layer_output(self, inputs):
        """ This function outputs the result for a one dimensional input vector """
        self.inputs = inputs
        output = []
        for neuron_n in range(len(inputs)):
            neuron_output = float(np.dot(inputs, self.weights[neuron_n]) + self.biases[neuron_n])
            output.append(neuron_output)
        return output

    def batch_layer_output(self, inputs):
        """ This function outputs the result for one batch of one dimensional input vectors """
        self.layer_output = np.dot(inputs, np.array(self.weights).T) + self.biases
        self. output = np.maximum(0, self.layer_output)  # Applying the ReLU function 'zeros' out the negative values
        return self.output

    def softmax(self, input_matrix):
        """ Softmax calculation for batch input data as a 3-dimensional matrix """
        output = []
        input_matrix = np.exp(input_matrix)  # Finding the exponential of each input value

        for i in range(len(input_matrix)):  # For each 2-dimensional list in the input vector
            temp = []
            total = 0
            for n in range(len(input_matrix[i])):  # For each item in the selected list
                total += input_matrix[i, n]  # Collect the sum of all values for the softmax calculation

            for n in range(len(input_matrix[i])):
                temp.append(float(input_matrix[i, n] / total))  # Build a new list of normalised probabilities
            output.append(temp)
        self.softmax_output = output  # Assigning the result to a variable for use in backpropagation
        return output

    @staticmethod
    def ccel_calculation(input_distribution_matrix, correct_distribution_matrix):
        """ Categorical cross entropy loss calculation """
        output_losses = []
        for i in range(len(correct_distribution_matrix)):  # For each list in the correct distribution matrix
            for n in range(len(correct_distribution_matrix[i])):  # For each item in the selected list from the matrix
                if correct_distribution_matrix[i][n] == 1:
                    # Searching for the index of the ground truth in the ideal output distribution
                    ccel = float(-(np.log(input_distribution_matrix[i][n])))  # loss calculation
                    output_losses.append(ccel)
        return output_losses

    @staticmethod
    def ccel_derivative(correct_distribution, softmax_output):
        """ Calculates the normalised derivative of the loss function  """
        batch_size = len(correct_distribution)  # This expects a matrix input as a batch
        dccel = (-correct_distribution)/softmax_output  # Differential calculation
        dccel_normalised = dccel / batch_size  # Normalises the output for the total batch input size
        return dccel_normalised

    def combined_derivative(self, correct_distribution):
        """ Calculates the derivative of the categorical cross entropy loss and the softmax output """
        batch_size = len(self.softmax_output)  # Number of separate inputs

        # convert one hot encoded vectors to the index location of the correct class instead
        correct_distribution = np.argmax(correct_distribution, axis=1)
        self.softmax_copy = self.softmax_output

        # Subtract 1 from the softmax output index that is the same as the correct index
        # This is the derivative
        self.softmax_copy[range(batch_size), correct_distribution] -= 1
        self.softmax_copy = self.softmax_copy / batch_size  # Normalise the results

    def ReLU_derivative(self):
        """ Computes a matrix of derivatives of the relu function where
        self.layer_output is the batch output from a current layer"""

        weights_copy = self.weights.T

        drelu = self.output  # Copying outputs so that they don't get changed
        # Where the output from the network is less than 0, set the relu derivative to 0, otherwise the output
        drelu[self.layer_output <= 0] = 0

        # Calculating derivatives as matrices thus a batch
        dinputs = np.dot(drelu, weights_copy.T)
        dweights = np.dot(self.inputs.T, drelu)
        dbiases = np.sum(drelu, axis=0, keepdims=True)

        print(self.weights)
        self.weights += -0.001 * dweights
        print(self.weights)

        ''' PARAMETER UPDATING NEEDED HERE '''
