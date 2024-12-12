import numpy as np
import json
from types import SimpleNamespace


class NeuralLayer:
    """ One layer of a neural network """
    def __init__(self, layerNum):
        """This layer takes a matrix of inputs and collects the
        weights and biases externally (each row is one neuron)"""
        self.layerNum = layerNum
        self.weights = None
        self.biases = None

    def fetch_values(self, file):
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

    def layer_output(self, inputs):
        """ This function outputs the result for a one dimensional input vector """
        output = []
        for neuron_n in range(len(inputs)):
            neuron_output = float(np.dot(inputs, self.weights[neuron_n]) + self.biases[neuron_n])
            output.append(neuron_output)
        return output

    def batch_layer_output(self, inputs):
        """ This function outputs the result for one batch of one dimensional input vectors """
        layer_output = np.dot(inputs, np.array(self.weights).T) + self.biases
        output = np.maximum(0, layer_output)  # Applying the ReLU function 'zeros' out the negative values
        return output

    @staticmethod
    def softmax(input_matrix):
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

# print(Neurons_l1.softmax(Neurons_l1.batch_layer_output()))
# print(Neurons_l1.ccel_calculation(Neurons_l1.softmax(Neurons_l1.batch_layer_output()), [[1, 0, 0], [1, 0, 0], [1, 0, 0]]))
# In future the correct distribution matrix will be the collected from the star rating of the amazon review.
