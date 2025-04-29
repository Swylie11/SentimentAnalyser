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
        self.averageLoss = None
        self.network_output = None
        self.avdweights = None
        self.avdbiases = None

    # Obsolete
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
        """ Calls the SQL query in Comms.py and fetches the
        weights and biases, also initializes backpropagation matrix"""
        values = com.fetch_layer(self.layerNum)
        self.weights = values[0]
        self.biases = values[1]
        self.avdweights = np.zeros_like(np.array(self.weights).T)
        self.avdbiases = np.zeros_like(self.biases)

    def initialise_values(self):
        values = com.fetch_layer(self.layerNum)  # Fetching values for neural layer shape
        self.weights = values[0]
        self.biases = values[1]
        neurons_in = len(self.weights)  # Neurons in prev layer
        neurons_out = len(self.weights[0])  # Neurons in current layer

        # Weight initialisation
        self.weights = self.glorot_normal(neurons_in, neurons_out)

        # Bias initialisation
        self.biases = np.full_like(self.biases, 0.01)

        # Database update
        com.update_values(self.layerNum, self.weights.tolist(), self.biases.tolist())

    @staticmethod
    def glorot_normal(n_in, n_out):
        standard_dev = np.sqrt(2 / (n_in + n_out))
        return np.random.normal(0, standard_dev, (n_in, n_out))

    def layer_output1(self, inputs):
        """ This function outputs the result for a one dimensional input vector """
        self.inputs = inputs
        output = []
        for neuron_n in range(len(inputs)):
            neuron_output = float(np.dot(inputs, self.weights[neuron_n]) + self.biases[neuron_n])
            output.append(neuron_output)
        return output

    def batch_layer_output(self, inputs):
        """ This function outputs the result for one batch of one dimensional input vectors """
        self.inputs = inputs
        self.layer_output = np.dot(inputs, np.array(self.weights).T) + self.biases
        self.output = np.maximum(0, self.layer_output)  # Applying the ReLU function 'zeros' out the negative values
        return self.output

    def softmax(self, inputs):
        """ Softmax calculation for batch input data as a 3-dimensional matrix """

        # Computing the layer output
        self.inputs = inputs
        self.layer_output = np.dot(inputs, np.array(self.weights).T) + self.biases
        self.network_output = self.layer_output

        # Softmax calculation
        input_matrix = np.exp(self.layer_output)  # Finding the exponential of each input value

        self.softmax_output = input_matrix / (sum(input_matrix[0]))

        '''
        for i in range(len(input_matrix)):  # For each 2-dimensional list in the input vector
            temp = []
            total = 0
            for n in range(len(input_matrix[i])):  # For each item in the selected list
                total += input_matrix[i, n]  # Collect the sum of all values for the softmax calculation

            for n in range(len(input_matrix[i])):
                temp.append(float(input_matrix[i, n] / total))  # Build a new list of normalised probabilities
            output2.append(temp)
        self.softmax_output = output2  # Assigning the result to a variable for use in backpropagation
        '''

        return [self.network_output, self.softmax_output]

    def ccel_calculation(self, correct_distribution_matrix):
        """ Categorical cross entropy loss calculation """
        output_losses = []
        for i in range(len(correct_distribution_matrix)):  # For each list in the correct distribution matrix
            for n in range(len(correct_distribution_matrix[i])):  # For each item in the selected list from the matrix
                if correct_distribution_matrix[i][n] == 1:
                    # Searching for the index of the ground truth in the ideal output distribution
                    ccel = float(-(np.log(self.softmax_output[i][n])))  # loss calculation
                    output_losses.append(ccel)
        self.averageLoss = sum(output_losses)/len(output_losses)
        return output_losses

    # Obsolete
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
        self.softmax_copy = np.array(self.softmax_output)

        # Subtract 1 from the softmax output index that is the same as the correct index
        # This is the derivative
        self.softmax_copy[range(batch_size), correct_distribution] -= 1
        self.softmax_copy = self.softmax_copy / batch_size  # Normalise the results
        return self.softmax_copy

    def ReLU_derivative(self):

        batch_drelu = []

        for i in range(len(self.inputs)):

            drelu = self.output[i]

            drelu[self.output[i] <= 0] = 0
            batch_drelu.append(drelu)

        return batch_drelu

    def calculate_derivatives(self, dvalues):
        """ Computes a matrix of derivatives of the relu function, inputs, weights and biases then
        updates the weights and biases for a given layer in an external database"""

        # Creating data structures
        weights_copy = np.array(self.weights).T
        batch_dinputs = []
        batch_dweights = []
        batch_dbiases = []

        # For each input (computing once per sentence)
        for i in range(len(self.inputs)):

            dvalue = dvalues[i]
            dvalue = np.array(dvalue).reshape(1, -1)
            inputs2 = self.inputs[i]
            inputs = np.array(inputs2).reshape(1, -1)

            # Calculating derivatives as matrices thus a batch
            dinputs = np.dot(dvalue, weights_copy.T)
            dweights = np.dot(inputs.T, dvalue)
            dbiases = np.sum(dvalue, axis=0, keepdims=True)

            # Updating batches of differentials
            batch_dinputs.append(dinputs)
            batch_dweights.append(dweights)
            batch_dbiases.append(dbiases)

        # Calculating averages
        self.avdweights = np.add(self.avdweights, np.array(sum(batch_dweights)/len(batch_dweights)))
        self.avdbiases = np.add(self.avdbiases, np.array(sum(batch_dbiases)/len(batch_dbiases)))
        avdinputs = sum(batch_dinputs)/len(batch_dinputs)

        return avdinputs

    def adjust_values(self, batch_size):
        """ Adjusts the weights and biases in the network based on current running derivatives """
        weights_copy = np.array(self.weights).T  # Copy required as weights is not saved transposed

        # Adjusting values
        new_weights = np.subtract(weights_copy, np.multiply(0.01, np.divide(self.avdweights, batch_size)))
        new_biases = np.subtract(self.biases, np.multiply(0.01, np.divide(self.avdbiases, batch_size)))[0]

        # Updating values
        com.update_values(self.layerNum, new_weights.T.tolist(), new_biases.tolist())

