import numpy as np
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
        self.averageLoss = None
        self.network_output = None
        self.avdweights = None
        self.avdbiases = None

    def fetch_values(self):
        """ Calls the SQL query in Comms.py and fetches the
        weights and biases, also initializes backpropagation matrix"""
        values = com.fetch_layer(self.layerNum)
        self.weights = values[0]
        self.biases = values[1]
        # Explicitly float: zeros_like would inherit an integer dtype from a
        # database holding whole-number weights and truncate every gradient.
        self.avdweights = np.zeros(np.array(self.weights).T.shape, dtype=float)
        self.avdbiases = np.zeros((1, len(self.biases)), dtype=float)

    def initialise_values(self):
        values = com.fetch_layer(self.layerNum)  # Fetching values for neural layer shape
        self.weights = values[0]
        self.biases = values[1]

        # Weights are stored as (n_out, n_in), so the fan in is the second axis.
        # The previous code read these the other way round; the shape came out right
        # because Glorot is symmetric in the two, but He is not.
        neurons_out = len(self.weights)  # Neurons in this layer
        neurons_in = len(self.weights[0])  # Neurons in the previous layer

        # Weight initialisation
        self.weights = self.he_normal(neurons_in, neurons_out)

        # Bias initialisation. A small positive bias suits ReLU: it starts every unit
        # on the active side of the clamp.
        self.biases = np.full_like(self.biases, 0.01)

        # Database update
        com.update_values(self.layerNum, self.weights.tolist(), self.biases.tolist())

    @staticmethod
    def he_normal(n_in, n_out):
        """ He initialisation, which is the right variance for a ReLU network.

        Glorot assumes an activation that is symmetric about zero, like tanh. ReLU
        discards the negative half, so Glorot under-scales the variance and, across
        five hidden layers, drives the activations toward zero and the units dead. """
        standard_dev = np.sqrt(2 / n_in)
        return np.random.normal(0, standard_dev, (n_out, n_in))

    def batch_layer_output(self, inputs):
        """ This function outputs the result for one batch of one dimensional input vectors """
        self.inputs = inputs
        self.layer_output = np.dot(inputs, np.array(self.weights).T) + self.biases
        self.output = np.maximum(0, self.layer_output)  # Applying the ReLU function 'zeros' out the negative values
        return self.output

    def softmax(self, inputs):
        """ Softmax over the class axis for a batch of input vectors """

        # Computing the layer output
        self.inputs = inputs
        self.layer_output = np.dot(inputs, np.array(self.weights).T) + self.biases
        self.network_output = self.layer_output

        # Subtracting the row maximum keeps the exponentials in range. Without it
        # np.exp overflows to inf on large logits and every output becomes nan.
        shifted = self.layer_output - np.max(self.layer_output, axis=1, keepdims=True)
        exps = np.exp(shifted)

        # Each row is normalised by its own sum, so every row is a distribution.
        self.softmax_output = exps / np.sum(exps, axis=1, keepdims=True)

        return [self.network_output, self.softmax_output]

    def ccel_calculation(self, correct_distribution_matrix):
        """ Categorical cross entropy loss calculation """
        output_losses = []
        for i in range(len(correct_distribution_matrix)):  # For each list in the correct distribution matrix
            for n in range(len(correct_distribution_matrix[i])):  # For each item in the selected list from the matrix
                if correct_distribution_matrix[i][n] == 1:
                    # Searching for the index of the ground truth in the ideal output distribution
                    p = np.clip(self.softmax_output[i][n], 1e-12, 1.0)
                    ccel = float(-np.log(p))  # loss calculation
                    output_losses.append(ccel)
        self.averageLoss = sum(output_losses)/len(output_losses)
        return output_losses

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
        """ Removed: this returned the layer's forward activations, not a derivative. """
        raise NotImplementedError(
            "ReLU_derivative did not compute a derivative: it gated self.output (already "
            "non-negative, so the mask was a no-op) and ignored the upstream gradient "
            "entirely. Use relu_backward(dvalues) instead."
        )

    def relu_backward(self, dvalues):
        """ Gradient of ReLU with respect to its input, given the gradient of its output.

        Gates on layer_output, the pre-activation, because that is what decides which
        units were clamped. Builds a new array so the stored forward pass is untouched. """
        dvalues = np.asarray(dvalues, dtype=float)
        mask = np.asarray(self.layer_output, dtype=float) > 0
        return dvalues * mask

    def calculate_derivatives(self, dvalues):
        """ Computes the gradients of this layer's inputs, weights and biases from the
        gradient of its output, and accumulates the weight and bias gradients.

        Returns the gradient with respect to this layer's inputs, shape (batch, n_in),
        which is what the layer below needs. The batch axis is kept: averaging it away
        here would destroy the per-sample gradients. """

        dvalues = np.asarray(dvalues, dtype=float)
        inputs = np.asarray(self.inputs, dtype=float)
        weights = np.asarray(self.weights, dtype=float)  # Stored as (n_out, n_in)

        dinputs = np.dot(dvalues, weights)                # (batch, n_in)
        dweights = np.dot(inputs.T, dvalues)              # (n_in, n_out)
        dbiases = np.sum(dvalues, axis=0, keepdims=True)  # (1, n_out)

        # Running totals, applied to the parameters by adjust_values
        self.avdweights = self.avdweights + dweights
        self.avdbiases = self.avdbiases + dbiases

        return dinputs

    def adjust_values(self, learning_rate):
        """ Applies the accumulated gradients to the weights and biases.

        No batch divisor here. combined_derivative already divides by the number of
        reviews in the batch, so avdweights holds the gradient of the mean loss and a
        second division would scale the learning rate by 1/batch_size. """
        weights_copy = np.array(self.weights).T  # Copy required as weights is not saved transposed

        # Adjusting values
        new_weights = np.subtract(weights_copy, np.multiply(learning_rate, self.avdweights))
        new_biases = np.subtract(self.biases, np.multiply(learning_rate, self.avdbiases))[0]

        self.weights = new_weights.T
        self.biases = new_biases

        # Clear the running totals so the next batch starts from zero
        self.avdweights = np.zeros_like(self.avdweights)
        self.avdbiases = np.zeros_like(self.avdbiases)

    def save(self):
        """ Writes the weights and biases to the database.

        Kept separate from adjust_values so a run can update in memory every batch
        and only round-trip through SQLite at checkpoints. """
        com.update_values(self.layerNum,
                          np.asarray(self.weights, dtype=float).tolist(),
                          np.asarray(self.biases, dtype=float).tolist())
