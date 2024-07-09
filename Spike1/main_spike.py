import numpy as np

all_inputs = [[1.1, 2, 3, 4],
              [1.2, 2.3, 0.3, 1.9],
              [0.2, 3.3, 0.7, 1.6]]

correct_output = [0, 0, 1]


class Layer:
    """ One layer of a neural network """
    def __init__(self, inputs):
        """This layer takes a matrix of inputs but has pre-defined -
        weights and biases for each neuron (each row is one neuron)"""
        self.inputs = inputs
        self.weights = [[0.5, 1.2, -0.9, 0.2],
                        [1.8, -0.2, 0.5, -1.3],
                        [-1.3, 0.2, 0.8, -0.3]]
        self.biases = [2, 3, 0.5]

    def layer_output(self):
        """ This function outputs the result for a one dimensional input vector """
        output = []
        for neuron_n in range(3):
            neuron_output = float(np.dot(self.inputs, self.weights[neuron_n]) + self.biases[neuron_n])
            output.append(neuron_output)
        return output

    def batch_layer_output(self):
        """ This function outputs the result for one batch of one dimensional input vectors """
        layer_output = np.dot(self.inputs, np.array(self.weights).T) + self.biases
        output = np.maximum(0, layer_output)  # Applying the ReLU function 'zeros' out the negative values
        return output


def softmax(input_matrix):
    output = []
    total = 0
    input_matrix = np.exp(input_matrix)

    for i in range(len(input_matrix)):
        temp = []
        total = 0
        for n in range(len(input_matrix[i])):
            total += input_matrix[i, n]

        for n in range(len(input_matrix[i])):
            temp.append(float(input_matrix[i, n] / total))
        output.append(temp)
    return output


Neurons_l1 = Layer(all_inputs)
print(Neurons_l1.batch_layer_output())
print(softmax(Neurons_l1.batch_layer_output()))
