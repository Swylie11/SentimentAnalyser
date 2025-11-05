import numpy as np

""" Computes a matrix of derivatives of the relu function where
self.layer_output is the batch output from a current layer"""

inputs = np.array([[[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]],
                  [[2, 1, 1.5, 2.5],
                  [2.5, 0.5, -1.5, 2],
                  [-1, 3.2, 2.5, -0.2]]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([[2, 3, 0.5]])

weights_copy = weights.copy()
biases_copy = biases.copy()

# batch data structure set up
batch_layer_outputs = []
batch_relu_outputs = []
batch_drelu = []
batch_dinputs = []
batch_dweights = []
batch_dbiases = []

for i in range(len(inputs)):  # for each batch

    # neuron layer calculations
    layer_outputs = np.dot(inputs[i], weights) + biases
    batch_layer_outputs.append(layer_outputs)
    relu_outputs = np.maximum(0, layer_outputs)  # ReLU activation output
    batch_relu_outputs.append(relu_outputs)

    drelu = layer_outputs  # Copying outputs so that they don't get changed

    drelu[layer_outputs <= 0] = 0  # Applying the relu derivative

    batch_drelu.append(drelu)

    # Finding derivatives of the weights, inputs and biases
    dinputs = np.dot(drelu, weights_copy.T)
    dweights = np.dot(inputs[i].T, drelu)
    dbiases = np.sum(drelu, axis=0, keepdims=True)
    batch_dinputs.append(dinputs)
    batch_dweights.append(dweights)
    batch_dbiases.append(dbiases)

avdweights = sum(batch_dweights)/len(batch_dweights)
avdbiases = sum(batch_dbiases)/len(batch_dbiases)
avdinputs = sum(batch_dinputs)/len(batch_dinputs)

print(avdweights)
print(avdbiases)
print(avdinputs)

# Now the derivatives for the weights and biases are used to edit their values and dinputs is passed to the next layer

''' PARAMETER UPDATING HERE '''
