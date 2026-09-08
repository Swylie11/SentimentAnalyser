import WordVectorConversions as wvc
import json
from types import SimpleNamespace
import time
from ConvolutionLayer import ConvLayer
from NeuralLayer import NeuralLayer
import Comms as com
import numpy as np


# Defining the convolutional layers
mode = int(input("To train a new model, press 1. To test the currently loaded model, press 2.\n"
                 "To load a model, input valid database files into the current folder titled:\n"
                 "'convolution_layers', and 'neuron_weights'.\n"
                 "WARNING: If training a new model, the current neuron weights and kernel files will be overridden.\n"))

convLayer1 = ConvLayer(1, 5)
convLayer2 = ConvLayer(2, 3)

# Defining the neural layers
neuralLayer1 = NeuralLayer(1)
neuralLayer2 = NeuralLayer(2)
neuralLayer3 = NeuralLayer(3)
neuralLayer4 = NeuralLayer(4)
neuralLayer5 = NeuralLayer(5)
outputLayer = NeuralLayer(6)


if mode == 1:
    # If training a new model: override value matrices with normally distributed values
    # If a new model is not being trained, the currently loaded values will be used.
    convLayer1.initialize_values()
    convLayer2.initialize_values()

    neuralLayer1.initialise_values()
    neuralLayer2.initialise_values()
    neuralLayer3.initialise_values()
    neuralLayer4.initialise_values()
    neuralLayer5.initialise_values()
    outputLayer.initialise_values()


reps = 0
correct_outputs = 0

LEARNING_RATE = 0.01

SENTIMENT_LABELS = {1: "Very negative", 2: "Negative", 3: "Neutral",
                    4: "Positive", 5: "Very positive"}


# Obsolete
def fetch_test_data(test_data, batch_size):
    """ Fetches batch review data from the test data file """
    count = 0
    with open(test_data, 'r', encoding='utf8') as f:
        output = []
        rating = []
        for line in f:  # For every review
            if count < batch_size:
                review = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
                output.append(review.text)  # Adds the review to the batch output
                rating.append(review.rating)
                count += 1
            else:
                break
        f.close()
        total_review = [output, rating]
        return total_review


def flatten_conv_output(inputTensor):
    """ Flattens a (batch, height, width) convolution output to (batch, height*width).

    One review becomes one feature vector. The batch axis stays as the review axis,
    so the dense stack, the loss and the labels all agree on what a row means. """
    tensor = np.asarray(inputTensor, dtype=float)
    return tensor.reshape(tensor.shape[0], -1)


def one_hot_ratings(star_ratings):
    """ Converts a list of 1-5 star ratings to a (batch, 5) one hot matrix. """
    encoded = np.zeros((len(star_ratings), 5))
    encoded[np.arange(len(star_ratings)), np.asarray(star_ratings, dtype=int) - 1] = 1
    return encoded


if mode == 1:  # Training new model
    # Batch input quantity setup from mode selection
    reviews_per_batch = int(input("Enter number of reviews per batch: "))
    num_batches = int(input("Enter number of batches to run: "))
else:
    # Run once (one test)
    num_batches = 1
    reviews_per_batch = 1
    input_sentence = [str(input("Enter data to be tested: "))]

# Start timer
start = time.time()

totalLoss = 0
bigLoss = 0

# One parameter update per batch, so this loop is the number of updates
for i in range(num_batches):

    if mode == 1:
        print(f'Batch: {i+1}')

    currentLoss = 0

    # Fetch the currently loaded values in the databases
    convLayer1.fetchKernel()
    convLayer2.fetchKernel()

    neuralLayer1.fetch_values()
    neuralLayer2.fetch_values()
    neuralLayer3.fetch_values()
    neuralLayer4.fetch_values()
    neuralLayer5.fetch_values()
    outputLayer.fetch_values()

    # Collect a whole batch of reviews before the forward pass, so the batch axis
    # of every tensor from here down is the review axis.
    if mode == 1:  # Training a new model, this fetches test data
        reviews = []
        star_ratings = []
        for r in range(reviews_per_batch):
            review_id = ((i + 1) * reviews_per_batch) - (reviews_per_batch - (r + 1))
            rating, text = com.fetch_test_data(review_id)
            reviews.append(text)
            star_ratings.append(int(rating))

        # Convert star ratings to one hot encoded vectors. This is the correct
        # distribution matrix, shape (reviews in batch, 5).
        ratings = one_hot_ratings(star_ratings)
    else:
        reviews = input_sentence
        star_ratings = None
        ratings = None  # There is no known correct answer

    # Prepare input to conv layer
    conv_input_data = wvc.pad_matrix(wvc.return_vector_matrix_jsonl(wvc.format_entry_data(reviews)))

    # Conv layer operations
    convTime = time.time()  # Convolutional layer operations timer starts
    inputMatrix1 = convLayer1.reflectMatrix(conv_input_data)
    convLayerOutput = convLayer1.convPass(inputMatrix1)
    inputMatrix2 = convLayer2.reflectMatrix(convLayerOutput)
    convLayerOutput2 = convLayer2.convPass(inputMatrix2)

    # Converting the tensor output to one feature vector per review
    neuralLayerInput = flatten_conv_output(convLayerOutput2)

    # Neural layer operations
    neuralTime = time.time()  # Neural layer operations timer starts
    neuralOutput1 = neuralLayer1.batch_layer_output(neuralLayerInput)
    neuralOutput2 = neuralLayer2.batch_layer_output(neuralOutput1)
    neuralOutput3 = neuralLayer3.batch_layer_output(neuralOutput2)
    neuralOutput4 = neuralLayer4.batch_layer_output(neuralOutput3)
    neuralOutput5 = neuralLayer5.batch_layer_output(neuralOutput4)
    neuralNetworkOutput = outputLayer.softmax(neuralOutput5)

    # Final statistical operations
    softmaxOutput = neuralNetworkOutput[1]
    predictions = np.argmax(softmaxOutput, axis=1) + 1  # Back to 1-5 stars

    if mode == 1:  # If training a new model, backpropagate

        # Loss calculation
        outputLayer.ccel_calculation(ratings)

        # Backpropagation function calls

        # Neural layer backpropagation. Each call returns the gradient with respect
        # to that layer's inputs, which is the gradient of the layer below's output,
        # so it has to be carried down the stack rather than discarded.
        grad = outputLayer.calculate_derivatives(outputLayer.combined_derivative(ratings))
        grad = neuralLayer5.calculate_derivatives(neuralLayer5.relu_backward(grad))
        grad = neuralLayer4.calculate_derivatives(neuralLayer4.relu_backward(grad))
        grad = neuralLayer3.calculate_derivatives(neuralLayer3.relu_backward(grad))
        grad = neuralLayer2.calculate_derivatives(neuralLayer2.relu_backward(grad))
        grad = neuralLayer1.calculate_derivatives(neuralLayer1.relu_backward(grad))

        # Reshape the flat per-review gradients back to the conv output tensor.
        conv2_output_shape = np.asarray(convLayer2.output, dtype=float).shape
        neural_input_derivatives = np.asarray(grad).reshape(conv2_output_shape)

        # Convolutional layer backpropagation, passing already-shaped tensors
        first_derivatives = convLayer2.backpropagate(neural_input_derivatives, False)
        second_derivatives = convLayer1.backpropagate(first_derivatives, False)

        totalLoss += outputLayer.averageLoss

        correct_outputs += int(np.sum(predictions == np.asarray(star_ratings)))
        reps += len(reviews)
    else:
        reps += len(reviews)
        for prediction in predictions:
            print(SENTIMENT_LABELS[int(prediction)])

    if mode == 1:  # If training new model
        # Updating values
        outputLayer.adjust_values(LEARNING_RATE)
        neuralLayer5.adjust_values(LEARNING_RATE)
        neuralLayer4.adjust_values(LEARNING_RATE)
        neuralLayer3.adjust_values(LEARNING_RATE)
        neuralLayer2.adjust_values(LEARNING_RATE)
        neuralLayer1.adjust_values(LEARNING_RATE)

        convLayer1.adjust_kernel_values(LEARNING_RATE)
        convLayer2.adjust_kernel_values(LEARNING_RATE)

end = time.time()
print(f"Total time elapsed: {end-start}")

if mode == 1:
    # totalLoss accumulates one batch mean per batch, so it is averaged over batches.
    print(f'Average loss: {totalLoss/num_batches}')
    accuracy = correct_outputs/reps
    print(f'Average accuracy = {accuracy * 100}%')
