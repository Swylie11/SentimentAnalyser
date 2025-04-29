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


def tensor_to_matrix(inputTensor):
    outputMatrix = []
    for i in range(len(inputTensor)):
        newRow = []
        for n in range(len(inputTensor[0])):
            for k in range(len(inputTensor[0][0])):
                newRow.append(inputTensor[i][n][k])
        outputMatrix.append(newRow)
    return outputMatrix


if mode == 1:  # Training new model
    # Batch input quantity setup from mode selection
    batch = int(input("Enter number of batches: "))  # How many batches will be run
    repetitions = int(input("Enter number of tests per batch: "))  # How many batches will be run
else:
    # Run once (one test)
    batch = 1
    repetitions = 1
    input_sentence = [str(input("Enter data to be tested: "))]

# Start timer
start = time.time()

totalLoss = 0
bigLoss = 0

# For each entered batch. This is the number used to determine when backpropagation happens
for i in range(batch):

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

    for r in range(repetitions):

        if mode == 1:  # Training a new model, this fetches test data
            batch_reviews1 = []
            entry_data_total = com.fetch_test_data(((i+1)*repetitions)-(repetitions-(r+1)))
            batch_reviews1.append(entry_data_total)
            batch_reviews = np.array(batch_reviews1).T
            ratings = batch_reviews[0]
            reviews = batch_reviews[1]
            correct_answer = int(ratings[0])

            entry_data_ratings = []
            pre_vector = [0, 0, 0, 0, 0]
            pre_vector[int(ratings[0])-1] = 1

            # Convert star ratings to one hot encoded vectors. This is the correct distribution matrix.
            entry_data_ratings.append(pre_vector)
            ratings = entry_data_ratings  # More conventional and easier to understand naming
        else:
            reviews = input_sentence
            ratings = None
            correct_answer = None  # There is no known correct answer

        # Prepare input to conv layer
        conv_input_data = wvc.pad_matrix(wvc.return_vector_matrix_jsonl(wvc.format_entry_data(reviews)))

        # Conv layer operations
        convTime = time.time()  # Convolutional layer operations timer starts
        inputMatrix1 = convLayer1.reflectMatrix(conv_input_data)
        convLayerOutput = convLayer1.convPass(inputMatrix1)
        inputMatrix2 = convLayer2.reflectMatrix(convLayerOutput)
        convLayerOutput2 = convLayer2.convPass(inputMatrix2)

        # Converting tensor output to matrix output
        neuralLayerInput = tensor_to_matrix(convLayerOutput2)

        # Neural layer operations
        neuralTime = time.time()  # Neural layer operations timer starts
        neuralOutput1 = neuralLayer1.batch_layer_output(neuralLayerInput)
        neuralOutput2 = neuralLayer2.batch_layer_output(neuralOutput1)
        neuralOutput3 = neuralLayer3.batch_layer_output(neuralOutput2)
        neuralOutput4 = neuralLayer4.batch_layer_output(neuralOutput3)
        neuralOutput5 = neuralLayer5.batch_layer_output(neuralOutput4)
        neuralNetworkOutput = outputLayer.softmax(neuralOutput5)
        neuralOutputMetrics = neuralNetworkOutput[0].tolist()

        # Final statistical operations
        softmaxOutput = neuralNetworkOutput[1]

        if mode == 1:  # If training a new model, backpropagate

            # Loss calculation
            outputLayer.ccel_calculation(ratings)

            # Backpropagation function calls

            # Neural layer backpropagation
            outputLayer.calculate_derivatives(outputLayer.combined_derivative(ratings))
            neuralLayer5.calculate_derivatives(neuralLayer5.ReLU_derivative())
            neuralLayer4.calculate_derivatives(neuralLayer4.ReLU_derivative())
            neuralLayer3.calculate_derivatives(neuralLayer3.ReLU_derivative())
            neuralLayer2.calculate_derivatives(neuralLayer2.ReLU_derivative())
            neural_input_derivatives = neuralLayer1.calculate_derivatives(neuralLayer1.ReLU_derivative())[0]

            # Convolutional layer backpropagation
            first_derivatives = convLayer2.backpropagate(neural_input_derivatives, True)
            second_derivatives = convLayer1.backpropagate(first_derivatives, False)

            totalLoss += outputLayer.averageLoss

        # Output metrics
        '''
        print(f'length of conv input. Items: {len(inputMatrix1)} Rows: {len(inputMatrix1[0])} Columns: {len(inputMatrix1[0][0])}')
        print(f'length of neural input. Items: {len(convLayerOutput2)} Rows: {len(convLayerOutput2[0])} Columns: {len(convLayerOutput2[0][0])}')
        print(f'length of neural output. Items: {len(neuralOutputMetrics)} Rows: {len(neuralOutputMetrics[0])} Columns: 1')
        '''
        choice = max(softmaxOutput[0])
        result = softmaxOutput[0].tolist().index(choice)+1
        reps += 1
        if mode == 1 and correct_answer == result:
            correct_outputs += 1
        else:
            if result == 1:
                print("Very negative")
            elif result == 2:
                print("Negative")
            elif result == 3:
                print("Neutral")
            elif result == 4:
                print("Positive")
            elif result == 5:
                print("Very positive")

    if mode == 1:  # If training new model
        # Updating values
        outputLayer.adjust_values(batch)
        neuralLayer5.adjust_values(batch)
        neuralLayer4.adjust_values(batch)
        neuralLayer3.adjust_values(batch)
        neuralLayer2.adjust_values(batch)
        neuralLayer1.adjust_values(batch)

        convLayer1.adjust_kernel_values(batch)
        convLayer2.adjust_kernel_values(batch)

end = time.time()
print(f"Total time elapsed: {end-start}")

if mode == 1:
    print(f'Average loss: {totalLoss/(repetitions*batch)}')
    accuracy = correct_outputs/(repetitions*batch)
    print(f'Average accuracy = {accuracy * 100}%')
