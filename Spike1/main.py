import WordVectorConversions as wvc
import json
from types import SimpleNamespace
import time
from ConvolutionLayer import ConvLayer
from NeuralLayer import NeuralLayer


# Defining the convolutional layers
kernel_file = "C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/Kernel_Values.jsonl"

convLayer1 = ConvLayer(1, 5)
convLayer2 = ConvLayer(2, 3)
convLayer3 = ConvLayer(3, 3)
convLayer1.fetchKernel()
convLayer2.fetchKernel()
convLayer3.fetchKernel()

# Defining the neural layers
neuralLayer1 = NeuralLayer(1)
neuralLayer2 = NeuralLayer(2)
neuralLayer3 = NeuralLayer(3)
neuralLayer4 = NeuralLayer(4)
neuralLayer5 = NeuralLayer(5)
outputLayer = NeuralLayer(6)

neuralLayer1.fetch_values()
neuralLayer2.fetch_values()
neuralLayer3.fetch_values()
neuralLayer4.fetch_values()
neuralLayer5.fetch_values()
outputLayer.fetch_values()


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


# Fetching batch input
batch = int(input("Enter batch size: "))  # Lets the user control the batch size

# Start timer
start = time.time()

test_data_file = 'C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/TestData.jsonl'
entry_data_total = fetch_test_data(test_data_file, batch)
entry_data = entry_data_total[0]  # The text within a review
entry_data_review_indexes = entry_data_total[1]  # The star rating from a review

entry_data_review = []
for i in range(batch):  # For the batch size
    pre_vector = [0, 0, 0, 0, 0]
    pre_vector[int(entry_data_review_indexes[i])-1] = 1

    # Convert star ratings to one hot encoded vectors. This is the correct distribution matrix.
    entry_data_review.append(pre_vector)

# Prepare input to conv layer
conv_input_data = wvc.pad_matrix(wvc.return_vector_matrix_jsonl(wvc.format_entry_data(entry_data)))

# Conv layer operations
convTime = time.time()  # Convolutional layer operations timer starts
inputMatrix1 = convLayer1.reflectMatrix(conv_input_data)
convLayerOutput = convLayer1.convPass(inputMatrix1)
convLayerOutput2 = convLayer2.convPass(convLayerOutput)

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
print(f'Neural network output: {neuralNetworkOutput[0]}')
softmaxOutput = neuralNetworkOutput[1]

# Data output
print(f'Softmax output: {softmaxOutput}')
outputLayer.ccel_calculation(entry_data_review)

outputLayer.calculate_derivatives(outputLayer.combined_derivative(entry_data_review))
neuralLayer5.calculate_derivatives(neuralLayer5.ReLU_derivative())
neuralLayer4.calculate_derivatives(neuralLayer4.ReLU_derivative())
neuralLayer3.calculate_derivatives(neuralLayer3.ReLU_derivative())
neuralLayer2.calculate_derivatives(neuralLayer2.ReLU_derivative())
neuralLayer1.calculate_derivatives(neuralLayer1.ReLU_derivative())

# Output metrics
print(f'length of conv input. Items: {len(inputMatrix1)} Rows: {len(inputMatrix1[0])} Columns: {len(inputMatrix1[0][0])}')
print(f'length of neural input. Items: {len(convLayerOutput2)} Rows: {len(convLayerOutput2[0])} Columns: {len(convLayerOutput2[0][0])}')
print(f'length of neural output. Items: {len(neuralOutputMetrics)} Rows: {len(neuralOutputMetrics[0])} Columns: 1')
end = time.time()
print(f"Time elapsed on data preparation: {convTime-start}")
print(f"Time elapsed on convolutions: {neuralTime-convTime}")
print(f"Time elapsed on neural layers: {end-neuralTime}")
print(f"Loss: {outputLayer.averageLoss}, correct matrix: {entry_data_review}")
print(f"Total time elapsed: {end-start}")
