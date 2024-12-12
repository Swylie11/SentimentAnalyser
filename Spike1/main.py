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
convLayer1.fetchKernel(kernel_file)
convLayer2.fetchKernel(kernel_file)
convLayer3.fetchKernel(kernel_file)

# Defining the neural layers
neurons_file = "C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/Neural_Values.jsonl"

neuralLayer1 = NeuralLayer(1)
neuralLayer2 = NeuralLayer(2)
neuralLayer3 = NeuralLayer(3)
neuralLayer4 = NeuralLayer(4)
neuralLayer5 = NeuralLayer(5)
outputLayer = NeuralLayer(6)

neuralLayer1.fetch_values(neurons_file)
neuralLayer2.fetch_values(neurons_file)
neuralLayer3.fetch_values(neurons_file)
neuralLayer4.fetch_values(neurons_file)
neuralLayer5.fetch_values(neurons_file)
outputLayer.fetch_values(neurons_file)


def fetch_test_data(test_data, batch_size):
    """ Fetches batch review data from the test data file """
    count = 0
    with open(test_data, 'r', encoding='utf8') as f:
        output = []
        for line in f:  # For every review
            if count < batch_size:
                review = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
                output.append(review.text)  # Adds the review to the batch output
                count += 1
            else:
                break
        f.close()
        return output


def tensor_to_matrix(inputTensor):
    outputMatrix = []
    for i in range(len(inputTensor)):
        newRow = []
        for n in range(len(inputTensor[0])):
            for k in range(len(inputTensor[0][0])):
                newRow.append(inputTensor[i][n][k])
        outputMatrix.append(newRow)
    return outputMatrix


# Start timer
start = time.time()

# Fetching batch input
batch = int(input("Enter batch size: "))  # Lets the user control the batch size
test_data_file = 'C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/TestData.jsonl'
entry_data = fetch_test_data(test_data_file, batch)

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
neuralNetworkOutput = outputLayer.batch_layer_output(neuralOutput5)
neuralOutputMetrics = neuralNetworkOutput.tolist()

# Final statistical operations
print(neuralNetworkOutput)
softmaxOutput = outputLayer.softmax(neuralNetworkOutput)
# ccelOutput = outputLayer.ccel_calculation(softmaxOutput, 'Correct distribution matrix')

# Data output
print(softmaxOutput)

# Output metrics
print(f'length of conv input. Items: {len(inputMatrix1)} Rows: {len(inputMatrix1[0])} Columns: {len(inputMatrix1[0][0])}')
print(f'length of neural input. Items: {len(convLayerOutput2)} Rows: {len(convLayerOutput2[0])} Columns: {len(convLayerOutput2[0][0])}')
print(f'length of neural output. Items: {len(neuralOutputMetrics)} Rows: {len(neuralOutputMetrics[0])} Columns: {len(neuralOutputMetrics)}')
end = time.time()
print(f"Time elapsed on data preparation: {convTime-start}")
print(f"Time elapsed on convolutions: {neuralTime-convTime}")
print(f"Time elapsed on neural layers: {end-neuralTime}")
print(f"Total time elapsed: {end-start}")
