import string
import numpy as np


def get_vector_line(word, embedding_file):
    """ Searches the word embeddings file for the line that contains the target word and returns it """
    with open(embedding_file, 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            # Converts each line into a list of words and checks if the first word is the target word
            words = line.strip().split()
            if words[0] == word:
                return line
            file.close()
        print('Word not found')  # If the word is not found, print not found and return nothing
        file.close()
        return None


def get_embedding(line):
    """ Takes the line from the word embeddings file and outputs the values as a list """
    values = line.strip().split()
    return values[1:]


def format_entry_data(batch_input):
    """ Converts a list of sentences in the wrong format into a list of cleaned sentences """
    output_batch = []
    for i in range(len(batch_input)):
        sentence = batch_input[i].lower()  # Convert to lowercase
        cleaned_sentence = ""
        # Iterate through each character in the sentence
        for char in sentence:
            # Add to cleaned_sentence if char is not punctuation or a number
            if char not in string.punctuation and not char.isdigit():
                cleaned_sentence += char
        cleaned_sentence = cleaned_sentence.replace('  ', ' ')  # Removes double spaces
        cleaned_sentence = cleaned_sentence.replace('   ', ' ')  # Removes triple spaces
        cleaned_sentence = cleaned_sentence.replace('    ', ' ')  # Removes quadruple spaces
        output_batch.append(cleaned_sentence)  # Adds the cleaned sentence to the output list

    return output_batch


def return_vector_matrix(list_input):
    """ Performs the get_embedding function for each word in a sentence and outputs the resulting matrix """
    output_tensor = []
    for i in range(len(list_input)):
        temp_matrix = []
        sentence = list_input[i]
        words = sentence.strip().split()
        for n in range(len(words)):
            temp_matrix.append(get_embedding(get_vector_line(words[n], input_file_directory)))
        output_tensor.append(temp_matrix)
    return output_tensor


def pad_matrix(tensor_input):
    """ Pads an input matrix so that all entered lists are of length 200 ready for the convolution process """
    for n in range(len(tensor_input)):  # Iterates through all sentences provided
        to_pad = 200 - len(tensor_input[n])  # Finds how many words are left in for the word limit
        print(to_pad, n)
        for i in range(to_pad):  # Iterates through all remaining 'empty' words in the list
            tensor_input[n].append(np.zeros(300))  # Adds fake words which have weights of all zeros
    return tensor_input


input_file_directory = ('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data'
                        '/WordEmbeddings.txt')

formatted_data = format_entry_data(['I like all people with hairy heads.', 'I do not like all people with hair'])
vector_result = return_vector_matrix(formatted_data)
padded_result = pad_matrix(vector_result)

print(formatted_data)
print(padded_result)
