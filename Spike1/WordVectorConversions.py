import string
import numpy as np
import json
from types import SimpleNamespace


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


def get_vector(target, embeddings):
    with open(embeddings, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            try:
                embedding = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
                option = embedding.word
                count += 1
                print(f'Words checked: {count}')
                if option == target:
                    return embedding.vector
            except:
                pass


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
        cleaned_sentence = cleaned_sentence.replace('     ', ' ')  # Removes quintuple spaces
        cleaned_sentence = cleaned_sentence.replace('      ', ' ')  # Removes sextuple spaces
        output_batch.append(cleaned_sentence)  # Adds the cleaned sentence to the output list

    return output_batch


def return_vector_matrix_txt(list_input):
    """ Fetches the embeddings from the text file DO NOT USE"""
    output_tensor = []
    for i in range(len(list_input)):
        temp_matrix = []
        sentence = list_input[i]
        words = sentence.strip().split()
        for n in range(len(words)):
            temp_matrix.append(get_embedding(get_vector_line(words[n], input_file_directory)))
        output_tensor.append(temp_matrix)
    return output_tensor


def return_vector_matrix_jsonl(list_input):
    """ Fetches the embeddings from a jsonl file """
    output_tensor = []
    for i in range(len(list_input)):
        temp_matrix = []
        sentence = list_input[i]
        words = sentence.strip().split()
        for n in range(len(words)):
            temp_matrix.append(get_vector(words[n], input_file_directory))
        output_tensor.append(temp_matrix)
    return output_tensor


def pad_matrix(tensor_input):
    """ Pads an input matrix so that all entered lists are of length 200 ready for the convolution process """
    for n in range(len(tensor_input)):  # Iterates through all sentences provided
        to_pad = 200 - len(tensor_input[n])  # Finds how many words are left in for the word limit
        print(to_pad, n)
        for i in range(to_pad):  # Iterates through all remaining 'empty' words in the list
            tensor_input[n].append(np.zeros(300).tolist())  # Adds fake words which have weights of all zeros
    return tensor_input


input_file_directory = ('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data'
                        '/WordEmbeddings.jsonl')

'''
# This is purely for comparison purposes
formatted_data = format_entry_data(['The tall man walked across the field'])

choice = int(input('Choose file search method\n1. Text file\n2. Jsonl file\n'))
if choice == 1:
    input_file_directory = ('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data'
                            '/WordEmbeddings.txt')
    start = time.time()
    vector_result = return_vector_matrix_txt(formatted_data)
else:
    input_file_directory = ('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data'
                            '/WordEmbeddings.jsonl')
    start = time.time()
    vector_result = return_vector_matrix_jsonl(formatted_data)

padded_result = pad_matrix(vector_result)
end = time.time()
print(padded_result)
print(f'Time elapsed: {end - start}')

input_file_directory = ('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data'
                        '/WordEmbeddings.jsonl')



start = time.time()
print(formatted_data)
vector_result = return_vector_matrix_jsonl(formatted_data)
padded_result = pad_matrix(vector_result)
print(padded_result)
end = time.time()
print(f'Time elapsed = {end - start}')
'''
