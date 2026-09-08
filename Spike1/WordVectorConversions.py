import string
import numpy as np
import Comms as com


def get_vector2(target):
    return com.fetch_embedding(target)


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


def return_vector_matrix_jsonl(list_input):
    """ Fetches the embeddings from a jsonl file """
    output_tensor = []
    for i in range(len(list_input)):
        temp_matrix = []
        sentence = list_input[i]
        words = sentence.strip().split()
        for n in range(len(words)):
            temp_matrix.append(get_vector2(words[n]))
        output_tensor.append(temp_matrix)
    return output_tensor


WORD_LIMIT = 200
EMBEDDING_DIM = 300


def pad_matrix(tensor_input):
    """ Pads or truncates every sentence to WORD_LIMIT words, ready for the convolution.

    Truncation matters now that a batch holds several reviews: one over-long review
    would otherwise make the batch ragged and break the whole forward pass rather
    than just its own row. """
    blue = np.arange(EMBEDDING_DIM)  # Blueprint for size of filled value matrix
    for n in range(len(tensor_input)):  # Iterates through all sentences provided
        del tensor_input[n][WORD_LIMIT:]  # Drops anything past the word limit
        to_pad = WORD_LIMIT - len(tensor_input[n])  # Finds how many words are left in for the word limit
        for i in range(to_pad):  # Iterates through all remaining 'empty' words in the list
            tensor_input[n].append(np.full_like(blue, 0.001, dtype=np.double).tolist())  # Adds fake words which have weights of tiny values top avoid dead neurons
    return tensor_input

