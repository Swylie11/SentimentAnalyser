import string


def get_vector_line(word, embedding_file):
    """ Searches the word embeddings file for the line that contains the target word and returns it """
    with open(embedding_file, 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            # Converts each line into a list of words and checks if the first word is the target word
            words = line.strip().split()
            if words[0] == word:
                return line
        print('Word not found')
        return None


def get_embedding(line):
    """ Takes the line from the word embeddings file and outputs the values as a list """
    values = line.strip().split()
    return values[1:]


def format_entry_data(batch_input):
    """ Converts a list of sentences in the wrong format into a list of cleaned sentences """
    output_batch = []
    for i in range(len(batch_input)):
        # Convert to lowercase
        sentence = batch_input[i].lower()

        # Initialize an empty string for the cleaned sentence
        cleaned_sentence = ""

        # Iterate through each character in the sentence
        for char in sentence:
            # Add to cleaned_sentence if char is not punctuation or a number
            if char not in string.punctuation and not char.isdigit():
                cleaned_sentence += char
        cleaned_sentence = cleaned_sentence.replace('  ', ' ')  # Removes double spaces
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


input_file_directory = ('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data'
                        '/WordEmbeddings.txt')

print(format_entry_data(['Hello, World! This is a test sentence with numbers 12345 and punctuation!!!',
                         'Hello, World! This is a test sentence with numbers 12345 and punctuation!!!',
                         'Hello, World! This is a test sentence with numbers 12345 and punctuation!!!']))

print(return_vector_matrix(format_entry_data(['Hello, World! This is a test sentence with numbers 12345 and punctuation!!!',
                                              'Hello, World! This is a test sentence with numbers 12345 and punctuation!!!',
                                              'Hello, World! This is a test sentence with numbers 12345 and punctuation!!!'])))
