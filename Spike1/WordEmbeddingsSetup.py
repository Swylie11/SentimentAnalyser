def file_to_dict(file_path):
    """ This function takes a word embeddings file as input and converts it to a dictionary """
    result_dict = {}
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:  # For each line in the input file
            try:
                parts = line.strip().split()
                if parts:
                    key = parts[0]
                    values = list(map(float, parts[1:]))  # references all items other than the first one in the list
                    result_dict[key] = values  # Constructs the new dictionary
            except:  # If the value cannot be read or written, ignore it.
                pass
    return result_dict


def write_dict_to_file(dictionary, output_file):
    """ This function takes a dictionary and writes it to a new file """
    with open(output_file, 'w') as file:
        file.write('{\n')
        for key, values in dictionary.items():  # For each key and value in the dictionary
            try:
                file.write(f"    '{key}': {values},\n")  # Write the key and value to the new file
            except:  # If read value cannot be written, ignore it.
                pass
        file.write('}\n')
    print(f"Dictionary written to {output_file}")


'''
def clean_file(input_file, output_file):
    """ This function takes input of a file and removes all lines where the first word contains unwanted content """
    with open(input_file, 'r', encoding='utf8') as filein, open(output_file, 'w', encoding='utf8') as fileout:
        for line in filein:  # For each line in the input file
            try:
                words = line.split()  # Separate the lines in the file to individual lists of their words
                if any(i not in ".,';[]=-/\|<>?:@{}_+!"£$%^&*()1234567890~#¬`" for i in words[0]):  # Search the lines for unwanted content
                    fileout.write(line)  # Write selected data to new file
            except:
                pass
    filein.close(), fileout.close()
    print(f'File cleaned and written to: {output_file}')
'''


def clean_file(input_file, output_file):
    """ This function takes input of a file and removes all lines where the first word contains unwanted content """
    with open(input_file, 'r', encoding='utf8') as filein, open(output_file, 'w', encoding='utf8') as fileout:
        for line in filein:  # For each line in the input file
            if "'" not in line:  # Search the lines for unwanted content
                fileout.write(line)  # Write selected data to new file
    filein.close(), fileout.close()
    print(f'File cleaned and written to: {output_file}')


# Example usage
# input_file_path = ('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/WordEmbeddings.txt')
# output_file_path = ('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Spike1/WordEmbeddingsDictionary.py')

# Convert file line by line to dictionary
# dictionary_input = file_to_dict(input_file_path)

# Write new dictionary to a new file
# write_dict_to_file(dictionary_input, output_file_path)

# Search raw data for unnecessary embeddings and remove them from file copy
# clean_file(input_file_path, output_file_path)
