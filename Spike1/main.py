import WordVectorConversions as wvc

entry_data = input('Please input the text to be analysed: ')

print(wvc.return_vector_matrix(wvc.format_entry_data(entry_data)))
