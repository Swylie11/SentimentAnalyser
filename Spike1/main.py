import WordVectorConversions as wvc
import json
from types import SimpleNamespace

# entry_data = input('Please input the text to be analysed: ')


def fetch_test_data(test_data):
    """ Fetches batch review data from the test data file """
    count = 0
    with open(test_data, 'r', encoding='utf8') as f:
        output = []
        for line in f:  # For every review
            if count < 10:
                review = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
                output.append(review.text)  # Adds the review to the batch output
                count += 1
            else:
                break
        f.close()
        return output


test_data_file = 'C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/TestData.jsonl'

entry_data = fetch_test_data(test_data_file)

print(entry_data)
print(len(entry_data))
# print(wvc.pad_matrix(wvc.return_vector_matrix(wvc.format_entry_data(entry_data))))
