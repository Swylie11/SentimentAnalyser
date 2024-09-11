import json
from types import SimpleNamespace


def create_new_file(source, end_location):
    """ This function takes input of a jsonl file and writes only the necessary data to a new file """
    with open(source, 'r', encoding='utf8') as f:
        with open(end_location, 'w', encoding='utf8') as e:
            count = 0
            for line in f:
                if count <= 100000:  # Writes 100000 lines to the new file (100,000 reviews)
                    review = json.loads(line, object_hook=lambda d: SimpleNamespace(**d))
                    # Setting review constraints below
                    if 15 <= len(review.text.split()) <= 200 and ("\\" not in review.text):
                        e.writelines(f'{{"rating": {review.rating}, "text": "{review.text}"}}\n')
                        count += 1
        e.close()
    f.close()


source_directory = 'C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/Software.jsonl'
end_directory = 'C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/TestData.jsonl'

create_new_file(source_directory, end_directory)
