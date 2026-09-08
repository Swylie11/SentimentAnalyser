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
                    if 15 <= len(review.text.split()) <= 200:
                        # json.dumps escapes quotes, backslashes and newlines. The old
                        # f-string did not, so any review containing a double quote
                        # produced an invalid line that InitDatabases.py then skipped
                        # silently. Reviews containing quotes are not a random subset,
                        # so that lost data and biased the corpus at the same time.
                        e.write(json.dumps({"rating": review.rating, "text": review.text},
                                           ensure_ascii=False) + "\n")
                        count += 1
        e.close()
    f.close()


if __name__ == "__main__":
    source_directory = 'C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/Software.jsonl'
    end_directory = 'C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/TestData.jsonl'

    create_new_file(source_directory, end_directory)
