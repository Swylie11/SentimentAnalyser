import json
from types import SimpleNamespace


def sort_words(source_file):
    with open(source_file, 'r', encoding='utf-8') as sf:
        count = 0
        for line in sf:
            count += 1
        pivotIndex = count // 2
        line = sf.read()
        print(line)
        embedding = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
        word = embedding.word
        print(word)


sort_words('C:/Users/samja/Documents/SchoolWork/ComputerScience/Project/SentimentAnalyser/Data/Test.jsonl')
