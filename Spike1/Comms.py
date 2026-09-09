import ast
import json
import os
import random
import sqlite3
from types import SimpleNamespace

import numpy as np


# Opening a fresh SQLite connection per row was a large part of why training was
# slow: fetch_embedding ran once per word, so a 200 word review opened 200
# connections. Connections are cached per database file and reused.
_connections = {}

# Word embeddings never change during a run, so they are memoised. The dataset
# vocabulary is small enough that this stays well within memory.
_embedding_cache = {}


def connect(db_name):
    """Returns a cached connection to a database file next to this module."""
    if db_name not in _connections:
        db_path = os.path.join(os.path.dirname(__file__), db_name)
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found: {db_path}. Run InitDatabases.py to create it.")
        _connections[db_name] = sqlite3.connect(db_path)
    return _connections[db_name]


def fetch_all_review_ids():
    """Returns every review id in the dataset, in table order."""
    cursor = connect("test_data.db").execute("SELECT id FROM test_dataset4 ORDER BY id")
    return [row[0] for row in cursor.fetchall()]


def split_ids(review_ids, validation_size, seed):
    """Splits ids into a training list and a held out validation list.

    The shuffle is seeded so the same split comes back on every run: a validation
    set that changes between runs is not a validation set. """
    shuffled = list(review_ids)
    random.Random(seed).shuffle(shuffled)

    validation_size = min(validation_size, len(shuffled) // 2)
    return shuffled[validation_size:], shuffled[:validation_size]


def fetch_batch(review_ids):
    """Fetches many reviews in one query rather than one query per review.

    Returns [(rating, text), ...] in the order the ids were given. """
    review_ids = list(review_ids)
    if not review_ids:
        return []

    conn = connect("test_data.db")
    by_id = {}

    # SQLite caps the number of bound variables per statement, so chunk the ids.
    chunk_size = 500
    for start in range(0, len(review_ids), chunk_size):
        chunk = review_ids[start:start + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT id, rating, text FROM test_dataset4 WHERE id IN ({placeholders})",
            chunk).fetchall()
        for row in rows:
            by_id[row[0]] = (row[1], row[2])

    missing = [i for i in review_ids if i not in by_id]
    if missing:
        raise LookupError(f"No test data for ids {missing[:5]} in test_data.db.")

    return [by_id[i] for i in review_ids]


def fetch_kernel(LayerNum):
    """Returns the stored kernel for a convolution layer."""
    row = connect("convolution_layers.db").execute(
        "SELECT kernel FROM kernels WHERE LayerNum = ?", (LayerNum,)).fetchone()

    if row is None:
        raise LookupError(f"No kernel row for LayerNum={LayerNum}. Run InitDatabases.py.")
    return ast.literal_eval(row[0])


def fetch_layer(LayerNum):
    """Returns [weights, biases] for a dense layer."""
    row = connect("neuron_weights.db").execute(
        "SELECT weights, biases FROM weights WHERE LayerNum = ?", (LayerNum,)).fetchone()

    if row is None:
        raise LookupError(f"No weights row for LayerNum={LayerNum}. Run InitDatabases.py.")
    return [ast.literal_eval(row[0]), ast.literal_eval(row[1])]


def format_data(encoded_data_file):

    conn3 = sqlite3.connect('test_data.db')
    curr = conn3.cursor()

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0

    with open(encoded_data_file, 'r', encoding='utf-8') as f:  # Opens file in read mode
        for line in f:  # Extracting each line
            embedding = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
            result_review = embedding.rating
            result_text = embedding.text  # Separating reviews and ratings

            if int(result_review) == 1 and count1 < 4000 and 200 >= len(result_text.split()) >= 25:
                count1 += 1

                # Adding data to dataset
                curr.execute("INSERT INTO Test_data (Rating, Text) VALUES (?, ?)", (int(result_review), str(result_text)))

            elif int(result_review) == 2 and count2 < 4000 and 200 >= len(result_text.split()) >= 25:
                count2 += 1

                # Adding data to dataset
                curr.execute("INSERT INTO Test_data (Rating, Text) VALUES (?, ?)", (int(result_review), str(result_text)))

            elif int(result_review) == 3 and count3 < 4000 and 200 >= len(result_text.split()) >= 25:
                count3 += 1

                # Adding data to dataset
                curr.execute("INSERT INTO Test_data (Rating, Text) VALUES (?, ?)", (int(result_review), str(result_text)))

            elif int(result_review) == 4 and count4 < 4000 and 200 >= len(result_text.split()) >= 25:
                count4 += 1

                # Adding data to dataset
                curr.execute("INSERT INTO Test_data (Rating, Text) VALUES (?, ?)", (int(result_review), str(result_text)))

            elif int(result_review) == 5 and count5 < 4000 and 200 >= len(result_text.split()) >= 25:
                count5 += 1

                # Adding data to dataset
                curr.execute("INSERT INTO Test_data (Rating, Text) VALUES (?, ?)", (int(result_review), str(result_text)))

    print('Done')

    # Closing database connection
    conn3.commit()
    conn3.close()


def fetch_embedding(word):
    """Returns the embedding for a word, or a zero vector if it is not in the table."""
    if word in _embedding_cache:
        return _embedding_cache[word]

    row = connect("word_embeddings.db").execute(
        "SELECT embedding FROM embeddings WHERE word = ?", (word,)).fetchone()

    embedding = np.zeros(300).tolist() if row is None else ast.literal_eval(row[0])
    _embedding_cache[word] = embedding
    return embedding


def update_values(layer_number, new_weights, new_biases):
    """Writes a dense layer's weights and biases back to the database."""
    conn = connect("neuron_weights.db")
    conn.execute("UPDATE weights SET weights = ?, biases = ? WHERE LayerNum = ?",
                 (str(new_weights), str(new_biases), layer_number))
    conn.commit()


def update_kernel(new_kernel, layer_number):
    """Writes a convolution layer's kernel back to the database."""
    conn = connect("convolution_layers.db")
    conn.execute("UPDATE kernels SET kernel = ? WHERE LayerNum = ?",
                 (str(new_kernel), layer_number))
    conn.commit()


def make_table():
    conn4 = sqlite3.connect("test_data.db")
    curr4 = conn4.cursor()

    curr4.execute("""CREATE TABLE IF NOT EXISTS test_dataset4
    (id INTEGER PRIMARY KEY AUTOINCREMENT, review_id INTEGER, rating INTEGER, text TEXT)""")

    conn4.commit()
    conn4.close()


def fix_id_values():
    conn4 = sqlite3.connect("test_data.db")
    curr4 = conn4.cursor()

    for count in range(100000):  # For each database value

        curr4.execute("""UPDATE test_data SET review_id = ? WHERE review_id = ?""", (int(count+1), int(count+119949)))

    conn4.commit()
    conn4.close()


def randomise_dataset():
    conn4 = sqlite3.connect("test_data.db")
    curr4 = conn4.cursor()

    order = np.arange(1, 100001, 1).tolist()
    random.shuffle(order)

    for i in range(100000):
        curr4.execute("""INSERT INTO test_dataset3 SELECT * FROM test_data WHERE review_id = ?""", (int(order[i]),))

    conn4.commit()
    conn4.close()
