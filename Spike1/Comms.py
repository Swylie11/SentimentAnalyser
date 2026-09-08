import sqlite3
import ast
from types import SimpleNamespace
import json
import random
import numpy as np


import os


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
    import os, sqlite3, ast
    # use DB file next to this module to avoid working-dir issues
    db_path = os.path.join(os.path.dirname(__file__), "convolution_layers.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn2 = sqlite3.connect(db_path)
    cur = conn2.cursor()

    # verify table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kernels'")
    if cur.fetchone() is None:
        # list existing tables for debugging
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        conn2.close()
        raise RuntimeError(f"Table 'kernels' not found in {db_path}. existing tables: {tables}")

    # fetch kernel row
    cur.execute("SELECT kernel FROM kernels WHERE LayerNum = ?", (LayerNum,))
    row = cur.fetchone()
    conn2.close()
    if row is None:
        raise LookupError(f"No kernel row for LayerNum={LayerNum} in {db_path}.")
    return ast.literal_eval(row[0])


def fetch_layer(LayerNum):
    import os, sqlite3, ast
    db_path = os.path.join(os.path.dirname(__file__), "neuron_weights.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT weights FROM weights WHERE LayerNum = ?", (LayerNum,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise LookupError(f"No weights row for LayerNum={LayerNum} in {db_path}.")
    weights = ast.literal_eval(row[0])

    cur.execute("SELECT biases FROM weights WHERE LayerNum = ?", (LayerNum,))
    row2 = cur.fetchone()
    conn.close()
    if row2 is None:
        raise LookupError(f"No biases row for LayerNum={LayerNum} in {db_path}.")
    biases = ast.literal_eval(row2[0])

    return [weights, biases]


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


def fetch_test_data(review_id):
    import os, sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "test_data.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM test_dataset4 WHERE id = ?", (review_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise LookupError(f"No test data for id={review_id} in {db_path}.")
    return [row[2], row[3]]


def update_values(layer_number, new_weights, new_biases):
    import os, sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "neuron_weights.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""UPDATE weights SET weights = ?, biases = ? WHERE LayerNum = ?""",
                (str(new_weights), str(new_biases), layer_number))
    conn.commit()
    conn.close()


# ...existing code...
def update_kernel(new_kernel, layer_number):
    import os, sqlite3
    # open the DB file next to this module to avoid working-dir issues
    db_path = os.path.join(os.path.dirname(__file__), "convolution_layers.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # verify table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kernels'")
    if cur.fetchone() is None:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        conn.close()
        raise RuntimeError(f"Table 'kernels' not found in {db_path}. existing tables: {tables}")

    # perform update
    cur.execute("UPDATE kernels SET kernel = ? WHERE LayerNum = ?", (str(new_kernel), layer_number))
    conn.commit()
    conn.close()
# ...existing code...


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
