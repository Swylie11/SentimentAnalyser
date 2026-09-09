"""Creates and seeds the SQLite databases the model reads from.

The four .db files are not kept in the repository, so this script builds them.
It has two modes:

  --synthetic   Builds a small self-contained corpus and embedding table. This
                exists so the gradient check and the overfit test can be run on
                a machine that does not have the Amazon review dump. Accuracy
                measured against it says nothing about real sentiment data.

  --embeddings / --reviews
                Imports the real data from the .jsonl files produced by
                WordEmbeddingsSetup.py and TestDataSetup.py.

Layer shapes are derived by running one forward pass through the convolution
layers, so the dense stack always matches whatever the kernels actually emit.
"""

import argparse
import json
import os
import random
import sqlite3

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CONV_DB = os.path.join(HERE, "convolution_layers.db")
WEIGHTS_DB = os.path.join(HERE, "neuron_weights.db")
EMBEDDINGS_DB = os.path.join(HERE, "word_embeddings.db")
TEST_DATA_DB = os.path.join(HERE, "test_data.db")

# Kernel sizes for the two convolution layers. The stride is set in main.py
# where the ConvLayer objects are constructed.
KERNEL_SHAPES = {1: (5, 5), 2: (3, 3)}

# Widths of the dense stack after the flatten. The input width is measured, and
# the final 5 is the number of star ratings.
HIDDEN_WIDTHS = [256, 128, 64, 32, 16]
NUM_CLASSES = 5

EMBEDDING_DIM = 300
MAX_WORDS = 200


def create_schemas():
    """Creates every table the model reads, without dropping existing rows."""
    with sqlite3.connect(CONV_DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS kernels (LayerNum INTEGER PRIMARY KEY, kernel TEXT)")

    with sqlite3.connect(WEIGHTS_DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS weights (LayerNum INTEGER PRIMARY KEY, weights TEXT, biases TEXT)")

    with sqlite3.connect(EMBEDDINGS_DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS embeddings (word TEXT PRIMARY KEY, embedding TEXT)")

    with sqlite3.connect(TEST_DATA_DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS test_dataset4
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, review_id INTEGER, rating INTEGER, text TEXT)""")


def seed_kernels():
    """Writes placeholder kernels so ConvLayer.fetchKernel has a shape to read."""
    with sqlite3.connect(CONV_DB) as conn:
        for layer_num, (rows, cols) in KERNEL_SHAPES.items():
            kernel = np.zeros((rows, cols)).tolist()
            conn.execute("INSERT OR REPLACE INTO kernels (LayerNum, kernel) VALUES (?, ?)",
                         (layer_num, str(kernel)))


def measure_flattened_width():
    """Runs one forward pass through both convolution layers to size the dense stack."""
    from ConvolutionLayer import ConvLayer

    conv1 = ConvLayer(1, 5)
    conv2 = ConvLayer(2, 3)
    conv1.fetchKernel()
    conv2.fetchKernel()

    # A single all-zero review of the maximum supported length.
    dummy = [np.zeros((MAX_WORDS, EMBEDDING_DIM)).tolist()]

    out1 = conv1.convPass(conv1.reflectMatrix(dummy))
    out2 = conv2.convPass(conv2.reflectMatrix(out1))

    shape = np.asarray(out2, dtype=float).shape  # (batch, height, width)
    return int(shape[1] * shape[2])


def seed_layers(input_width):
    """Writes correctly shaped placeholder weights for all six dense layers.

    Values are zeros; main.py calls initialise_values() to overwrite them with a
    He-scaled draw. Only the shape matters here.
    """
    widths = [input_width] + HIDDEN_WIDTHS + [NUM_CLASSES]

    with sqlite3.connect(WEIGHTS_DB) as conn:
        for layer_num in range(1, len(widths)):
            n_in = widths[layer_num - 1]
            n_out = widths[layer_num]

            # Stored orientation is (n_out, n_in); batch_layer_output transposes it.
            weights = np.zeros((n_out, n_in)).tolist()
            biases = np.zeros(n_out).tolist()

            conn.execute("INSERT OR REPLACE INTO weights (LayerNum, weights, biases) VALUES (?, ?, ?)",
                         (layer_num, str(weights), str(biases)))

    print(f"Seeded {len(widths) - 1} dense layers: {' -> '.join(str(w) for w in widths)}")


def import_embeddings(path):
    """Loads embeddings from the .jsonl file written by WordEmbeddingsSetup.write_as_jsonl."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line, strict=False)
            except ValueError:
                continue
            rows.append((record["word"], str(record["vector"])))

    with sqlite3.connect(EMBEDDINGS_DB) as conn:
        # Clear first, as import_reviews does. Without this, an earlier --synthetic
        # run leaves its random vectors behind for any word the real file does not
        # also contain, and the model silently reads noise for those words.
        conn.execute("DELETE FROM embeddings")
        conn.executemany("INSERT OR REPLACE INTO embeddings (word, embedding) VALUES (?, ?)", rows)

    print(f"Imported {len(rows)} embeddings from {path}")


def import_reviews(path):
    """Loads reviews from the .jsonl file written by TestDataSetup.create_new_file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for review_id, line in enumerate(f, start=1):
            try:
                record = json.loads(line, strict=False)
            except ValueError:
                continue
            rows.append((review_id, int(record["rating"]), str(record["text"])))

    with sqlite3.connect(TEST_DATA_DB) as conn:
        conn.execute("DELETE FROM test_dataset4")
        conn.executemany("INSERT INTO test_dataset4 (review_id, rating, text) VALUES (?, ?, ?)", rows)

    print(f"Imported {len(rows)} reviews from {path}")


# Vocabulary used to build the synthetic corpus. Each rating draws from its own
# word pool, so the task is learnable and an overfit test is meaningful.
SYNTHETIC_VOCAB = {
    1: ["awful", "broken", "useless", "refund", "terrible", "worst", "garbage", "unusable"],
    2: ["poor", "clunky", "disappointing", "flawed", "sluggish", "weak", "annoying", "lacking"],
    3: ["fine", "okay", "average", "adequate", "middling", "acceptable", "plain", "ordinary"],
    4: ["good", "solid", "helpful", "reliable", "pleasant", "capable", "handy", "decent"],
    5: ["excellent", "brilliant", "perfect", "outstanding", "superb", "flawless", "wonderful", "best"],
}

SYNTHETIC_FILLER = ["the", "and", "a", "of", "it", "was", "for", "this", "with", "to", "is", "in"]


def seed_synthetic(num_reviews, seed):
    """Builds a synthetic corpus plus matching embeddings.

    Each rating has a distinct word pool and each word gets a random embedding,
    so a correctly wired network can separate the classes. Classes are balanced.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    vocab = sorted({w for pool in SYNTHETIC_VOCAB.values() for w in pool} | set(SYNTHETIC_FILLER))

    embedding_rows = []
    for word in vocab:
        vector = np_rng.normal(0.0, 1.0, EMBEDDING_DIM).round(6).tolist()
        embedding_rows.append((word, str(vector)))

    with sqlite3.connect(EMBEDDINGS_DB) as conn:
        conn.execute("DELETE FROM embeddings")
        conn.executemany("INSERT INTO embeddings (word, embedding) VALUES (?, ?)", embedding_rows)

    review_rows = []
    for review_id in range(1, num_reviews + 1):
        rating = 1 + (review_id - 1) % NUM_CLASSES  # Balanced across the five classes.
        signal = SYNTHETIC_VOCAB[rating]

        # 30 words: a mix of rating-specific words and shared filler.
        words = [rng.choice(signal) if rng.random() < 0.6 else rng.choice(SYNTHETIC_FILLER)
                 for _ in range(30)]
        review_rows.append((review_id, rating, " ".join(words)))

    rng.shuffle(review_rows)
    # review_id stays the shuffled row's own identifier; id is assigned in insert order.
    with sqlite3.connect(TEST_DATA_DB) as conn:
        conn.execute("DELETE FROM test_dataset4")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'test_dataset4'")
        conn.executemany("INSERT INTO test_dataset4 (review_id, rating, text) VALUES (?, ?, ?)", review_rows)

    print(f"Seeded {len(review_rows)} synthetic reviews and {len(embedding_rows)} embeddings (seed={seed})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", type=int, metavar="N",
                        help="Seed N synthetic reviews instead of importing real data")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --synthetic")
    parser.add_argument("--embeddings", metavar="JSONL",
                        help="Path to the word embeddings .jsonl file")
    parser.add_argument("--reviews", metavar="JSONL",
                        help="Path to the review .jsonl file")
    args = parser.parse_args()

    create_schemas()
    seed_kernels()

    if args.embeddings:
        import_embeddings(args.embeddings)
    if args.reviews:
        import_reviews(args.reviews)
    if args.synthetic:
        seed_synthetic(args.synthetic, args.seed)

    seed_layers(measure_flattened_width())
    print("Databases ready.")


if __name__ == "__main__":
    main()
