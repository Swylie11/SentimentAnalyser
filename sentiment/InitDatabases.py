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

The dense stack's input width is one feature per filter bank filter, summed across
every branch in FILTER_BANK_CONFIG -- global max-pooling makes this independent of
review length, so it is computed directly rather than measured with a dummy pass.
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

# Parallel filter bank branches, Kim (2014) style: (LayerNum, region_size, num_filters).
# Each branch slides its filters along the word axis only (kernel width is always the
# full embedding dimension) and globally max-pools over word position, so branches run
# in parallel over the same embedding input rather than stacking like the old
# ConvLayer pair did. See FilterBankLayer.py.
FILTER_BANK_CONFIG = [(1, 3, 100), (2, 4, 100), (3, 5, 100)]

# Widths of the dense stack after the flatten. The input width is measured, and
# the final 5 is the number of star ratings.
HIDDEN_WIDTHS = [256, 128, 64, 32, 16]
NUM_CLASSES = 5

EMBEDDING_DIM = 300
MAX_WORDS = 200


def create_schemas():
    """Creates every table the model reads, without dropping existing rows.

    The kernels table is the one exception: it only ever holds regenerable,
    backed-up model state (never source data), and its old schema (kernel only, no
    bias) predates FilterBankLayer, so it is dropped and recreated with a bias column
    rather than migrated in place.
    """
    with sqlite3.connect(CONV_DB) as conn:
        conn.execute("DROP TABLE IF EXISTS kernels")
        conn.execute("CREATE TABLE kernels (LayerNum INTEGER PRIMARY KEY, kernel TEXT, bias TEXT)")

    with sqlite3.connect(WEIGHTS_DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS weights (LayerNum INTEGER PRIMARY KEY, weights TEXT, biases TEXT)")

    with sqlite3.connect(EMBEDDINGS_DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS embeddings (word TEXT PRIMARY KEY, embedding TEXT)")

    with sqlite3.connect(TEST_DATA_DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS test_dataset4
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, review_id INTEGER, rating INTEGER, text TEXT)""")


def seed_kernels():
    """Writes placeholder kernels and biases so FilterBankLayer.fetchKernel has a
    shape to read."""
    with sqlite3.connect(CONV_DB) as conn:
        for layer_num, region_size, num_filters in FILTER_BANK_CONFIG:
            kernel = np.zeros((num_filters, region_size, EMBEDDING_DIM)).tolist()
            bias = np.zeros(num_filters).tolist()
            conn.execute("INSERT OR REPLACE INTO kernels (LayerNum, kernel, bias) VALUES (?, ?, ?)",
                         (layer_num, str(kernel), str(bias)))


def flattened_width():
    """The dense stack's input width: one feature per filter, concatenated across
    every branch. Global max-pooling makes this independent of MAX_WORDS/WORD_LIMIT,
    unlike the old stacked-conv design which needed a dummy forward pass to measure it."""
    return sum(num_filters for _, _, num_filters in FILTER_BANK_CONFIG)


def sanity_check_filter_banks():
    """Runs one forward pass per branch on a dummy all-zero review, purely to catch
    config typos (e.g. a region size larger than MAX_WORDS) before a real run."""
    from FilterBankLayer import FilterBankLayer

    dummy = np.zeros((1, MAX_WORDS, EMBEDDING_DIM))
    for layer_num, region_size, num_filters in FILTER_BANK_CONFIG:
        assert region_size <= MAX_WORDS, (
            f"branch {layer_num}: region_size {region_size} exceeds MAX_WORDS {MAX_WORDS}")
        layer = FilterBankLayer(layer_num, region_size, num_filters, EMBEDDING_DIM)
        layer.kernel = np.zeros((num_filters, region_size, EMBEDDING_DIM))
        layer.bias = np.zeros(num_filters)
        output = layer.convPass(dummy)
        assert output.shape == (1, num_filters), (
            f"branch {layer_num} produced shape {output.shape}, expected (1, {num_filters})")


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
    sanity_check_filter_banks()

    if args.embeddings:
        import_embeddings(args.embeddings)
    if args.reviews:
        import_reviews(args.reviews)
    if args.synthetic:
        seed_synthetic(args.synthetic, args.seed)

    seed_layers(flattened_width())
    print("Databases ready.")


if __name__ == "__main__":
    main()
