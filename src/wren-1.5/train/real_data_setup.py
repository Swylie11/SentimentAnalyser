"""Builds the two .jsonl files `wren setup` imports, from real source data.

This replaces the hardcoded-path scripts (TestDataSetup.py, WordEmbeddingsSetup.py)
for the Amazon Reviews 2023 dump and the fastText wiki-news-300d-1M embeddings file. It differs from them
in three ways that matter for getting an honest accuracy figure:

  Balanced classes   Amazon ratings skew hard to 5 star. Taking the file in order
                     gives a corpus that is more than half 5 star, and a model can
                     then score well by leaning on the prior rather than reading
                     the review. Each rating is sampled to the same count.

  Uniform sampling   Reviews are reservoir sampled across the whole file rather
                     than taken from the front, so the sample is not confined to
                     whichever products happen to appear first.

  Correct escaping   TestDataSetup.py built its JSON with an f-string, so any
                     review containing a double quote or a newline produced an
                     invalid line that InitDatabases.py then silently skipped.
                     Reviews with quotes in them are not a random subset, so that
                     was quiet selection bias as well as lost data. json.dumps
                     handles the escaping.

The embedding table is restricted to the vocabulary the corpus actually uses.
The full 1M word fastText file would make a multi-gigabyte SQLite table that the model
would never read most of. Words absent from fastText keep the existing behaviour:
Comms.fetch_embedding returns a zero vector for them.

Usage:
    python -m wren.train.real_data_setup --reviews-gz  ../../Data/Software.jsonl.gz \
                            --vec         ../../Data/wiki-news-300d-1M.vec \
                            --per-class   1000 \
                            --out-dir     ../../Data
"""

import argparse
import gzip
import json
import os
import random
import re
import sys

from .. import text as wvc

# Matches the word-count window the model is built around: MAX_WORDS is 200, and
# reviews shorter than 15 words carry too little signal to be worth a row.
MIN_WORDS = 15
MAX_WORDS = 200

NUM_CLASSES = 5

# Amazon review text contains literal HTML breaks. Left in place, the cleaner
# strips the angle brackets and leaves "br" behind as a high frequency token.
HTML_TAG = re.compile(r"<[^>]+>")


def open_maybe_gzip(path):
    """Opens a .jsonl or .jsonl.gz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def normalise(text):
    """Lowercased word tokens joined by single spaces, as evaluation/build_test_set.py
    uses, so formatting differences (case, punctuation, whitespace) don't hide a duplicate."""
    return " ".join(re.findall(r"\w+", text.lower()))


def sample_reviews(path, per_class, seed, skip=frozenset()):
    """Reservoir samples `per_class` reviews for each star rating.

    One pass over the file. Every qualifying review has an equal chance of
    ending up in its class's sample regardless of where it sits in the file.
    Reviews whose normalise()d text is in `skip` are never eligible.
    """
    rng = random.Random(seed)
    reservoirs = {star: [] for star in range(1, NUM_CLASSES + 1)}
    seen_counts = {star: 0 for star in range(1, NUM_CLASSES + 1)}
    total_read = 0
    kept_eligible = 0

    with open_maybe_gzip(path) as f:
        for line in f:
            total_read += 1
            try:
                record = json.loads(line)
            except ValueError:
                continue

            text = record.get("text")
            rating = record.get("rating")
            if not text or rating is None:
                continue

            star = int(rating)
            if star not in reservoirs:
                continue

            text = HTML_TAG.sub(" ", text).strip()
            if not (MIN_WORDS <= len(text.split()) <= MAX_WORDS):
                continue
            if skip and normalise(text) in skip:
                continue

            kept_eligible += 1
            seen_counts[star] += 1
            reservoir = reservoirs[star]

            if len(reservoir) < per_class:
                reservoir.append(text)
            else:
                # Standard reservoir sampling: the nth eligible item replaces a
                # uniformly chosen slot with probability per_class/n.
                j = rng.randrange(seen_counts[star])
                if j < per_class:
                    reservoir[j] = text

            if total_read % 500000 == 0:
                filled = sum(len(r) for r in reservoirs.values())
                print(f"  read {total_read:,} lines, {filled:,}/{per_class * NUM_CLASSES:,} sampled",
                      file=sys.stderr)

    print(f"  read {total_read:,} lines, {kept_eligible:,} within the "
          f"{MIN_WORDS}-{MAX_WORDS} word window", file=sys.stderr)
    for star in range(1, NUM_CLASSES + 1):
        print(f"    {star}*: {seen_counts[star]:>8,} eligible -> {len(reservoirs[star]):>6,} sampled",
              file=sys.stderr)

    return reservoirs


def write_reviews(rows, out_path, seed):
    """Writes {"rating": ..., "text": ...} rows as lines, shuffled."""
    rows = list(rows)

    # Interleave the classes. InitDatabases assigns review_id in file order and
    # main.py shuffles before splitting, but a class-ordered file is a trap for
    # anything that takes a prefix of the table.
    random.Random(seed).shuffle(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows):,} reviews to {out_path}")
    return rows


def corpus_vocabulary(rows):
    """Returns the token set the model will actually look up.

    Uses the model's own cleaner, so the words stored are exactly the words
    return_vector_matrix_jsonl asks for. Reimplementing the cleaning here would
    let the two drift apart and quietly turn known words into zero vectors.
    """
    vocab = set()
    chunk = 2000
    texts = [row["text"] for row in rows]
    for start in range(0, len(texts), chunk):
        for cleaned in wvc.format_entry_data(texts[start:start + chunk]):
            vocab.update(cleaned.split())
    return vocab


def write_embeddings(vec_path, vocab, out_path):
    """Writes fastText vectors for the corpus vocabulary as {"word":..., "vector":[...]}."""
    written = 0
    dim = None

    with open(vec_path, "r", encoding="utf-8", errors="replace") as f, \
            open(out_path, "w", encoding="utf-8") as e:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 3:
                continue
            word = parts[0]
            if word not in vocab:
                continue
            try:
                vector = [float(v) for v in parts[1:]]
            except ValueError:
                continue

            if dim is None:
                dim = len(vector)
            elif len(vector) != dim:
                continue

            e.write(json.dumps({"word": word, "vector": vector}) + "\n")
            written += 1

    covered = 100.0 * written / max(len(vocab), 1)
    print(f"Wrote {written:,} embeddings (dim {dim}) to {out_path}")
    print(f"Vocabulary coverage: {written:,}/{len(vocab):,} tokens ({covered:.1f}%) "
          f"found in the .vec file; the rest fall back to zero vectors.")

    if dim is not None and dim != wvc.EMBEDDING_DIM:
        print(f"\nWARNING: embeddings are {dim}-dimensional but the model expects "
              f"{wvc.EMBEDDING_DIM}. Use wiki-news-300d-1M.vec.")

    return written, dim


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews-gz", required=True, nargs="+",
                        help="Amazon Reviews 2023 .jsonl or .jsonl.gz; with several, "
                             "--per-class is sampled from each")
    parser.add_argument("--base", metavar="JSONL",
                        help="an existing {rating, text} corpus to keep as it is and add to")
    parser.add_argument("--exclude", metavar="JSONL", nargs="*", default=[],
                        help="{rating, text} files whose reviews must never be sampled, "
                             "such as the test set")
    parser.add_argument("--vec", required=True,
                        help="fastText wiki-news-300d-1M.vec")
    parser.add_argument("--per-class", type=int, default=1000,
                        help="Reviews to sample per star rating (default 1000)")
    parser.add_argument("--out-dir", default=".", help="Where to write the .jsonl files")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reviews_out = os.path.join(args.out_dir, "TestData.jsonl")
    embeddings_out = os.path.join(args.out_dir, "WordEmbeddings.jsonl")

    def read_rows(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    rows = read_rows(args.base) if args.base else []
    # Never sample a test review, or a review already in the corpus (the same text
    # can be posted under more than one category).
    skip = {normalise(row["text"]) for path in args.exclude for row in read_rows(path)}
    skip.update(normalise(row["text"]) for row in rows)
    if args.base:
        print(f"Keeping {len(rows):,} reviews from {args.base}")

    for path in args.reviews_gz:
        print(f"Sampling {args.per_class:,} reviews per class from {path}")
        reservoirs = sample_reviews(path, args.per_class, args.seed, skip)

        short = [s for s, r in reservoirs.items() if len(r) < args.per_class]
        if short:
            print(f"\nWARNING: ratings {short} had fewer than {args.per_class} eligible "
                  f"reviews in {path}. The corpus is not perfectly balanced.")

        for star, texts in reservoirs.items():
            rows.extend({"rating": star, "text": text} for text in texts)
            skip.update(normalise(text) for text in texts)

    rows = write_reviews(rows, reviews_out, args.seed)

    vocab = corpus_vocabulary(rows)
    print(f"Corpus vocabulary: {len(vocab):,} distinct tokens")
    write_embeddings(args.vec, vocab, embeddings_out)

    print("\nNext:")
    print(f"  wren setup --reviews {reviews_out} --embeddings {embeddings_out}")


if __name__ == "__main__":
    main()
