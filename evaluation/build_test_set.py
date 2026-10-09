"""Builds a fresh, balanced test set the model has never seen.

The training corpus (sentiment/test_data.db) was split into train and validation,
and the validation set was also used to pick the best checkpoint. This draws new
reviews from the Amazon dump and drops any whose text already appears in that
database, so the result is untouched by training and by checkpoint selection.

It reuses RealDataSetup.sample_reviews, so the star balance, the word-count window
and the HTML stripping match the training corpus exactly. It never writes to
test_data.db.

Usage:
    python evaluation/build_test_set.py --reviews-gz Data/Software.jsonl.gz
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalise(text):
    """Lowercased word tokens joined by single spaces, so formatting differences
    (case, punctuation, whitespace) do not hide a duplicate."""
    return " ".join(re.findall(r"\w+", text.lower()))


def known_texts(db_path):
    """Normalised text of every review already in the training database."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return {normalise(row[0]) for row in conn.execute("SELECT text FROM test_dataset4")}


def select_fresh(reservoirs, known, per_class, seed):
    """Takes up to per_class reviews per star that are not in `known` and not
    duplicates of each other. The pool is shuffled first (a reservoir's slot order
    is not uniformly random), so what is kept is still a uniform sample.

    Returns ({star: [text, ...]}, {"in_training": n, "repeated": n})."""
    rng = random.Random(seed)
    seen = set()
    dropped = {"in_training": 0, "repeated": 0}
    chosen = {}

    for star, texts in sorted(reservoirs.items()):
        pool = list(texts)
        rng.shuffle(pool)
        chosen[star] = []
        for text in pool:
            if len(chosen[star]) == per_class:
                break
            key = normalise(text)
            if key in known:
                dropped["in_training"] += 1
            elif key in seen:
                dropped["repeated"] += 1
            else:
                seen.add(key)
                chosen[star].append(text)

    return chosen, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews-gz", default=os.path.join(REPO, "Data", "Software.jsonl.gz"))
    parser.add_argument("--db", default=os.path.join(REPO, "sentiment", "test_data.db"),
                        help="Training database whose reviews must be excluded (read only)")
    parser.add_argument("--out", default=os.path.join(REPO, "Data", "eval_test.jsonl"))
    parser.add_argument("--per-class", type=int, default=400,
                        help="Reviews per star rating (default 400, so 2,000 in total)")
    parser.add_argument("--oversample", type=int, default=4,
                        help="Candidates drawn per kept review, to survive the exclusions")
    parser.add_argument("--seed", type=int, default=1,
                        help="Sampling seed (the exclusion, not the seed, guarantees freshness)")
    args = parser.parse_args()

    # Imported here, not at module level, so evaluate.py can import normalise()
    # without pulling the real sentiment/ modules in ahead of its own --model-dir.
    sys.path.insert(0, os.path.join(REPO, "sentiment"))
    import RealDataSetup as rds

    known = known_texts(args.db)
    print(f"{len(known):,} distinct training reviews will be excluded.")

    print(f"Sampling candidates from {args.reviews_gz}")
    reservoirs = rds.sample_reviews(args.reviews_gz, args.per_class * args.oversample, args.seed)
    chosen, dropped = select_fresh(reservoirs, known, args.per_class, args.seed)

    short = [star for star, texts in chosen.items() if len(texts) < args.per_class]
    if short:
        print(f"WARNING: ratings {short} ended with fewer than {args.per_class} reviews; "
              f"raise --oversample.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = rds.write_reviews(chosen, args.out, args.seed)

    manifest = {
        "seed": args.seed,
        "source": os.path.basename(args.reviews_gz),
        "per_class_requested": args.per_class,
        "per_class_kept": {str(star): len(texts) for star, texts in chosen.items()},
        "total": len(rows),
        "candidates_dropped_in_training": dropped["in_training"],
        "candidates_dropped_repeated": dropped["repeated"],
    }
    manifest_path = os.path.splitext(args.out)[0] + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
