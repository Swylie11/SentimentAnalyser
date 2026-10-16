"""The `wren` command."""

import argparse
import json
import os
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def predict(args):
    from . import Analyser
    texts = args.text
    if texts == ["-"]:
        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    analyser = Analyser()
    probabilities = analyser.predict_proba(texts)
    for text, row in zip(texts, probabilities):
        stars = int(row.argmax()) + 1
        if args.json:
            print(json.dumps({"text": text, "stars": stars, "label": analyser.label(stars),
                              "probabilities": [round(float(p), 4) for p in row]}))
        elif args.proba:
            print(f"{stars}\t" + " ".join(f"{p:.3f}" for p in row))
        else:
            print(f"{stars}\t{analyser.label(stars)}")


def evaluate(args):
    import numpy as np
    from . import Analyser
    with open(args.data, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    labels = np.array([int(r["rating"]) for r in records])
    predictions = np.array(Analyser().predict([r["text"] for r in records]))
    print(f"{len(labels)} reviews")
    print(f"  Exact accuracy  : {100 * np.mean(predictions == labels):.1f}%")
    print(f"  Within one star : {100 * np.mean(abs(predictions - labels) <= 1):.1f}%")


def fetch(args):
    from . import store
    for path in store.ensure():
        print(path)


def setup(args):
    from .train import comms, init_databases
    if args.workdir:
        comms.set_db_dir(args.workdir)
    init_databases.setup(args.synthetic, args.seed, args.embeddings, args.reviews)


def train(args):
    from .train import comms, loop
    if args.workdir:
        comms.set_db_dir(args.workdir)
    loop.train(args.batch_size, args.batches, args.yes, args.seeds, args.log,
               None if args.no_eval else args.eval_data, args.history)


def watch(args):
    from .train import comms, dashboard
    dashboard.watch(args.workdir or comms.DB_DIR, args.interval)


def demo(args):
    from .train import dashboard
    dashboard.demo(args.interval)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wren", description="Five-class star-rating sentiment analyser.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="rate review text from 1 to 5 stars")
    p.add_argument("text", nargs="+", help='review text, or "-" to read one review per line from stdin')
    p.add_argument("--proba", action="store_true", help="also print the probability of each star")
    p.add_argument("--json", action="store_true", help="one JSON object per review")
    p.set_defaults(run=predict)

    p = sub.add_parser("eval", help="accuracy on a .jsonl file of {rating, text} reviews")
    p.add_argument("--data", required=True)
    p.set_defaults(run=evaluate)

    p = sub.add_parser("fetch", help="download the Wren 1.0 files now instead of on first use")
    p.set_defaults(run=fetch)

    p = sub.add_parser("setup", help="build the training databases (training only)")
    p.add_argument("--workdir", help="database directory (default $WREN_WORKDIR or ./wren-work)")
    p.add_argument("--synthetic", type=int, metavar="N", help="N synthetic reviews instead of real data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embeddings", metavar="JSONL", help="from `python -m wren.train.real_data_setup`")
    p.add_argument("--reviews", metavar="JSONL", help="from `python -m wren.train.real_data_setup`")
    p.set_defaults(run=setup)

    p = sub.add_parser("train", help="train a new model, replacing the one in --workdir")
    p.add_argument("--workdir", help="database directory (default $WREN_WORKDIR or ./wren-work)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, required=True)
    p.add_argument("--yes", action="store_true", help="skip the overwrite confirmation")
    p.add_argument("--seeds", type=int, nargs="+", metavar="S",
                   help="one run per seed; only the best on validation is saved")
    p.add_argument("--log", metavar="FILE", help="also append every message to FILE")
    p.add_argument("--eval-data", metavar="JSONL", default=os.path.join(REPO, "Data", "eval_test_multi.jsonl"),
                   help="fresh test set scored once training ends (default Data/eval_test_multi.jsonl, "
                        "the five training categories)")
    p.add_argument("--history", metavar="FILE",
                   default=os.path.join(REPO, "evaluation", "wren15_history.jsonl"),
                   help="one line per training run is appended here "
                        "(default evaluation/wren15_history.jsonl)")
    p.add_argument("--no-eval", action="store_true", help="skip the test set evaluation")
    p.set_defaults(run=train)

    p = sub.add_parser("watch", help="live training stats; run it in a second window")
    p.add_argument("--workdir", help="database directory (default $WREN_WORKDIR or ./wren-work)")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between refreshes")
    p.set_defaults(run=watch)

    p = sub.add_parser("demo", help="the watch panel on a fake run; trains nothing, touches no workdir")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between refreshes")
    p.set_defaults(run=demo)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (OSError, RuntimeError, LookupError) as e:
        print(f"wren: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
