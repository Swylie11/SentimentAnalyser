"""Finds the Wren 1.0 files, downloading them on first use if they are not already here."""

import hashlib
import os
import sys
import urllib.error
import urllib.request

NAME = "wren-1.0"
RELEASE_URL = "https://github.com/Swylie11/SentimentAnalyser/releases/download/v1.0/"

# sha256 of each file, from models/wren-1.0/MODEL.md. Checked after every download.
FILES = {
    f"{NAME}-weights.npz": "25702430d6b6bb2b079108784eb958771e46230bc5796e5a0f82333284726485",
    f"{NAME}-embeddings.npz": "100c4c929b9d758e3c69f5ebac2378da4a605dde8e5f33fa3e13a53699eb7f3e",
}

CHECKOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", NAME)


def cache_dir():
    return os.path.abspath(os.environ.get("WREN_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "wren"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(name, directory):
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, name)
    partial = target + ".part"
    print(f"Downloading {name} ...", file=sys.stderr, flush=True)
    try:
        urllib.request.urlretrieve(RELEASE_URL + name, partial)
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"could not download {RELEASE_URL + name} ({e}). Put the Wren 1.0 files "
                           f"in {directory} (or set WREN_HOME to the folder holding them).") from None
    if sha256(partial) != FILES[name]:
        os.remove(partial)
        raise RuntimeError(f"{name} failed its sha256 check; deleted it.")
    os.replace(partial, target)
    return target


def ensure():
    """Returns (weights_path, embeddings_path). Looks in $WREN_HOME/~/.cache/wren, then the
    repo checkout's models/wren-1.0/, and downloads into the cache only if neither has them."""
    paths = []
    for name in FILES:
        found = next((p for p in (os.path.join(cache_dir(), name), os.path.join(CHECKOUT, name))
                      if os.path.exists(p)), None)
        paths.append(found or _download(name, cache_dir()))
    return tuple(paths)
