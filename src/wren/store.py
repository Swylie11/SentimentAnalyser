"""Finds a released Wren model's files, downloading them on first use if they are not already here."""

import hashlib
import os
import sys
import urllib.error
import urllib.request

DEFAULT = "1.5"
RELEASES_URL = "https://github.com/Swylie11/SentimentAnalyser/releases/download/"

# sha256 of each file, from models/wren-<version>/MODEL.md. Checked after every download.
FILES = {
    "1.0": {
        "wren-1.0-weights.npz": "25702430d6b6bb2b079108784eb958771e46230bc5796e5a0f82333284726485",
        "wren-1.0-embeddings.npz": "100c4c929b9d758e3c69f5ebac2378da4a605dde8e5f33fa3e13a53699eb7f3e",
    },
    "1.5": {
        "wren-1.5-weights.npz": "cd0b7addc30cf408880d24bb8bfcf81d81fa97d69f741cbd2334584c72cd8dcd",
        "wren-1.5-embeddings.npz": "f8fa4cacbe4a1dcb9450161ce44dfca1778799027ff8f602c2720593ef3a6e84",
    },
}

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")


def cache_dir():
    return os.path.abspath(os.environ.get("WREN_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "wren"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(version, name, directory):
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, name)
    partial = target + ".part"
    print(f"Downloading {name} ...", file=sys.stderr, flush=True)
    url = f"{RELEASES_URL}v{version}/{name}"
    try:
        urllib.request.urlretrieve(url, partial)
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"could not download {url} ({e}). Put the Wren {version} files "
                           f"in {directory} (or set WREN_HOME to the folder holding them).") from None
    if sha256(partial) != FILES[version][name]:
        os.remove(partial)
        raise RuntimeError(f"{name} failed its sha256 check; deleted it.")
    os.replace(partial, target)
    return target


def ensure(version=DEFAULT):
    """Returns (weights_path, embeddings_path) for a released version. Looks in
    $WREN_HOME/~/.cache/wren, then the repo checkout's models/wren-<version>/, and
    downloads into the cache only if neither has them."""
    if version not in FILES:
        raise LookupError(f"no Wren {version}; released versions are {', '.join(FILES)}")
    checkout = os.path.join(MODELS, f"wren-{version}")
    paths = []
    for name in FILES[version]:
        found = next((p for p in (os.path.join(cache_dir(), name), os.path.join(checkout, name))
                      if os.path.exists(p)), None)
        paths.append(found or _download(version, name, cache_dir()))
    return tuple(paths)
