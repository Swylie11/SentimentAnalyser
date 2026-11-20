"""Finds a released model's files, downloading them on first use if they are not already here."""

import hashlib
import os
import sys
import urllib.error
import urllib.request

from .registry import resolve

# The repository was called SentimentAnalyser until 2026-11-20; GitHub redirects the old URL.
RELEASES_URL = "https://github.com/Swylie11/cadello/releases/download/"

# releases/<family>/<version>/ in a repo checkout, where exported files are written.
CHECKOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "releases")


def cache_dir():
    """Where downloads go: $CADELLO_HOME, or ~/.cache/cadello."""
    return os.path.abspath(os.environ.get("CADELLO_HOME")
                           or os.path.join(os.path.expanduser("~"), ".cache", "cadello"))


def search_dirs(release):
    """Folders checked for a release's files, in order. The old `wren` cache locations are
    read too, so files downloaded before the rename are not downloaded again."""
    dirs = [cache_dir()]
    if os.environ.get("WREN_HOME"):
        dirs.append(os.environ["WREN_HOME"])
    dirs.append(os.path.join(os.path.expanduser("~"), ".cache", "wren"))
    dirs.append(os.path.join(CHECKOUT, release.family, release.version))
    return [os.path.abspath(d) for d in dirs]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def url(file):
    return f"{RELEASES_URL}{file.tag}/{file.name}"


def _download(release, file, directory):
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, file.name)
    partial = target + ".part"
    print(f"Downloading {file.name} ...", file=sys.stderr, flush=True)
    try:
        urllib.request.urlretrieve(url(file), partial)
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"could not download {url(file)} ({e}). Put the {release.name} files "
                           f"in {directory} (or set CADELLO_HOME to the folder holding them).") from None
    if sha256(partial) != file.sha256:
        os.remove(partial)
        raise RuntimeError(f"{file.name} failed its sha256 check; deleted it.")
    os.replace(partial, target)
    return target


def find(release):
    """The local path of each of a release's files, or None where it isn't here yet."""
    dirs = search_dirs(release)
    return [next((p for p in (os.path.join(d, f.name) for d in dirs) if os.path.exists(p)), None)
            for f in release.files]


def ensure(model=None):
    """Paths to a release's files, in the order its Analyser takes them, downloading
    into cache_dir() any that aren't here. `model` is a name, an alias or a Release."""
    release = resolve(model) if model is None or isinstance(model, str) else model
    return tuple(path or _download(release, f, cache_dir())
                 for f, path in zip(release.files, find(release)))
