"""Every released Cadello model, by name.

Model classes are named after birds by size (Wren under 1M parameters, Kestrel 1M to 10M).
To add a release, add a `Release` below. To add a new model class, put its code in its own
subpackage (as `cadello/wren/` is) with an `Analyser` subclass, and point `analyser` at it.
The CLI, `Analyser(...)` and `store.ensure` pick it up from here; nothing else changes.
"""

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class File:
    name: str
    sha256: str
    tag: str  # the GitHub release that holds it; a model can reuse another release's file


@dataclass(frozen=True)
class Release:
    name: str      # what users type: "wren-1.5"
    family: str    # the model class: "wren"
    version: str   # "1.5"
    summary: str
    files: tuple   # File entries, in the order the model's Analyser takes them
    analyser: str  # "module:Class", imported only when the model is loaded

    def analyser_class(self):
        module, _, cls = self.analyser.partition(":")
        return getattr(importlib.import_module(module), cls)


RELEASES = {r.name: r for r in [
    Release("wren-1.5", "wren", "1.5",
            "Text CNN, 439k params, five product categories. 52.5% exact, 90.9% within one",
            (File("wren-1.5-weights.npz",
                  "cd0b7addc30cf408880d24bb8bfcf81d81fa97d69f741cbd2334584c72cd8dcd", "v1.5"),
             File("wren-1.5-embeddings.npz",
                  "f8fa4cacbe4a1dcb9450161ce44dfca1778799027ff8f602c2720593ef3a6e84", "v1.5")),
            "cadello.wren.model:WrenAnalyser"),
    Release("wren-1.0", "wren", "1.0",
            "Text CNN, 481k params, Software reviews only. 49.2% exact, 87.9% within one",
            (File("wren-1.0-weights.npz",
                  "25702430d6b6bb2b079108784eb958771e46230bc5796e5a0f82333284726485", "v1.0"),
             File("wren-1.0-embeddings.npz",
                  "100c4c929b9d758e3c69f5ebac2378da4a605dde8e5f33fa3e13a53699eb7f3e", "v1.0")),
            "cadello.wren.model:WrenAnalyser"),
]}

DEFAULT_MODEL = "wren-1.5"

# Other names accepted for a release. The bare versions are what `wren --model` and
# `Analyser(version=...)` took before the rename; Kestrel names are never bare numbers,
# so they can't collide.
ALIASES = {"wren": "wren-1.5", "1.5": "wren-1.5", "1.0": "wren-1.0"}


def resolve(name=None):
    """The Release for a model name or alias; the default model when name is None."""
    key = DEFAULT_MODEL if name is None else str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in RELEASES:
        raise LookupError(f"no model called {name!r}; released models are {', '.join(RELEASES)}")
    return RELEASES[key]
