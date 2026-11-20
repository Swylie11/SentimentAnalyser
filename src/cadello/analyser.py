"""The interface every Cadello model shares.

    Analyser()                    the default model (Wren 1.5), downloaded on first use
    Analyser("wren-1.0")          any released model by name (see `cadello models`)
    Analyser(version="1.0")       the older keyword, still accepted

`Analyser(...)` returns an instance of the right model class (for example `WrenAnalyser`),
which is still an `Analyser`, the same way `pathlib.Path(...)` returns a `PosixPath`.
A new model class subclasses this and implements `predict_proba`.
"""

import numpy as np

SENTIMENT_LABELS = {1: "Very negative", 2: "Negative", 3: "Neutral",
                    4: "Positive", 5: "Very positive"}


class Analyser:
    """A trained Cadello model that rates text from 1 to 5 stars."""

    release = None  # the registry.Release this instance was loaded from, if any

    def __new__(cls, model=None, *args, **kwargs):
        if cls is Analyser:
            from .registry import resolve
            cls = resolve(model if model is not None else kwargs.get("version")).analyser_class()
        return super().__new__(cls)

    @property
    def name(self):
        return self.release.name if self.release else "custom"

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"

    def predict_proba(self, texts):
        """Probability of each star rating: array of shape (len(texts), 5)."""
        raise NotImplementedError

    def predict(self, texts):
        """Star rating 1 to 5. A str gives an int, a list of str gives a list of int."""
        stars = (np.argmax(self.predict_proba(texts), axis=1) + 1).tolist()
        return stars[0] if isinstance(texts, str) else stars

    @staticmethod
    def label(stars):
        return SENTIMENT_LABELS[int(stars)]
