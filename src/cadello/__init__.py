"""Cadello: a family of star-rating sentiment models written from scratch in NumPy.

    from cadello import Analyser
    Analyser().predict("Does exactly what it says")      # 5, using the default model
    Analyser("wren-1.0").predict(["great", "awful"])     # [5, 1]
"""

from .analyser import SENTIMENT_LABELS, Analyser
from .registry import DEFAULT_MODEL, RELEASES

__all__ = ["Analyser", "DEFAULT_MODEL", "RELEASES", "SENTIMENT_LABELS"]
__version__ = "0.1.0"  # the package; model versions are separate (wren-1.5, wren-1.0, ...)
