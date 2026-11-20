"""Deprecated: the `wren` package is now `cadello`. This keeps old code working for now.

    from wren import Analyser        ->  from cadello import Analyser
    Analyser(version="1.0")          ->  Analyser("wren-1.0")
"""

import warnings

from cadello.wren.model import WrenAnalyser

warnings.warn("the `wren` package is now `cadello`: use `from cadello import Analyser`. "
              "`wren` will be removed in a later release.", FutureWarning, stacklevel=2)

__all__ = ["Analyser"]
__version__ = "1.5.0"


def Analyser(weights=None, embeddings=None, version="1.5"):  # noqa: N802 (was a class)
    """Wren 1.5, or `version="1.0"`, with the old argument order."""
    return WrenAnalyser(version, weights, embeddings)
