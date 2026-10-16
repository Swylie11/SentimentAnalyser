"""Checks the Analyser against the shipped Wren 1.5 files (and that 1.0 still loads). Needs no database.

    python tests/test_predict.py        (or: pytest tests/test_predict.py)
"""

import numpy as np

from wren import Analyser

analyser = Analyser()


def test_known_sentences():
    assert analyser.predict("Does exactly what it says, works great") == 5
    assert analyser.predict("Total garbage, crashed and lost my work") == 1


def test_probabilities():
    probabilities = analyser.predict_proba(["great", "awful", "it is fine"])
    assert probabilities.shape == (3, 5)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_input_shapes():
    assert isinstance(analyser.predict("great"), int)
    assert analyser.predict(["great", "awful"]) == [analyser.predict("great"), analyser.predict("awful")]
    assert analyser.predict([]) == []
    assert 1 <= analyser.predict("") <= 5  # nothing to read still gives an answer
    assert 1 <= analyser.predict("great " * 500) <= 5  # longer than the 200 word limit


def test_batching_does_not_change_results():
    texts = ["good value"] * 3 + ["broke after a week"] * 60  # spans two chunks
    assert analyser.predict(texts) == [analyser.predict(t) for t in texts]


def test_wren10_still_loads():
    old = Analyser(version="1.0")
    assert old.predict("Total garbage, crashed and lost my work") == 1
    assert len(old.dense) == 5 and len(analyser.dense) == 1


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("pass ", name)
