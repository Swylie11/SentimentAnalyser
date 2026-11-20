"""Model names, file lookup and the command line, for every released model. Uses the shipped
Wren files (downloaded on first use, like test_predict.py). Needs no database:

    python tests/test_registry.py        (or: pytest tests/test_registry.py)
"""

import json
import os
import subprocess
import sys
import tempfile
import warnings

import cadello
from cadello import Analyser, registry, store
from cadello.wren import WrenAnalyser


def cli(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "cadello", *args], input=stdin,
                          capture_output=True, text=True)


def test_resolve():
    assert registry.resolve().name == registry.DEFAULT_MODEL == "wren-1.5"
    for alias, name in registry.ALIASES.items():
        assert registry.resolve(alias).name == name
    assert registry.resolve(" Wren-1.0 ").name == "wren-1.0"
    try:
        registry.resolve("kestrel-s")
    except LookupError as e:
        assert "wren-1.5" in str(e)  # the message lists what exists
    else:
        raise AssertionError("an unknown model resolved")


def test_every_release_is_well_formed():
    for name, release in registry.RELEASES.items():
        assert release.name == name and name.startswith(release.family + "-")
        assert issubclass(release.analyser_class(), Analyser)
        for f in release.files:
            assert len(f.sha256) == 64 and f.tag
            assert store.url(f).endswith(f"/{f.tag}/{f.name}")


def test_analyser_returns_the_model_class():
    a = Analyser("wren-1.0")
    assert type(a) is WrenAnalyser and isinstance(a, Analyser)
    assert repr(a) == "WrenAnalyser('wren-1.0')"


def test_old_cache_is_still_found():
    """Files downloaded by `wren` (to $WREN_HOME or ~/.cache/wren) are used, not downloaded again."""
    release = registry.resolve("wren-1.5")
    saved = {k: os.environ.get(k) for k in ("CADELLO_HOME", "WREN_HOME")}
    with tempfile.TemporaryDirectory() as new, tempfile.TemporaryDirectory() as old:
        try:
            os.environ["CADELLO_HOME"], os.environ["WREN_HOME"] = new, old
            for f in release.files:
                open(os.path.join(old, f.name), "w").close()
            assert store.find(release) == [os.path.join(os.path.abspath(old), f.name) for f in release.files]
            open(os.path.join(new, release.files[0].name), "w").close()  # the new cache wins
            assert store.find(release)[0] == os.path.join(os.path.abspath(new), release.files[0].name)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_cli_json_contract():
    """Kestrel-S spec 8.4: exactly {text, stars, label, probabilities}, for every model."""
    for name in registry.RELEASES:
        run = cli("predict", "--model", name, "--json", "-", stdin="great\nawful\n")
        assert run.returncode == 0, run.stderr
        rows = [json.loads(line) for line in run.stdout.splitlines()]
        assert len(rows) == 2
        for row in rows:
            assert set(row) == {"text", "stars", "label", "probabilities"}
            assert 1 <= row["stars"] <= 5 and len(row["probabilities"]) == 5
            assert abs(sum(row["probabilities"]) - 1) < 1e-3


def test_cli_rejects_unknown_model():
    run = cli("predict", "--model", "heron-9", "hello")
    assert run.returncode == 2 and "no model called" in run.stderr


def test_cli_lists_models():
    run = cli("models")
    assert run.returncode == 0
    assert [line.split()[0] for line in run.stdout.splitlines()] == list(registry.RELEASES)


def test_old_wren_package_still_works():
    sys.modules.pop("wren", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import wren
    assert any(issubclass(w.category, FutureWarning) for w in caught)
    assert wren.Analyser(version="1.0").name == "wren-1.0"
    assert wren.Analyser().predict("great") == cadello.Analyser().predict("great")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("pass ", name)
