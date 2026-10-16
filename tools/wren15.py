"""Runs the Wren 1.5 command line. The installed `wren` command is Wren 1.0.

    python tools/wren15.py train --workdir wren-work --batches 15000 --seeds 0 1 2
"""

import importlib.util
import os
import sys

# src/wren-1.5 isn't a valid package name, so load it by path as `wren15`.
PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "wren-1.5")
spec = importlib.util.spec_from_file_location(
    "wren15", os.path.join(PACKAGE_DIR, "__init__.py"), submodule_search_locations=[PACKAGE_DIR])
sys.modules["wren15"] = module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from wren15 import cli  # noqa: E402

if __name__ == "__main__":
    if sys.argv[1:2] == ["data"]:  # the corpus builder, train/real_data_setup.py
        from wren15.train import real_data_setup
        sys.argv = [sys.argv[0] + " data"] + sys.argv[2:]
        sys.exit(real_data_setup.main())
    sys.exit(cli.main())
