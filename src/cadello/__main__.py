"""`python -m cadello ...` runs the `cadello` command."""

import sys

from .cli import main

sys.exit(main())
