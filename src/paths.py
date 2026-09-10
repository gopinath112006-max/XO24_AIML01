"""Single source of truth for repo-relative paths.

The package lives in ``src/``; data and secrets live at the repository root
(``data/`` and ``.credentials.py``). Resolving them relative to ``__file__``
keeps every writer/reader consistent regardless of how it is invoked
(``python -m src.profiler`` from the root, ``python scripts/...``, pytest).
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CRED_FILE = os.path.join(ROOT, ".credentials.py")
