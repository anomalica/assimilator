"""Where the assimilator keeps its state on this machine.

Defaults to ~/.local/share/assimilator. ASSIMILATOR_DATA_DIR moves the whole
directory - the container mounts it at /data, where HOME is not the host's -
and a specific file's own variable (ASSIMILATOR_DB, ANOMALICA_MERGE_CANDIDATES,
...) still wins for that file. The first scheduled merge run found none of
its memos because every default was under HOME and only some files had a
variable of their own; it re-scored 171,000 pairs into a directory nothing
would read.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    return Path(
        os.environ.get(
            "ASSIMILATOR_DATA_DIR",
            str(Path.home() / ".local" / "share" / "assimilator"),
        )
    )


def data_path(name: str, env: str | None = None) -> Path:
    """The file `name` under the data directory, unless `env` names it."""
    if env and os.environ.get(env):
        return Path(os.environ[env])
    return data_dir() / name
