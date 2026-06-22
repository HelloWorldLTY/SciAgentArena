"""Minimal .env loader (no third-party dependency).

Faithful to the loaders used across the source branches: reads ``KEY=VALUE``
lines, honors a leading ``export ``, strips matching surrounding quotes, and
uses ``os.environ.setdefault`` so it never overrides a variable already set in
the environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path) -> bool:
    """Load ``path`` into ``os.environ`` (setdefault). Return True if read."""
    p = Path(path)
    if not p.exists():
        return False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
    return True
