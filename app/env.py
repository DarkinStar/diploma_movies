"""Minimal .env loader (no third-party dependency).

Reads KEY=VALUE lines from the project-level .env file into os.environ without
overriding variables that are already set in the environment.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = PROJECT_DIR / ".env") -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
