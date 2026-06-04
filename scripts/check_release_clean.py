from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CACHE_NAMES = {".pytest_cache", "__pycache__"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".pt", ".npz"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".csv", ".txt", ".cff", ".gitignore"}
SECRET_RE = re.compile(r"(?i)(api[_-]?key|password|secret|bearer\s+[a-z0-9._-]+|access[_-]?token)")
LOCAL_PATH_RE = re.compile(r"(?i)([A-Z]:\\Users\\|[A-Z]:\\.*\\Desktop\\|Administrator)")
MAX_TRACKABLE_BYTES = 1_000_000


def main() -> int:
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts:
            continue
        if path.is_dir():
            if path.name in CACHE_NAMES:
                offenders.append(f"cache directory: {rel}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(f"forbidden artifact: {rel}")
        if path.stat().st_size > MAX_TRACKABLE_BYTES:
            offenders.append(f"large file: {rel} ({path.stat().st_size} bytes)")
        if path.name == "check_release_clean.py":
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8", errors="ignore")
            if LOCAL_PATH_RE.search(text):
                offenders.append(f"local absolute path or username: {rel}")
            if SECRET_RE.search(text):
                offenders.append(f"secret-like string: {rel}")

    if offenders:
        print("Release cleanliness check failed:")
        for offender in offenders:
            print(f"- {offender}")
        return 1
    print("Release cleanliness check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
