"""Small immutable-value JSON cache with stable SHA-256 keys and atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional


def atomic_write_text(path: Path, content: str) -> None:
    """Replace one text file atomically so incomplete cache/artifact files are never read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def atomic_write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    """Serialize JSON deterministically and commit it atomically."""
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


class JsonFileCache:
    """Cache-aside storage scoped by namespace and caller-provided deterministic key data."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @staticmethod
    def key_for(value: Mapping[str, Any] | list[Any]) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, namespace: str, key: str) -> Optional[dict[str, Any]]:
        path = self._path(namespace, key)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, namespace: str, key: str, value: Mapping[str, Any]) -> None:
        atomic_write_json(self._path(namespace, key), dict(value))

    def _path(self, namespace: str, key: str) -> Path:
        if not namespace.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Cache namespace must contain only letters, digits, hyphens, or underscores.")
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Cache key must be a SHA-256 hexadecimal digest.")
        return self._root / namespace / f"{key}.json"
