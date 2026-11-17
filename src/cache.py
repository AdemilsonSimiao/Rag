import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def file_digest(path: Path) -> str:
    """Short hash of a file's content, or "" if it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def index_signature(index_path: Path) -> str:
    """Identify one build of the index by its size and modification time."""
    try:
        stat = index_path.stat()
    except OSError:
        return "no-index"
    return f"{stat.st_size}-{stat.st_mtime_ns}"


class JsonCache:
    """A small key-value cache saved as a JSON file.

    The cache remembers a signature. If the saved signature differs from
    the current one (the index was rebuilt, the code changed), the old
    entries are ignored, so the cache never returns stale results. Any
    problem with the file only means an empty cache, never an error.
    """

    def __init__(
        self, path: Path, signature: str, max_entries: int = 10000
    ) -> None:
        """Load the cache file if it exists and its signature matches."""
        self.path = path
        self.signature = signature
        self.max_entries = max_entries
        self.entries: Dict[str, Any] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(data, dict)
                and data.get("signature") == signature
                and isinstance(data.get("entries"), dict)
            ):
                self.entries = data["entries"]
        except (OSError, ValueError):
            pass

    @staticmethod
    def make_key(*parts: str) -> str:
        """Build a cache key from the parts that define a request."""
        joined = "\x00".join(parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or None if there is none."""
        return self.entries.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store a value, dropping the oldest entries above the limit."""
        self.entries.pop(key, None)
        self.entries[key] = value
        while len(self.entries) > self.max_entries:
            del self.entries[next(iter(self.entries))]

    def save(self) -> None:
        """Write the cache to disk; a failure is ignored (cache only)."""
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(
                    {"signature": self.signature, "entries": self.entries}
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError:
            pass
