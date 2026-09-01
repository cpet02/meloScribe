"""Content-addressed cache for expensive pipeline stages.

The old pipeline cached stems by *filename*, which meant changing a model or a
parameter silently returned stale output from the previous run. Here the key is
a hash of the input file's content plus every parameter that affects the
result, so a changed setting is a cache miss by construction and identical work
is never repeated - including across two files that happen to be the same audio
under different names.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Hashing a whole song is I/O-bound and pointless: the first and last chunk
# plus the exact byte length identifies a file for our purposes, and cannot be
# collided by anything short of deliberate effort.
_HEAD_BYTES = 1 << 20  # 1 MiB


def file_fingerprint(path) -> str:
    """A short, stable content fingerprint for an audio file."""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(size).encode())

    with open(path, 'rb') as handle:
        digest.update(handle.read(_HEAD_BYTES))
        if size > 2 * _HEAD_BYTES:
            handle.seek(-_HEAD_BYTES, 2)
            digest.update(handle.read(_HEAD_BYTES))

    return digest.hexdigest()


def params_fingerprint(params: Dict[str, Any]) -> str:
    """Hash a parameter dict, order-independently."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.blake2b(canonical.encode(), digest_size=8).hexdigest()


def cache_key(input_path, stage: str, params: Optional[Dict[str, Any]] = None) -> str:
    """The directory name a given (input, stage, params) triple maps to."""
    return f"{stage}-{file_fingerprint(input_path)}-{params_fingerprint(params or {})}"


@dataclass
class CacheEntry:
    """One cached stage result: a directory plus its metadata sidecar."""
    path: Path
    key: str
    hit: bool

    @property
    def meta_path(self) -> Path:
        return self.path / '_meta.json'

    def read_meta(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return {}

    def write_meta(self, meta: Dict[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str),
                                  encoding='utf-8')

    def complete(self, expected: Iterable[str]) -> bool:
        """Whether every expected output file is present.

        Guards against a half-written entry left behind by a crash or a
        cancelled job being served as though it were a real hit.
        """
        return self.meta_path.exists() and all(
            (self.path / name).exists() for name in expected)


class Cache:
    """A directory of stage results, keyed by content and parameters."""

    def __init__(self, root='.meloscribe_cache'):
        self.root = Path(root)

    def entry(self, input_path, stage: str,
              params: Optional[Dict[str, Any]] = None,
              expected: Optional[Iterable[str]] = None) -> CacheEntry:
        """Look up (but do not create) the entry for a stage."""
        key = cache_key(input_path, stage, params)
        path = self.root / key
        entry = CacheEntry(path=path, key=key, hit=False)
        entry.hit = entry.complete(expected or [])
        return entry

    def clear(self, stage: Optional[str] = None) -> int:
        """Delete cached entries, optionally only those for one stage."""
        if not self.root.exists():
            return 0

        removed = 0
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            if stage is not None and not child.name.startswith(f"{stage}-"):
                continue
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
        return removed

    def size_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(f.stat().st_size for f in self.root.rglob('*') if f.is_file())
