"""Provider, provenance, checksum, and cache contracts.

Only normalized :class:`~theme_leadership_os.domain.Observation` objects cross
the provider boundary.  Raw provider response objects are intentionally absent
from these interfaces, making it harder for a personal-research adapter to be
used as a redistribution mechanism by accident.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

from ..domain.models import Observation, Temporal, parse_temporal, temporal_leq


class DataContractError(ValueError):
    """Raised when a source, cache, or provider violates a data contract."""


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 checksum for immutable source bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Short aliases are useful to callers that refer to checksums rather than the
# concrete algorithm.
checksum_bytes = sha256_bytes
checksum_file = sha256_file


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _json_temporal(value: Temporal | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Auditable metadata for a source artifact.

    ``sha256`` is optional only for a manifest drafted before retrieval.  Once
    content is present, :meth:`from_bytes` or :meth:`from_path` populates it,
    and :meth:`verify_bytes` can enforce immutability at read time.
    """

    source_id: str
    uri: str
    sha256: str = ""
    retrieved_at: Temporal | str | None = None
    evidence_available_at: Temporal | str | None = None
    content_type: str = "application/octet-stream"
    row_count: int | None = None
    coverage_start: Temporal | str | None = None
    coverage_end: Temporal | str | None = None
    complete: bool = True
    redistributable: bool = True
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        uri = str(self.uri).strip()
        if not source_id:
            raise ValueError("source_id must not be empty")
        if not uri:
            raise ValueError("uri must not be empty")
        digest = str(self.sha256).strip().lower()
        if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        retrieved_at = parse_temporal(self.retrieved_at) if self.retrieved_at is not None else None
        evidence_available_at = (
            parse_temporal(self.evidence_available_at)
            if self.evidence_available_at is not None
            else None
        )
        coverage_start = (
            parse_temporal(self.coverage_start) if self.coverage_start is not None else None
        )
        coverage_end = parse_temporal(self.coverage_end) if self.coverage_end is not None else None
        if coverage_start is not None and coverage_end is not None:
            if not temporal_leq(coverage_start, coverage_end):
                raise ValueError("coverage_end must not precede coverage_start")
        if self.row_count is not None and (isinstance(self.row_count, bool) or self.row_count < 0):
            raise ValueError("row_count must be a non-negative integer or None")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "retrieved_at", retrieved_at)
        object.__setattr__(self, "evidence_available_at", evidence_available_at)
        object.__setattr__(self, "coverage_start", coverage_start)
        object.__setattr__(self, "coverage_end", coverage_end)
        object.__setattr__(
            self,
            "content_type",
            str(self.content_type).strip() or "application/octet-stream",
        )
        object.__setattr__(self, "notes", str(self.notes).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def checksum(self) -> str:
        return self.sha256

    @property
    def retrieval_at(self) -> Temporal | None:
        return self.retrieved_at

    @property
    def available_at(self) -> Temporal | None:
        return self.evidence_available_at

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        source_id: str,
        uri: str,
        retrieved_at: Temporal | str | None = None,
        **kwargs: Any,
    ) -> SourceManifest:
        return cls(
            source_id=source_id,
            uri=uri,
            sha256=sha256_bytes(payload),
            retrieved_at=retrieved_at or _now_utc(),
            **kwargs,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        source_id: str | None = None,
        uri: str | None = None,
        retrieved_at: Temporal | str | None = None,
        **kwargs: Any,
    ) -> SourceManifest:
        file_path = Path(path)
        return cls(
            source_id=source_id or file_path.name,
            uri=uri or file_path.resolve().as_uri(),
            sha256=sha256_file(file_path),
            retrieved_at=retrieved_at or _now_utc(),
            **kwargs,
        )

    def verify_bytes(self, payload: bytes) -> bool:
        """Verify content against the manifest; missing checksums fail closed."""

        return bool(self.sha256) and sha256_bytes(payload) == self.sha256

    def verify_path(self, path: str | Path) -> bool:
        """Verify a local artifact without exposing its bytes."""

        return bool(self.sha256) and sha256_file(path) == self.sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "uri": self.uri,
            "sha256": self.sha256,
            "retrieved_at": _json_temporal(self.retrieved_at),
            "evidence_available_at": _json_temporal(self.evidence_available_at),
            "content_type": self.content_type,
            "row_count": self.row_count,
            "coverage_start": _json_temporal(self.coverage_start),
            "coverage_end": _json_temporal(self.coverage_end),
            "complete": self.complete,
            "redistributable": self.redistributable,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


CacheValue: TypeAlias = bytes | bytearray | memoryview | str


@runtime_checkable
class Cache(Protocol):
    """Minimal content cache interface shared by source providers."""

    def get(self, key: str) -> bytes | None:
        ...

    def put(self, key: str, value: CacheValue, *, manifest: SourceManifest | None = None) -> None:
        ...

    def contains(self, key: str) -> bool:
        ...

    def delete(self, key: str) -> None:
        ...


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    value: bytes
    manifest: SourceManifest | None = None


class MemoryCache:
    """Deterministic cache implementation suitable for tests and small jobs."""

    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}
        self._manifests: dict[str, SourceManifest] = {}

    def get(self, key: str) -> bytes | None:
        return self._values.get(str(key))

    def put(self, key: str, value: CacheValue, *, manifest: SourceManifest | None = None) -> None:
        if isinstance(value, str):
            payload = value.encode("utf-8")
        elif isinstance(value, (bytearray, memoryview)):
            payload = bytes(value)
        elif isinstance(value, bytes):
            payload = value
        else:
            raise TypeError("cache values must be bytes-like or str")
        cache_key = str(key)
        self._values[cache_key] = payload
        if manifest is not None:
            self._manifests[cache_key] = manifest
        else:
            # Replacing a manifest-backed entry without a manifest must not
            # leave provenance from the previous payload attached to the new
            # bytes.  A stale sidecar would make cache reads unverifiable.
            self._manifests.pop(cache_key, None)

    set = put

    def get_manifest(self, key: str) -> SourceManifest | None:
        return self._manifests.get(str(key))

    def contains(self, key: str) -> bool:
        return str(key) in self._values

    def delete(self, key: str) -> None:
        cache_key = str(key)
        self._values.pop(cache_key, None)
        self._manifests.pop(cache_key, None)

    def clear(self) -> None:
        self._values.clear()
        self._manifests.clear()


class FileCache:
    """Small content-addressed-ish cache with manifest sidecars.

    Keys are encoded into SHA-256 filenames, so callers cannot escape the
    configured directory.  The cache stores bytes only; providers decide what
    is safe to persist.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _stem(self, key: str) -> str:
        return hashlib.sha256(str(key).encode("utf-8")).hexdigest()

    def _payload_path(self, key: str) -> Path:
        return self.root / f"{self._stem(key)}.bin"

    def _manifest_path(self, key: str) -> Path:
        return self.root / f"{self._stem(key)}.manifest.json"

    def get(self, key: str) -> bytes | None:
        path = self._payload_path(key)
        return path.read_bytes() if path.is_file() else None

    def put(self, key: str, value: CacheValue, *, manifest: SourceManifest | None = None) -> None:
        if isinstance(value, str):
            payload = value.encode("utf-8")
        elif isinstance(value, (bytearray, memoryview)):
            payload = bytes(value)
        elif isinstance(value, bytes):
            payload = value
        else:
            raise TypeError("cache values must be bytes-like or str")
        self._payload_path(key).write_bytes(payload)
        if manifest is not None:
            self._manifest_path(key).write_text(
                json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8"
            )
        else:
            # Do not retain a sidecar for a replacement payload whose
            # provenance was not supplied.
            manifest_path = self._manifest_path(key)
            if manifest_path.exists():
                manifest_path.unlink()

    set = put

    def get_manifest(self, key: str) -> SourceManifest | None:
        path = self._manifest_path(key)
        if not path.is_file():
            return None
        try:
            return SourceManifest(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataContractError(f"invalid cache manifest: {path}") from exc

    def contains(self, key: str) -> bool:
        return self._payload_path(key).is_file()

    def delete(self, key: str) -> None:
        for path in (self._payload_path(key), self._manifest_path(key)):
            if path.exists():
                path.unlink()


@runtime_checkable
class PriceProvider(Protocol):
    """Provider-neutral normalized historical-price interface."""

    def get_prices(
        self,
        permanent_id: str,
        start: Temporal | str,
        end: Temporal | str,
        *,
        as_of: Temporal | str | None = None,
    ) -> Sequence[Observation]:
        """Return normalized observations in an inclusive date range."""


class ProviderError(RuntimeError):
    """Base class for provider failures."""


__all__ = [
    "Cache",
    "CacheEntry",
    "CacheValue",
    "DataContractError",
    "FileCache",
    "MemoryCache",
    "PriceProvider",
    "ProviderError",
    "SourceManifest",
    "checksum_bytes",
    "checksum_file",
    "sha256_bytes",
    "sha256_file",
]
