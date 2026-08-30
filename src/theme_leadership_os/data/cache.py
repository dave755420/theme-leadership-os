"""Compatibility import surface for cache contracts and implementations."""

from .contracts import Cache, CacheEntry, CacheValue, FileCache, MemoryCache

__all__ = ["Cache", "CacheEntry", "CacheValue", "FileCache", "MemoryCache"]
