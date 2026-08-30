from datetime import date

import pytest

from theme_leadership_os.data import (
    CSVPriceProvider,
    MemoryCache,
    PersonalResearchDisabled,
    PriceProvider,
    SourceManifest,
    YFinancePriceProvider,
    sha256_bytes,
)


def test_manifest_checksum_and_memory_cache() -> None:
    payload = b"permanent_id,date,close\nticker:aaa,2020-01-01,10\n"
    manifest = SourceManifest.from_bytes(
        payload,
        source_id="fixture:prices",
        uri="fixture://prices",
        evidence_available_at=date(2020, 1, 1),
    )
    assert manifest.sha256 == sha256_bytes(payload)
    assert manifest.verify_bytes(payload)
    assert not manifest.verify_bytes(payload + b"\n")
    cache = MemoryCache()
    cache.put("prices", payload, manifest=manifest)
    assert cache.get("prices") == payload
    assert cache.get_manifest("prices") == manifest
    assert cache.contains("prices")
    cache.delete("prices")
    assert not cache.contains("prices")


def test_cache_replacement_without_manifest_drops_old_provenance(tmp_path) -> None:
    payload = b"one"
    manifest = SourceManifest.from_bytes(
        payload, source_id="fixture:one", uri="fixture://one"
    )
    memory = MemoryCache()
    memory.put("key", payload, manifest=manifest)
    memory.put("key", b"two")
    assert memory.get_manifest("key") is None

    from theme_leadership_os.data import FileCache

    file_cache = FileCache(tmp_path / "cache")
    file_cache.put("key", payload, manifest=manifest)
    file_cache.put("key", b"two")
    assert file_cache.get_manifest("key") is None


def test_csv_provider_is_provider_neutral_and_pit_safe(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "permanent_id,symbol,date,evidence_available_at,close,source\n"
        "ticker:aaa,AAA,2020-01-01,2020-01-01,10,fixture\n"
        "ticker:aaa,AAA,2020-01-02,2020-02-01,11,fixture\n"
        "ticker:aaa,AAA,2020-01-03,2020-01-03,12,fixture\n",
        encoding="utf-8",
    )
    provider = CSVPriceProvider(path)
    assert isinstance(provider, PriceProvider)
    early = provider.get_prices("AAA", date(2020, 1, 1), date(2020, 1, 3), as_of=date(2020, 1, 15))
    assert [row.value for row in early] == [10.0, 12.0]
    assert [row.observed_at for row in early] == [date(2020, 1, 1), date(2020, 1, 3)]
    late = provider.get_observations(
        "ticker:aaa", date(2020, 1, 1), date(2020, 1, 3), as_of=date(2020, 2, 1)
    )
    assert [row.value for row in late] == [10.0, 11.0, 12.0]
    assert provider.manifest.row_count == 3
    assert provider.manifest.verify_bytes(path.read_bytes())


def test_personal_research_adapter_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THEME_LEADERSHIP_ENABLE_YFINANCE", raising=False)
    with pytest.raises(PersonalResearchDisabled):
        YFinancePriceProvider()
