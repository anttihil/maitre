"""Tests for the VRAM-aware LRU cache."""

from __future__ import annotations

import pytest

from maitre.cache import InsufficientVRAMError, ModelCache
from maitre.models import ModelSpec, ModelStatus


@pytest.fixture
def cache(stub_provider, stub_vram):
    return ModelCache(
        providers={"stub": stub_provider},
        vram_manager=stub_vram,
        max_loaded=3,
    )


@pytest.fixture
def small_cache(stub_provider, stub_vram):
    """Cache with tight VRAM (4000 MB usable after 512 reserved from 4512)."""
    stub_vram.total_mb = 4512
    stub_vram.reserved_mb = 512
    return ModelCache(
        providers={"stub": stub_provider},
        vram_manager=stub_vram,
        max_loaded=10,
    )


@pytest.mark.asyncio
async def test_load_and_status(cache, sample_specs):
    spec = sample_specs[0]  # model-a, 2000 MB
    info = await cache.load(spec)
    assert info.status == ModelStatus.LOADED
    assert info.spec.name == "model-a"
    assert cache.allocated_mb == 2000
    assert "model-a" in cache.loaded_names


@pytest.mark.asyncio
async def test_load_same_model_twice_is_noop(cache, sample_specs, stub_provider):
    spec = sample_specs[0]
    await cache.load(spec)
    await cache.load(spec)
    # Provider's load_model should only be called once
    assert stub_provider._load_mock.call_count == 1
    assert cache.allocated_mb == 2000


@pytest.mark.asyncio
async def test_explicit_unload(cache, sample_specs):
    spec = sample_specs[0]
    await cache.load(spec)
    assert cache.allocated_mb == 2000
    await cache.unload("model-a")
    assert cache.allocated_mb == 0
    assert cache.status("model-a") is None


@pytest.mark.asyncio
async def test_lru_eviction_on_max_loaded(cache, sample_specs):
    """When max_loaded=3, loading a 4th model should evict the LRU."""
    a, b, c = sample_specs[0], sample_specs[1], sample_specs[2]
    await cache.load(a)  # oldest
    await cache.load(b)
    await cache.load(c)
    assert len(cache.loaded_names) == 3

    # Touch 'a' so it becomes most-recently-used
    await cache.load(a)

    # Load a new model (reuse spec c's shape but different name)
    d = ModelSpec(name="model-d", provider="stub", vram_mb=1000, tags=["test"])
    await cache.load(d)
    # 'b' should be evicted (it was LRU after 'a' was bumped)
    assert "model-b" not in cache.loaded_names
    assert "model-a" in cache.loaded_names
    assert "model-d" in cache.loaded_names


@pytest.mark.asyncio
async def test_vram_eviction(small_cache, stub_provider):
    """When VRAM is tight, LRU models are evicted to make room."""
    cache = small_cache  # 4000 MB usable
    a = ModelSpec(name="a", provider="stub", vram_mb=2000)
    b = ModelSpec(name="b", provider="stub", vram_mb=2000)
    c = ModelSpec(name="c", provider="stub", vram_mb=2500)

    await cache.load(a)
    await cache.load(b)
    assert cache.allocated_mb == 4000

    # Loading c (2500 MB) should evict a (LRU) to free 2000, still not enough,
    # then evict b to free another 2000
    await cache.load(c)
    assert "a" not in cache.loaded_names
    assert "b" not in cache.loaded_names
    assert "c" in cache.loaded_names
    assert cache.allocated_mb == 2500


@pytest.mark.asyncio
async def test_insufficient_vram_raises(small_cache, sample_specs):
    """A model that's too large even after full eviction raises an error."""
    huge = sample_specs[3]  # 20000 MB
    with pytest.raises(InsufficientVRAMError):
        await small_cache.load(huge)


@pytest.mark.asyncio
async def test_get_handle(cache, sample_specs):
    spec = sample_specs[0]
    await cache.load(spec)
    handle = await cache.get_handle("model-a")
    assert handle == "handle:model-a"


@pytest.mark.asyncio
async def test_get_handle_nonexistent(cache):
    handle = await cache.get_handle("nope")
    assert handle is None


@pytest.mark.asyncio
async def test_unload_all(cache, sample_specs):
    for spec in sample_specs[:3]:
        await cache.load(spec)
    assert len(cache.loaded_names) == 3
    await cache.unload_all()
    assert len(cache.loaded_names) == 0
    assert cache.allocated_mb == 0
