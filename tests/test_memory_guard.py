import pytest
from tiny_recursive_gemma.memory_guard import (
    init_memory_guardrails,
    get_available_system_memory_gb,
    flush_memory,
    get_metal_memory_stats,
    guarded_memory_scope
)

def test_get_available_system_memory():
    avail_gb = get_available_system_memory_gb()
    assert isinstance(avail_gb, float)
    assert avail_gb > 0.0

def test_init_memory_guardrails():
    # Enforces 3.5 GB ceiling and 0.5 GB cache limit
    init_memory_guardrails(max_memory_gb=3.5, max_cache_gb=0.5)
    stats = get_metal_memory_stats()
    assert "active_mb" in stats
    assert "peak_mb" in stats

def test_flush_memory():
    flush_memory()

def test_guarded_memory_scope():
    with guarded_memory_scope(max_memory_gb=3.5, max_cache_gb=0.5) as stats:
        assert isinstance(stats, dict)
        assert "system_available_gb" in stats
