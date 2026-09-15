"""
Apple Silicon Memory Guardrails for Tiny Recursive Gemma (MLX).
Provides hard allocation ceilings, automatic Metal cache management,
host memory inspection, and safe evaluation context managers.
"""

import gc
import os
import re
import subprocess
import resource
from contextlib import contextmanager
from typing import Generator, Optional, Tuple
import mlx.core as mx

# Default safe limits for Apple Silicon (M-series with 16GB unified memory)
DEFAULT_MAX_METAL_MEMORY_GB: float = 3.5
DEFAULT_MAX_METAL_CACHE_GB: float = 0.5
MIN_REQUIRED_SYSTEM_FREE_GB: float = 1.5

def get_available_system_memory_gb() -> float:
    """
    Returns estimated available system memory in GB on macOS via vm_stat.
    Falls back to a conservative estimate if vm_stat is unavailable.
    """
    try:
        vm_stat_out = subprocess.check_output(["vm_stat"], timeout=2).decode("utf-8")
        page_size_match = re.search(r"page size of (\d+) bytes", vm_stat_out)
        page_size = int(page_size_match.group(1)) if page_size_match else 16384

        free_match = re.search(r"Pages free:\s+(\d+)\.", vm_stat_out)
        inactive_match = re.search(r"Pages inactive:\s+(\d+)\.", vm_stat_out)
        speculative_match = re.search(r"Pages speculative:\s+(\d+)\.", vm_stat_out)

        free_pages = int(free_match.group(1)) if free_match else 0
        inactive_pages = int(inactive_match.group(1)) if inactive_match else 0
        speculative_pages = int(speculative_match.group(1)) if speculative_match else 0

        avail_bytes = (free_pages + inactive_pages + speculative_pages) * page_size
        return round(avail_bytes / (1024 ** 3), 2)
    except Exception:
        # Fallback using standard resource module
        return 4.0

def init_memory_guardrails(
    max_memory_gb: float = DEFAULT_MAX_METAL_MEMORY_GB,
    max_cache_gb: float = DEFAULT_MAX_METAL_CACHE_GB,
    enforce_cache_clearing: bool = True
) -> None:
    """
    Initializes Apple Silicon Metal allocation guardrails.
    - Sets hard memory ceiling so MLX raises MemoryError instead of crashing the kernel.
    - Sets cache ceiling so freed tensor buffers return to macOS immediately.
    """
    if hasattr(mx, "metal") and mx.metal.is_available():
        try:
            # 1. Set hard allocation limit (bytes)
            max_bytes = int(max_memory_gb * (1024 ** 3))
            mx.metal.set_memory_limit(max_bytes)

            # 2. Set cache limit (bytes)
            cache_bytes = int(max_cache_gb * (1024 ** 3))
            mx.metal.set_cache_limit(cache_bytes)

            if enforce_cache_clearing:
                mx.metal.clear_cache()
                gc.collect()
        except Exception as e:
            # Non-fatal if specific Metal API call is unsupported
            print(f"[MemoryGuard] Warning setting Metal limits: {e}")

def flush_memory() -> None:
    """
    Forces an explicit cleanup of unused Metal GPU buffers and Python GC.
    """
    if hasattr(mx, "clear_cache"):
        try:
            mx.clear_cache()
        except Exception:
            pass
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        try:
            mx.metal.clear_cache()
        except Exception:
            pass
    gc.collect()

def get_metal_memory_stats() -> dict:
    """
    Returns active, cache, and peak Metal memory in megabytes.
    """
    try:
        get_active = getattr(mx, "get_active_memory", getattr(getattr(mx, "metal", None), "get_active_memory", None))
        get_cache = getattr(mx, "get_cache_memory", getattr(getattr(mx, "metal", None), "get_cache_memory", None))
        get_peak = getattr(mx, "get_peak_memory", getattr(getattr(mx, "metal", None), "get_peak_memory", None))

        active_mb = round(get_active() / (1024 ** 2), 2) if get_active else 0.0
        cache_mb = round(get_cache() / (1024 ** 2), 2) if get_cache else 0.0
        peak_mb = round(get_peak() / (1024 ** 2), 2) if get_peak else 0.0
        return {
            "active_mb": active_mb,
            "cache_mb": cache_mb,
            "peak_mb": peak_mb,
            "system_available_gb": get_available_system_memory_gb()
        }
    except Exception:
        return {"active_mb": 0.0, "cache_mb": 0.0, "peak_mb": 0.0, "system_available_gb": 0.0}

@contextmanager
def guarded_memory_scope(
    max_memory_gb: float = DEFAULT_MAX_METAL_MEMORY_GB,
    max_cache_gb: float = DEFAULT_MAX_METAL_CACHE_GB,
    min_system_free_gb: float = MIN_REQUIRED_SYSTEM_FREE_GB
) -> Generator[dict, None, None]:
    """
    Context manager that verifies memory headroom before entering,
    applies Metal memory limits, and guarantees cleanup on exit.
    """
    init_memory_guardrails(max_memory_gb, max_cache_gb)
    
    avail_gb = get_available_system_memory_gb()
    if avail_gb < min_system_free_gb:
        flush_memory()
        avail_gb = get_available_system_memory_gb()
        if avail_gb < min_system_free_gb:
            raise MemoryError(
                f"[MemoryGuard] Insufficient available host RAM: {avail_gb} GB free, "
                f"minimum required is {min_system_free_gb} GB. Aborting to protect system stability."
            )

    try:
        yield get_metal_memory_stats()
    finally:
        flush_memory()

def verify_recursion_guardrails(
    recurrent_layer_count: int = 2, 
    iterations_T: int = 1, 
    reasoning_steps_n: int = 1,
    allow_override: bool = False,
    recurrent_layers: Optional[int] = None,
    iterations: Optional[int] = None,
) -> bool:
    """
    Guards against unrolling heavy multi-layer recurrence on Apple Silicon unified memory.
    Enforces Top-K layer recycling (recurrent_layer_count <= 4) or low iteration depth.
    Full 18-layer Gemma unrolling over multiple steps is blocked to protect macOS kernel stability.
    """
    if recurrent_layers is not None:
        recurrent_layer_count = recurrent_layers
    if iterations is not None:
        iterations_T = iterations

    total_layer_passes = recurrent_layer_count * iterations_T * reasoning_steps_n
    MAX_SAFE_LOCAL_LAYER_PASSES = 36  # Safe ceiling for local testing
    
    if total_layer_passes > MAX_SAFE_LOCAL_LAYER_PASSES and not allow_override:
        raise RuntimeError(
            f"[MemoryGuard Security] Heavy recurrence requested: {recurrent_layer_count} layers x "
            f"{iterations_T} iters x {reasoning_steps_n} steps = {total_layer_passes} layer passes. "
            f"Local Apple Silicon limit is {MAX_SAFE_LOCAL_LAYER_PASSES} layer passes. "
            f"Use Top-K layer recycling (recurrent_layer_count <= 2) or run on Google Cloud Vertex AI Custom Training (NVIDIA L4)."
        )
    return True
