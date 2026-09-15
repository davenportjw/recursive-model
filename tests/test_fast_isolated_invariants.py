import json
import os
import pytest
import mlx.core as mx
import mlx.nn as nn

from tiny_recursive_gemma.memory_guard import verify_recursion_guardrails
from tiny_recursive_gemma.continuous_model import ACTHaltingHead, rms_norm


def test_recursion_guardrails_safety():
    """Verify recursion guardrails allow safe params and block dangerous unrolls on Apple Silicon."""
    # Safe on Apple Silicon: 2 recurrent layers * 3 iterations = 6 < 36
    assert verify_recursion_guardrails(recurrent_layers=2, iterations=3) is True
    # Safe boundary: 18 layers * 2 iterations = 36 <= 36
    assert verify_recursion_guardrails(recurrent_layers=18, iterations=2) is True
    # Dangerous: 18 layers * 5 iterations = 90 > 36
    with pytest.raises(RuntimeError, match="MemoryGuard Security"):
        verify_recursion_guardrails(recurrent_layers=18, iterations=5)


def test_act_halting_head_dual_latent_mlx():
    """Verify ACTHaltingHead accepts dual latents (y, z) and outputs probabilities in [0, 1]."""
    hidden_dim = 64
    batch_size = 2
    act_head = ACTHaltingHead(hidden_dim)
    
    y = mx.random.normal((batch_size, 1, hidden_dim))
    z = mx.random.normal((batch_size, 1, hidden_dim))
    
    prob = act_head(y, z)
    assert prob.shape == (batch_size, 1, 1)
    assert (prob >= 0.0).all().item()
    assert (prob <= 1.0).all().item()


def test_rms_norm_invariance():
    """Verify RMSNorm stabilizes tensor variance across hidden dimension."""
    x = mx.random.normal((2, 4, 128)) * 10.0
    normed = rms_norm(x)
    assert normed.shape == x.shape
    var = mx.mean(normed ** 2, axis=-1)
    assert mx.allclose(var, mx.ones_like(var), atol=1e-2).item()


def test_causal_prefix_indexing_invariant_mlx():
    """
    Verify causal prefix ordering: [prompt_embeds, z_inj, y_inj, target_embeds_in]
    Ensures that target tokens appear strictly after latents (z, y).
    """
    B, Lp, Lt, D = 2, 8, 12, 32
    prompt_embeds = mx.random.normal((B, Lp, D))
    target_embeds_in = mx.random.normal((B, Lt, D))

    z_inj = mx.random.normal((B, 1, D))
    y_inj = mx.random.normal((B, 1, D))

    # Prefix concatenation: [prompt, z, y, target]
    full_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj, target_embeds_in], axis=1)
    assert full_embeds.shape == (B, Lp + 2 + Lt, D)

    # Verify positions:
    # prompt: 0 .. Lp - 1
    # z_inj: Lp
    # y_inj: Lp + 1
    # target_embeds_in: Lp + 2 .. Lp + 2 + Lt - 1
    prefix_len = Lp + 2
    target_slice = full_embeds[:, prefix_len:, :]
    assert target_slice.shape == (B, Lt, D)
    assert mx.allclose(target_slice, target_embeds_in).item()


def test_hard_reasoning_suite_integrity():
    """Verify the 100 tasks in hard_reasoning_suite_100.jsonl meet all integrity requirements."""
    suite_path = "eval/hard_reasoning_suite_100.jsonl"
    assert os.path.exists(suite_path), f"Suite file {suite_path} must exist"

    with open(suite_path, "r") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    assert len(tasks) == 100, f"Expected 100 tasks, found {len(tasks)}"

    for idx, t in enumerate(tasks):
        assert "task_id" in t
        assert "prompt" in t and len(t["prompt"].strip()) > 10
        assert "entry_point" in t and len(t["entry_point"]) > 0
        assert "canonical_solution" in t and len(t["canonical_solution"].strip()) > 0
        assert "test" in t and "check" in t["test"]
        # Verify no fake dummy answers
        assert "TODO" not in t["canonical_solution"]
        assert "..." != t["canonical_solution"].strip()
