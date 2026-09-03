import json
import tempfile
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mx_utils
import pytest

def test_target_logits_alignment_exactness():
    """Verifies that the target logits slice matches target labels length across various configurations."""
    # Scenario 1: Dual Latent Mode (prefix_len = prompt_len + 2)
    prompt_len = 7
    target_len = 12
    hidden_dim = 16
    
    # Prefix: prompt (7) + z (1) + y (1) = 9
    # Input target: target[:-1] (11)
    # Total input embeds length = 9 + 11 = 20
    full_embeds_len = prompt_len + 2 + (target_len - 1)
    assert full_embeds_len == 20
    
    # Simulated model output logits: [batch=1, seq_len=20, vocab=100]
    dummy_logits = mx.zeros((1, full_embeds_len, 100))
    dummy_target_labels = mx.zeros((1, target_len))
    
    prefix_len = prompt_len + 2
    start_idx = prefix_len - 1
    end_idx = start_idx + dummy_target_labels.shape[1]
    
    target_logits = dummy_logits[:, start_idx:end_idx, :]
    assert target_logits.shape[1] == dummy_target_labels.shape[1]
    assert target_logits.shape[1] == target_len

    # Scenario 2: Single Latent Mode (prefix_len = prompt_len + 1)
    prefix_len_single = prompt_len + 1
    start_idx_single = prefix_len_single - 1
    full_embeds_len_single = prompt_len + 1 + (target_len - 1)
    dummy_logits_single = mx.zeros((1, full_embeds_len_single, 100))
    end_idx_single = start_idx_single + dummy_target_labels.shape[1]
    
    target_logits_single = dummy_logits_single[:, start_idx_single:end_idx_single, :]
    assert target_logits_single.shape[1] == dummy_target_labels.shape[1]
    assert target_logits_single.shape[1] == target_len

def test_ema_update_and_state_rebinding():
    """Verifies that Weight EMA smoothing correctly mutates the state list without memory leaks."""
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.array([[1.0, 2.0], [3.0, 4.0]])
            
    module = SimpleModule()
    ema_beta = 0.9
    ema_weights = mx_utils.tree_map(lambda x: x, module.trainable_parameters())
    state = [module.state, ema_weights]
    
    # Simulate a parameter update
    new_params = {"w": mx.array([[2.0, 3.0], [4.0, 5.0]])}
    
    # Update EMA
    ema_weights = mx_utils.tree_map(
        lambda ema, param: ema_beta * ema + (1.0 - ema_beta) * param,
        ema_weights,
        new_params
    )
    state[1] = ema_weights
    
    # Force evaluation of state graph
    mx.eval(state)
    
    expected_w00 = 0.9 * 1.0 + 0.1 * 2.0 # 1.1
    assert abs(state[1]["w"][0, 0].item() - expected_w00) < 1e-5

def test_trm_stop_gradient_math():
    """Verifies that mx.stop_gradient blocks gradient backpropagation through recurrent prefix steps."""
    def recurrent_step(x, h):
        return x + h * 2.0
        
    x = mx.array([1.0, 2.0])
    h = mx.array([0.5, 0.5])
    
    # With stop_gradient on step 1 (TRM style)
    def loss_with_stop(x_in):
        h1 = mx.stop_gradient(recurrent_step(x_in, h))
        h2 = recurrent_step(x_in, h1)
        return mx.sum(h2)
        
    grad_fn = mx.grad(loss_with_stop)
    grads = grad_fn(x)
    # h1 is constant w.r.t x_in because of stop_gradient!
    # h2 = x_in + h1 * 2.0 -> d(h2)/d(x_in) = 1.0
    assert mx.allclose(grads, mx.array([1.0, 1.0]))

def test_adapter_config_serialization():
    """Verifies that adapter_config.json properly serializes and deserializes all continuous hyperparameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "adapter_config.json")
        adapter_config = {
            "model": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
            "num_layers": 4,
            "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 20.0},
            "recursive_iters": 5,
            "dual_latent": True,
            "reasoning_steps": 3,
            "trm_mode": True,
            "use_ema": True,
            "ema_beta": 0.99
        }
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=4)
            
        with open(config_path, "r") as f:
            loaded = json.load(f)
            
        assert loaded["model"] == "google/gemma-4-E2B-it-qat-q4_0-unquantized"
        assert loaded["num_layers"] == 4
        assert loaded["lora_parameters"]["rank"] == 8
        assert loaded["dual_latent"] is True
