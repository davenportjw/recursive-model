import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from tiny_recursive_gemma.continuous_model import ACTHaltingHead, rms_norm, generate_continuous

def test_act_halting_head_initialization_and_forward():
    """Verifies ACTHaltingHead negative bias initialization and output bounds."""
    hidden_dim = 32
    head = ACTHaltingHead(hidden_dim)
    
    # Check that initial bias is negative (-2.0)
    assert float(head.fc.bias[0].item()) == pytest.approx(-2.0, abs=1e-3)
    
    # Feed zero latents: output should be exactly sigmoid(-2.0) ~ 0.1192
    y_zero = mx.zeros((1, 1, hidden_dim))
    z_zero = mx.zeros((1, 1, hidden_dim))
    prob = head(y_zero, z_zero)
    
    assert prob.shape == (1, 1, 1)
    val = float(prob[0, 0, 0].item())
    assert 0.11 < val < 0.13
    assert val < 0.20 # Early stopping is conservative at initialization

def test_act_halting_head_optimization():
    """Verifies that the ACT head learns to increase halting probability under BCE loss."""
    hidden_dim = 16
    head = ACTHaltingHead(hidden_dim)
    opt = optim.Adam(learning_rate=0.1)
    
    y = mx.ones((1, 1, hidden_dim))
    z = mx.ones((1, 1, hidden_dim))
    
    def loss_fn(h, y_in, z_in):
        p = h(y_in, z_in)
        return mx.mean(-mx.log(p + 1e-7))
        
    loss_and_grad = nn.value_and_grad(head, loss_fn)
    
    initial_p = float(head(y, z)[0, 0, 0].item())
    for _ in range(25):
        loss, grads = loss_and_grad(head, y, z)
        opt.update(head, grads)
        mx.eval(head.state, opt.state)
        
    final_p = float(head(y, z)[0, 0, 0].item())
    assert final_p > initial_p
    assert final_p > 0.85

def test_rms_norm_invariance():
    """Verifies that rms_norm scales vectors so that quadratic mean is unit variance."""
    hidden_dim = 64
    x = mx.random.normal((2, 5, hidden_dim)) * 15.0 # Large arbitrary magnitude
    normalized = rms_norm(x)
    
    # Mean of squared elements along the last axis should be approximately 1.0
    ms = mx.mean(normalized ** 2, axis=-1)
    assert mx.allclose(ms, mx.ones_like(ms), atol=1e-3)

def test_deep_supervision_decay_weights():
    """Verifies power decay weights sum to 1.0 and increase monotonically."""
    iterations = 5
    gamma = 1.0
    raw_w = [(t + 1) ** gamma for t in range(iterations)]
    sum_w = sum(raw_w)
    weights = [w / sum_w for w in raw_w]
    
    assert pytest.approx(sum(weights), 1e-5) == 1.0
    assert weights[0] < weights[1] < weights[2] < weights[3] < weights[4]
    assert weights[-1] == pytest.approx(5.0 / 15.0, abs=1e-4)

def test_generate_continuous_with_act_early_halting():
    """Verifies that generate_continuous triggers early halting when ACT head emits high confidence."""
    hidden_dim = 16
    
    # Mock model conforming to MLX LM structure (model.model is the transformer)
    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(50, hidden_dim)
        def __call__(self, input_embeddings, cache=None):
            return input_embeddings
            
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MockTransformer()
            self.lm_head = nn.Linear(hidden_dim, 50)
            
    class MockTokenizer:
        def encode(self, text):
            return [1, 2, 3]
        def decode(self, tokens):
            return "def solution(): return 42"
        eos_token_id = 49
        additional_special_tokens_ids = []

    model = MockModel()
    tokenizer = MockTokenizer()
    
    # Halting head configured to halt immediately
    head = ACTHaltingHead(hidden_dim)
    head.fc.bias = mx.full((1,), 10.0) # Sigmoid(10.0) ~ 0.9999
    
    text, telemetry = generate_continuous(
        model=model,
        tokenizer=tokenizer,
        prompt="Write solution",
        max_tokens=4,
        iterations=8,
        min_iterations=2,
        halting_head=head,
        halt_threshold=0.85,
        return_telemetry=True
    )
    
    # Must have halted early at min_iterations = 2 instead of running all 8 iterations
    assert telemetry["halted_early"] is True
    assert telemetry["iterations"] == 2
    assert "def solution" in text
