import os
import pytest
import mlx.core as mx
from dotenv import load_dotenv
from tiny_recursive_gemma.memory_guard import (
    init_memory_guardrails,
    flush_memory,
    get_metal_memory_stats,
    guarded_memory_scope
)
from tiny_recursive_gemma.continuous_model import (
    get_or_load_model,
    get_transformer_layers,
    get_logits,
    generate_continuous,
    ContinuousLatentPipeline
)

# Guard against heavy local execution on macOS
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_HEAVY_TESTS") != "1",
    reason="Heavy model loading disabled on local Mac to prevent kernel memory watchdog panics. Run in Google Cloud."
)

# We use the E2B model with Apple Silicon memory guardrails
MODEL_ID = os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized")

@pytest.fixture(scope="module")
def model_and_tokenizer():
    init_memory_guardrails(max_memory_gb=3.5, max_cache_gb=0.5)
    model, tokenizer = get_or_load_model(MODEL_ID)
    yield model, tokenizer
    flush_memory()

@pytest.fixture(autouse=True)
def per_test_guardrails():
    # Purge cache before and after every test
    flush_memory()
    yield
    flush_memory()

def test_get_transformer_layers(model_and_tokenizer):
    model, _ = model_and_tokenizer
    transformer, lm = get_transformer_layers(model)
    assert transformer is not None
    assert lm is not None

def test_get_logits(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    transformer, lm = get_transformer_layers(model)
    
    prompt = "Hello"
    tokens = mx.array(tokenizer.encode(prompt))[None]
    embeddings = transformer.embed_tokens(tokens)
    
    logits, hidden_states = get_logits(model, transformer, lm, embeddings)
    assert logits.shape[0] == 1
    assert logits.shape[1] == tokens.shape[1]
    
    # Vocabulary size can vary, check that logits are 3D
    assert len(logits.shape) == 3
    assert hidden_states.shape == embeddings.shape

def test_generate_continuous_with_guardrails(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "2+2="
    
    with guarded_memory_scope(max_memory_gb=3.5, max_cache_gb=0.5):
        response, telemetry = generate_continuous(
            model,
            tokenizer,
            prompt,
            max_tokens=10,
            iterations=2,
            dual_latent=True,
            reasoning_steps=2,
            return_telemetry=True
        )
    assert isinstance(response, str)
    assert len(response) > 0
    assert telemetry["iterations"] == 2
    assert "trajectory_distances" in telemetry

def test_pipeline_initialization_reuses_singleton(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    # Injects existing loaded model to avoid duplicate memory allocation
    pipeline = ContinuousLatentPipeline(model=model, tokenizer=tokenizer)
    assert pipeline.model is not None
    assert pipeline.tokenizer is not None
    
    response = pipeline("What is 3+3?", max_tokens=10, iterations=2)
    assert isinstance(response, str)
    assert len(response) > 0

def test_memory_remains_bounded(model_and_tokenizer):
    stats = get_metal_memory_stats()
    # Active memory should strictly remain under our 3.5GB ceiling
    assert stats["active_mb"] < 3500.0
