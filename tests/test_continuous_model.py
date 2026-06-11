import os
import pytest
import mlx.core as mx
from mlx_lm import load
from dotenv import load_dotenv
from tiny_recursive_gemma.continuous_model import (
    get_transformer_layers,
    get_logits,
    generate_continuous,
    ContinuousLatentPipeline
)

load_dotenv()

# We use the small, fast E2B model for local tests or the model from environment variables
MODEL_ID = os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized")

@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = load(MODEL_ID)
    return model, tokenizer

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

def test_generate_continuous(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "2+2="
    response = generate_continuous(
        model,
        tokenizer,
        prompt,
        max_tokens=10,
        iterations=2,
        dual_latent=True,
        reasoning_steps=2
    )
    assert isinstance(response, str)
    assert len(response) > 0

def test_pipeline_initialization():
    pipeline = ContinuousLatentPipeline(MODEL_ID)
    assert pipeline.model is not None
    assert pipeline.tokenizer is not None
    
    response = pipeline("What is 3+3?", max_tokens=10, iterations=2)
    assert isinstance(response, str)
    assert len(response) > 0
