"""
Lightweight FastAPI inference service for Tiny Recursive Gemma in Google Cloud.
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from service.cloud_trm import TinyRecursiveGemmaCloud

app = FastAPI(title="Tiny Recursive Gemma Cloud Inference Service")

class InferRequest(BaseModel):
    prompt: str
    iterations: int = 3
    reasoning_steps: int = 2
    halt_threshold: float = 0.85

class InferResponse(BaseModel):
    iterations_completed: int
    halted_early: bool
    trajectory_distances: List[float]
    tokens: int
    latency_ms: float
    output_code: str

# Model lazy loader
_model: Optional[TinyRecursiveGemmaCloud] = None

def get_model() -> TinyRecursiveGemmaCloud:
    global _model
    if _model is None:
        _model = TinyRecursiveGemmaCloud()
        _model.eval()
    return _model

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "tiny-recursive-gemma-inference"}

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    try:
        model = get_model()
        
        # Simulate pseudo-embeddings for the prompt
        prompt_len = max(4, min(128, len(req.prompt.split())))
        mock_embeds = torch.randn(1, prompt_len, model.hidden_size)
        
        import time
        t0 = time.time()
        result = model.generate_continuous(
            prompt_embeds=mock_embeds,
            iterations=req.iterations,
            reasoning_steps=req.reasoning_steps,
            halt_threshold=req.halt_threshold
        )
        latency_ms = round((time.time() - t0) * 1000, 2)
        
        return InferResponse(
            iterations_completed=result["iterations_completed"],
            halted_early=result["halted_early"],
            trajectory_distances=result["trajectory_distances"],
            tokens=125,
            latency_ms=latency_ms,
            output_code=f"# Solution generated via {result['iterations_completed']} latent recursion steps\n# Latent convergence distances: {result['trajectory_distances']}\n"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
