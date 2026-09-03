"""
Portable PyTorch implementation of Tiny Recursive Gemma (TRM) for Google Cloud.
Adapts the dual-latent (y, z) recurrence architecture with stop-gradients to PyTorch/CUDA/CPU.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyRecursiveGemmaCloud(nn.Module):
    """
    Cloud-ready TRM module mirroring the continuous latent state injection mechanism.
    Maintains dual latent states:
      - z: reasoning scratchpad updated n times per iteration
      - y: candidate solution representation updated 1 time per iteration
    """
    def __init__(self, hidden_size: int = 2048, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Latent projection and norm layers
        self.norm_z = nn.RMSNorm(hidden_size)
        self.norm_y = nn.RMSNorm(hidden_size)
        
        # Adaptive Computation Time (ACT) Halting Head
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Compact Recursive Reasoning Block (simulates top LoRA layers)
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

    def forward_step(self, prompt_embeds: torch.Tensor, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        """
        Concatenates [prompt, state_a, state_b] and passes through transformer reasoning block.
        Returns the final pooled state.
        """
        # Shape: [batch, seq_len + 2, hidden_size]
        x = torch.cat([prompt_embeds, state_a, state_b], dim=1)
        for layer in self.reasoning_layers:
            x = layer(x)
        return x[:, -1:, :]

    @torch.no_grad()
    def generate_continuous(
        self,
        prompt_embeds: torch.Tensor,
        iterations: int = 3,
        reasoning_steps: int = 2,
        halt_threshold: float = 0.85
    ) -> Dict[str, any]:
        """
        Runs continuous latent recurrence with stop-gradient prefix and ACT halting.
        """
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device

        # Initialize dual latents (z reasoning, y solution)
        z = torch.zeros(batch_size, 1, self.hidden_size, device=device)
        y = torch.zeros(batch_size, 1, self.hidden_size, device=device)

        trajectory_distances: List[float] = []
        halted_early = False
        completed_iters = iterations

        for t in range(1, iterations + 1):
            prev_z = z.clone()

            # Inner reasoning loop: update z n times
            for _ in range(reasoning_steps):
                z_norm = self.norm_z(z)
                y_norm = self.norm_y(y)
                z = self.forward_step(prompt_embeds, y_norm, z_norm)

            # Update solution representation y
            z_norm = self.norm_z(z)
            y_norm = self.norm_y(y)
            y = self.forward_step(prompt_embeds, z_norm, y_norm)

            # Measure cosine distance d(z_t, z_{t-1})
            cos_sim = F.cosine_similarity(z.squeeze(1), prev_z.squeeze(1), dim=-1).mean().item()
            cos_dist = float(max(0.0, 1.0 - cos_sim))
            trajectory_distances.append(round(cos_dist, 4))

            # Adaptive Computation Time check
            halt_prob = float(self.act_head(y).squeeze().item())
            if t >= 2 and (halt_prob >= halt_threshold or cos_dist < 0.02):
                halted_early = True
                completed_iters = t
                break

        return {
            "iterations_completed": completed_iters,
            "halted_early": halted_early,
            "trajectory_distances": trajectory_distances,
            "final_y": y,
            "final_z": z
        }
