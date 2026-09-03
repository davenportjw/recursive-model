#!/usr/bin/env python3
"""
Google Cloud PyTorch Training Script for Tiny Recursive Gemma (TRM).
Runs on Google Vertex AI Custom Training with NVIDIA L4/T4 GPUs.
Implements:
  - Dual continuous latents (z reasoning, y solution)
  - Detached multi-step deep supervision with power decay weights
  - Adaptive Computation Time (ACT) halting head
  - LoRA fine-tuning on Gemma language model
  - Direct checkpoint export to Google Cloud Storage (gs://davenport-boutique-vertex-staging/)
"""

import argparse
import json
import math
import os
import sys
import time
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional imports for cloud execution with graceful fallbacks
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class ACTHaltingHead(nn.Module):
    """Adaptive Computation Time Halting classifier head."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, hidden_size] or [batch, 1, hidden_size]
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.head(x).squeeze(-1)


def compute_step_weights(T: int, gamma: float = 1.5, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Computes power decay weights for multi-step deep supervision: w_t = t^gamma / sum(j^gamma)."""
    steps = torch.arange(1, T + 1, dtype=torch.float32, device=device)
    weights = torch.pow(steps, gamma)
    weights = weights / torch.sum(weights)
    return weights


class CodeReasoningDataset(Dataset):
    """Dataset for recursive continuous latent training."""
    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 512):
        self.samples = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        prompt = item.get("prompt", "")
                        solution = item.get("solution") or item.get("code") or ""
                        if prompt and solution:
                            self.samples.append({"prompt": prompt, "solution": solution})
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            # Synthetic fallback for mock runs
            prompt = "def add(a, b):\n"
            solution = "    return a + b\n"
        else:
            item = self.samples[idx]
            prompt = item["prompt"]
            solution = item["solution"]

        full_text = prompt + solution
        if self.tokenizer:
            prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
            full_enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
            
            input_ids = full_enc["input_ids"].squeeze(0)
            labels = input_ids.clone()
            # Mask prompt tokens with -100
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels[:prompt_len] = -100

            return {
                "input_ids": input_ids,
                "labels": labels,
                "prompt_len": torch.tensor(prompt_len, dtype=torch.long)
            }
        else:
            # Synthetic tensor fallback
            return {
                "input_ids": torch.randint(10, 1000, (64,)),
                "labels": torch.randint(10, 1000, (64,)),
                "prompt_len": torch.tensor(20, dtype=torch.long)
            }


def upload_directory_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str) -> None:
    """Uploads trained checkpoints directly to Google Cloud Storage."""
    if not GCS_AVAILABLE:
        print("[Cloud TRM Warning] google-cloud-storage not installed; skipping GCS upload.")
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    print(f"[Cloud TRM] Uploading artifacts from {local_dir} to gs://{bucket_name}/{gcs_prefix} ...")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            blob_path = os.path.join(gcs_prefix, relative_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"  ✓ Uploaded gs://{bucket_name}/{blob_path}")


def train(args):
    print("=" * 70)
    print("Google Cloud Vertex AI Training: Tiny Recursive Gemma (PyTorch TRM)")
    print("=" * 70)
    print(f"Model ID:              {args.model_id}")
    print(f"Dataset Path:          {args.data_path}")
    print(f"Output Directory:      {args.output_dir}")
    print(f"Epochs:                {args.epochs}")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Learning Rate:         {args.lr}")
    print(f"Recurrence Steps (T):  {args.iterations}")
    print(f"Reasoning Steps (n):   {args.reasoning_steps}")
    print(f"Decay Gamma:           {args.decay_gamma}")
    print(f"ACT Halting Head:      {args.act}")
    print(f"GCS Output Bucket:     {args.gcs_output_bucket}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Cloud TRM] Active Device: {device}")
    if torch.cuda.is_available():
        print(f"[Cloud TRM] GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"[Cloud TRM] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = None
    model = None

    if TRANSFORMERS_AVAILABLE and not args.mock_model:
        print(f"[Cloud TRM] Loading tokenizer and model: {args.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        # Apply LoRA on attention projection layers
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        print("[Cloud TRM] Running in lightweight simulation/mock mode for local validation.")
        # Lightweight dummy module with same interfaces
        class MockModel(nn.Module):
            def __init__(self, hidden_size=256, vocab_size=1000):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
                    num_layers=2
                )
                self.lm_head = nn.Linear(hidden_size, vocab_size)
            def forward(self, input_ids=None, inputs_embeds=None, labels=None):
                if inputs_embeds is None:
                    inputs_embeds = self.embed(input_ids)
                h = self.layers(inputs_embeds)
                logits = self.lm_head(h)
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                return type('Outputs', (), {'loss': loss, 'logits': logits, 'hidden_states': h})()
            def get_input_embeddings(self):
                return self.embed
            def save_pretrained(self, path):
                torch.save(self.state_dict(), os.path.join(path, "adapter_model.bin"))

        model = MockModel().to(device)

    # Initialize ACT Halting Head
    hidden_size = getattr(model.config, "hidden_size", 256) if hasattr(model, "config") else 256
    act_head = ACTHaltingHead(hidden_size).to(device)

    # Optimizers
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(act_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    dataset = CodeReasoningDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    step_weights = compute_step_weights(args.iterations, args.decay_gamma, device=device)
    print(f"[Cloud TRM] Deep supervision weights for T={args.iterations} (gamma={args.decay_gamma}): {step_weights.tolist()}")

    # Training Loop
    model.train()
    act_head.train()
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Embeddings
            embed_fn = model.get_input_embeddings()
            inputs_embeds = embed_fn(input_ids)
            B, L, D = inputs_embeds.shape

            # Initialize continuous latents z (reasoning) and y (solution)
            z = torch.zeros(B, 1, D, device=device)
            y = torch.zeros(B, 1, D, device=device)

            total_loss = torch.tensor(0.0, device=device)

            # Recurrent loop with detached deep supervision
            for t in range(1, args.iterations + 1):
                # Detached recurrence for O(1) memory scaling
                z_in = z.detach()
                y_in = y.detach()

                # Reason step: update z n times
                for _ in range(args.reasoning_steps):
                    z_in = F.rms_norm(z_in, (D,))
                    y_in = F.rms_norm(y_in, (D,))
                    # Prefix injection
                    step_embeds = torch.cat([inputs_embeds, y_in, z_in], dim=1)
                    outputs = model(inputs_embeds=step_embeds)
                    z_in = outputs.logits[:, -1:, :D]  # Latent update

                z = z_in
                # Update solution state y
                y = F.rms_norm(y_in, (D,))

                # Emitted generation loss for step t
                loss_step = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)

                # ACT Halting head prediction and BCE loss
                if args.act:
                    halt_prob = act_head(z)
                    # Ideal halt target: 1.0 on last step, 0.0 before
                    target_halt = torch.ones_like(halt_prob) if t == args.iterations else torch.zeros_like(halt_prob)
                    bce_loss = F.binary_cross_entropy(halt_prob, target_halt)
                else:
                    bce_loss = torch.tensor(0.0, device=device)

                # Deep supervision step contribution
                wt = step_weights[t - 1]
                total_loss = total_loss + (wt * loss_step) + (args.act_loss_weight * bce_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Batch {batch_idx}/{len(dataloader)} - Loss: {total_loss.item():.4f}")

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"[Epoch {epoch}/{args.epochs}] Complete - Avg Loss: {avg_loss:.4f} - Elapsed: {time.time() - start_time:.1f}s")

    # Save trained checkpoint
    print(f"[Cloud TRM] Saving model checkpoint to {args.output_dir} ...")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(args.output_dir)
    else:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        
    torch.save(act_head.state_dict(), os.path.join(args.output_dir, "act_head.pt"))
    
    # Save training metadata
    meta = {
        "model_id": args.model_id,
        "iterations": args.iterations,
        "reasoning_steps": args.reasoning_steps,
        "decay_gamma": args.decay_gamma,
        "epochs": args.epochs,
        "final_loss": round(avg_loss, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Cloud TRM] Local checkpoint saved successfully at {args.output_dir}.")

    # Upload to GCS if specified
    if args.gcs_output_bucket:
        gcs_prefix = f"checkpoints/trm-job-{int(time.time())}"
        upload_directory_to_gcs(args.output_dir, args.gcs_output_bucket, gcs_prefix)


def main():
    parser = argparse.ArgumentParser(description="Google Cloud Vertex AI Training for Tiny Recursive Gemma")
    parser.add_argument("--model-id", default="google/gemma-2-2b-it", help="Hugging Face model ID")
    parser.add_argument("--data-path", default="data/continuous_train_augmented.jsonl", help="Training dataset path")
    parser.add_argument("--output-dir", default="/tmp/trm_checkpoints", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=3, help="Number of recurrent unrolls (T)")
    parser.add_argument("--reasoning-steps", type=int, default=2, help="Inner reasoning steps per unroll (n)")
    parser.add_argument("--decay-gamma", type=float, default=1.5, help="Power decay weight gamma for deep supervision")
    parser.add_argument("--act", action="store_true", default=True, help="Enable ACT halting head training")
    parser.add_argument("--act-loss-weight", type=float, default=0.1, help="Weight for ACT BCE loss")
    parser.add_argument("--gcs-output-bucket", default="davenport-boutique-vertex-staging", help="GCS bucket for checkpoint export")
    parser.add_argument("--mock-model", action="store_true", help="Use lightweight mock model for local testing")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
