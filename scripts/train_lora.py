import argparse
import os
from dotenv import load_dotenv
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.tuner import train
from mlx_lm.tuner.datasets import load_dataset
import json

def main(model_path, data_path, iters, lora_layers):
    print(f"Loading model {model_path}...")
    # Load base model and tokenizer
    model, tokenizer = load(model_path)
    
    # Freeze base model parameters
    model.freeze()
    
    # Configure LoRA
    # In MLX LM, the LoRA configuration is typically passed to the training loop.
    # We will use the conversational dataset format.
    
    # Ensure dataset is loaded properly (MLX expects train, valid, test splits, we just have one JSONL)
    # We will pass the data_path to the custom loading config
    
    # Note: MLX LM's default training script uses YAML/CLI arguments. 
    # For a custom script, we setup a dict of args.
    
    adapter_path = os.getenv("ADAPTER_PATH", "adapters")
    training_args = {
        "model": model_path,
        "train": True,
        "data": data_path,
        "iters": iters,
        "batch_size": 1,      # Crucial for 16GB RAM
        "lora_layers": lora_layers,
        "learning_rate": 1e-4,
        "steps_per_report": 5,
        "save_every": 20,
        "adapter_path": adapter_path,
        "max_seq_length": 2048 # Adjust if OOM
    }
    
    # The actual train method in MLX-LM might require specific config objects.
    # To keep it standard with the MLX ecosystem, it's often easier to invoke their CLI programmatically 
    # or build a simple wrapper. Let's use `mlx_lm.tuner.train` directly if possible, or construct the args object.
    
    print("Starting LoRA fine-tuning...")
    # Since mlx_lm.tuner is highly CLI driven, it's often safer to run it via CLI.
    # But we'll try an programmatic invocation or recommend the CLI.
    print(f"Configuration: {training_args}")
    
    # Ensure valid.jsonl exists in data directory, or create a copy/split so mlx_lm doesn't crash
    if os.path.isdir(data_path):
        train_file = os.path.join(data_path, "train.jsonl")
        valid_file = os.path.join(data_path, "valid.jsonl")
        if not os.path.exists(valid_file) and os.path.exists(train_file):
            print(f"Creating default valid.jsonl from {train_file} for mlx_lm...")
            with open(train_file, "r") as tf:
                lines = tf.readlines()
            # Use last 10% (min 1 sample) for validation
            split_idx = max(1, int(len(lines) * 0.1))
            valid_lines = lines[-split_idx:] if len(lines) > 1 else lines
            with open(valid_file, "w") as vf:
                vf.writelines(valid_lines)

    import sys
    import subprocess
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", data_path,
        "--iters", str(iters),
        "--batch-size", "1",
        "--num-layers", str(lora_layers),
        "--max-seq-length", "2048",
        "--adapter-path", adapter_path
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        default=os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized"), 
        help="Model path/name"
    )
    parser.add_argument("--data", default="data", help="Directory containing train.jsonl")
    parser.add_argument("--iters", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--lora-layers", type=int, default=8, help="Number of LoRA layers")
    args = parser.parse_args()
    
    main(args.model, args.data, args.iters, args.lora_layers)
