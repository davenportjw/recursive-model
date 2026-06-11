import argparse
import os
from dotenv import load_dotenv
from tiny_recursive_gemma import run_recursive_inference

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        default=os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized"), 
        help="Model path"
    )
    parser.add_argument(
        "--adapter", 
        default=os.getenv("ADAPTER_PATH", "adapters"), 
        help="Path to LoRA adapters"
    )
    parser.add_argument("--iters", type=int, default=2, help="Number of recursive improvements")
    parser.add_argument("--prompt", type=str, required=True, help="The coding task")
    args = parser.parse_args()
    
    run_recursive_inference(args.model, args.adapter, args.prompt, args.iters)

