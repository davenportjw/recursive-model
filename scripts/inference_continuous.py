import argparse
import os
from dotenv import load_dotenv
from tiny_recursive_gemma import ContinuousLatentPipeline

load_dotenv()

def main(model_path, weights_path, prompt):
    print(f"--- Running Continuous Inference ---")
    pipeline = ContinuousLatentPipeline(model_path, weights_path)
    
    response = pipeline(prompt)
    print(f"\nGenerated Output:\n{response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        default=os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized")
    )
    parser.add_argument(
        "--weights", 
        default=os.path.join(os.getenv("ADAPTER_PATH", "adapters"), "continuous_weights.safetensors")
    )
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    
    main(args.model, args.weights, args.prompt)
