import argparse
import os
from dotenv import load_dotenv
from tiny_recursive_gemma import train_continuous_model

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        default=os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized"), 
        help="Base model path"
    )
    parser.add_argument(
        "--data", 
        default="data/continuous_train.jsonl" if os.path.exists("data/continuous_train.jsonl") else "data/train.jsonl", 
        help="Path to JSONL training data"
    )
    parser.add_argument("--iters", type=int, default=5, help="Number of training steps/samples to optimize")
    parser.add_argument(
        "--output", 
        default=os.path.join(os.getenv("ADAPTER_PATH", "adapters"), "continuous_weights.safetensors"), 
        help="Output path for weights"
    )
    parser.add_argument("--lora-layers", type=int, default=2, help="Number of transformer layers from the end to apply LoRA to")
    parser.add_argument("--recursive-iters", type=int, default=3, help="Number of recursive steps (T)")
    parser.add_argument("--no-trm", dest="trm_mode", action="store_false", help="Disable Samsung TRM gradient-free prefix (defaults to True)")
    
    parser.add_argument("--no-dual-latent", dest="dual_latent", action="store_false", help="Disable Dual-Latent (y and z) reasoning (defaults to True)")
    parser.add_argument("--reasoning-steps", type=int, default=3, help="Number of reasoning steps (n) per iteration in Dual-Latent mode")
    
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", help="Disable Weight EMA stabilization (defaults to True)")
    parser.add_argument("--ema-beta", type=float, default=0.99, help="Weight EMA decay factor (beta)")
    
    parser.set_defaults(trm_mode=True, dual_latent=True, use_ema=True)
    args = parser.parse_args()
    
    train_continuous_model(
        model_path=args.model,
        data_path=args.data,
        iters=args.iters,
        output_path=args.output,
        lora_layers=args.lora_layers,
        trm_mode=args.trm_mode,
        recursive_iters=args.recursive_iters,
        dual_latent=args.dual_latent,
        reasoning_steps=args.reasoning_steps,
        use_ema=args.use_ema,
        ema_beta=args.ema_beta
    )
