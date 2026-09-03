import os
import json
import re
from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm

import argparse

# Initialize Gemini Client for Vertex AI
import os
import json
import re
import ast
from typing import Optional
from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm

import argparse

# Teacher model for synthetic multi-step reasoning data generation
MODEL_NAME = "gemini-3.8-flash"

def init_client(project_id: Optional[str] = None, location: str = "us-central1"):
    """Initializes Gemini client via Google AI Studio API key or Vertex AI."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"Initializing Gemini Client with API key...")
        return genai.Client(api_key=api_key)
        
    if project_id:
        try:
            print(f"Initializing Vertex AI Gemini Client (Project: {project_id}, Region: {location})...")
            return genai.Client(vertexai=True, project=project_id, location=location)
        except Exception as e:
            print(f"Could not initialize Vertex AI Gemini Client: {e}")
            return None
            
    print("Error: Neither GEMINI_API_KEY environment variable nor --project argument was provided.")
    return None

SYSTEM_PROMPT = """You are an expert programmer acting as a teacher model.
You will be given a Python coding task. You must follow this EXACT process:
1. Write a quick, naive solution (it can have minor bugs or be suboptimal) inside <draft> tags. Do not write markdown blocks inside the tags, just the raw python code.
2. Analyze your naive solution inside <critique> tags. Point out edge cases it missed, potential bugs, or performance inefficiencies.
3. Write the perfect, robust solution inside <final_code> tags. Do not write markdown blocks inside the tags, just the raw python code.

Ensure your code inside the tags contains ONLY valid Python code without formatting ticks.
"""

def clean_code(code_str: str) -> str:
    """Strips any accidental markdown backticks from code blocks."""
    code_str = code_str.strip()
    code_str = re.sub(r'^```(?:python)?\s*\n?', '', code_str)
    code_str = re.sub(r'\n?```$', '', code_str)
    return code_str.strip()

def parse_response(text: str):
    draft_match = re.search(r'<draft>(.*?)</draft>', text, re.DOTALL)
    critique_match = re.search(r'<critique>(.*?)</critique>', text, re.DOTALL)
    final_match = re.search(r'<final_code>(.*?)</final_code>', text, re.DOTALL)
    
    if draft_match and critique_match and final_match:
        draft = clean_code(draft_match.group(1))
        critique = critique_match.group(1).strip()
        final_code = clean_code(final_match.group(1))
        
        # Validate that the final code is syntactically valid Python
        try:
            ast.parse(final_code)
        except SyntaxError:
            print(f"Warning: Syntactically invalid final code generated. Discarding sample.")
            return None
            
        return {
            "draft": draft,
            "critique": critique,
            "final_code": final_code
        }
    return None

def main(client, num_samples=50, output_file="data/synthetic_humaneval.jsonl"):
    print("Loading HumanEval dataset...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    
    # We take the first `num_samples`
    samples = list(ds)[:num_samples]
    
    print(f"Generating synthetic trajectories for {num_samples} tasks using {MODEL_NAME}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    successful = 0
    with open(output_file, "w") as f:
        for item in tqdm(samples):
            prompt = f"Task:\n{item['prompt']}"
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.4,
                    )
                )
                
                parsed = parse_response(response.text)
                if parsed:
                    record = {
                        "task_id": item["task_id"],
                        "prompt": item["prompt"],
                        "test": item["test"],
                        "entry_point": item["entry_point"],
                        "trajectory": parsed
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    successful += 1
                else:
                    print(f"\nFailed to parse response for {item['task_id']}")
            except Exception as e:
                print(f"\nError generating for {item['task_id']}: {e}")
                
    print(f"\nDone! Successfully generated {successful}/{num_samples} trajectories.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data via Gemini 3.8 Flash")
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="Google Cloud Project ID (for Vertex AI)")
    parser.add_argument("--location", default="us-central1", help="Google Cloud Region (e.g., us-central1)")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to generate")
    args = parser.parse_args()
    
    client = init_client(args.project, args.location)
    if client:
        main(client=client, num_samples=args.samples)
