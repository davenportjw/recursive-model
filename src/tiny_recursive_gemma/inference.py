import re
from typing import Tuple
from mlx_lm import load, generate

def extract_thought_and_code(response: str) -> Tuple[str, str]:
    """Extracts thought process and code update from a model response.
    
    Employs a multi-tier fallback hierarchy:
    1. Explicit XML tags (<thought> and <code_update>) with markdown strip
    2. Markdown code blocks (```python ... ```)
    3. Unclosed tags or code block headers
    4. Raw Python code heuristic
    
    Args:
        response: Raw string response from the model.
        
    Returns:
        Tuple of (extracted_thought, extracted_code).
    """
    if not response or not response.strip():
        return "", ""
        
    # Tier 1: XML tags
    thought_match = re.search(r'<thought>(.*?)(?:</thought>|$)', response, re.DOTALL)
    code_match = re.search(r'<code_update>(.*?)(?:</code_update>|$)', response, re.DOTALL)
    
    if code_match:
        raw_code = code_match.group(1).strip()
        # Clean potential markdown fences within code_update
        fence_match = re.search(r'```(?:python)?\s*\n?(.*?)(?:```|$)', raw_code, re.DOTALL)
        if fence_match:
            code = fence_match.group(1).strip()
        else:
            code = raw_code
            
        thought = thought_match.group(1).strip() if thought_match else ""
        return thought, code
        
    # Tier 2: Markdown code blocks
    code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)(?:```|$)', response, re.DOTALL)
    if code_blocks:
        code = code_blocks[-1].strip()
        # Everything prior to the code block is treated as thought
        thought = re.split(r'```(?:python)?', response)[0].strip()
        # Remove any lingering thought tags if present
        thought = re.sub(r'</?thought>', '', thought).strip()
        return thought, code
        
    # Tier 3: Heuristic line parsing if model generated pure code or mixed commentary
    lines = response.splitlines()
    code_lines = []
    thought_lines = []
    in_code = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'return ', '@', 'if __name__')):
            in_code = True
        if in_code:
            code_lines.append(line)
        else:
            thought_lines.append(line)
            
    if code_lines:
        return "\n".join(thought_lines).strip(), "\n".join(code_lines).strip()
        
    # Tier 4: Fallback return raw response
    return "", response.strip()


def run_recursive_inference(
    model_path: str,
    adapter_path: str,
    prompt: str,
    max_iters: int = 2,
    model = None,
    tokenizer = None,
):
    """
    Runs the model recursively, feeding its thoughts and code back into itself.
    If model and tokenizer are provided, they are used directly to avoid reloading.
    """
    if model is None or tokenizer is None:
        print(f"Loading model {model_path} with adapters from {adapter_path}...")
        model, tokenizer = load(model_path, adapter_path=adapter_path)

    current_code = ""
    current_thought = ""
    
    for i in range(max_iters):
        print(f"\n--- Iteration {i+1} ---")
        
        # Build prompt using the same format as training
        user_msg = (
            f"Task:\n{prompt}\n\n"
            f"Current Code:\n{current_code}\n\n"
            f"Previous Thought:\n{current_thought}\n\n"
            f"Analyze the current code and provide an updated thought process, followed by the refined code."
        )
        
        # Format conversation using the tokenizer's chat template
        messages = [{"role": "user", "content": user_msg}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate next step
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=1024,
            verbose=False
        )
        
        print(f"Raw Output:\n{response}\n")
        
        # Parse output using robust multi-tier extractor
        current_thought, current_code = extract_thought_and_code(response)
            
        print(f"Extracted Thought:\n{current_thought}\n")
        print(f"Extracted Code Update:\n{current_code}\n")
            
    print("\n=== FINAL RESULT ===")
    print(current_code)
    return current_code
