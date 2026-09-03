import json
import argparse
import os

def format_prompt(x: str, y: str, z_old: str):
    """
    Format the user prompt for the recursive model.
    """
    return f"Task:\n{x}\n\nCurrent Code:\n{y}\n\nPrevious Thought:\n{z_old}\n\nAnalyze the current code and provide an updated thought process, followed by the refined code."

def format_response(z_new: str, y_new: str):
    """
    Format the assistant response.
    """
    return f"<thought>\n{z_new}\n</thought>\n<code_update>\n{y_new}\n</code_update>"

def main(input_file: str, discrete_output: str, continuous_output: str):
    print(f"Reading synthetic trajectories from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: input file {input_file} does not exist.")
        return
        
    discrete_data = []
    continuous_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record['prompt']
            traj = record.get('trajectory', {})
            
            draft = traj.get('draft', '')
            critique = traj.get('critique', '')
            final_code = traj.get('final_code', '')
            
            if not final_code:
                continue
            
            # 1. Discrete format: Multi-turn prompt loops with <thought> and <code_update>
            z_initial = "I need to write an initial naive solution to this problem."
            user_msg_1 = format_prompt(prompt, "", "")
            assistant_msg_1 = format_response(z_initial, draft if draft else final_code)
            
            discrete_data.append({
                "messages": [
                    {"role": "user", "content": user_msg_1},
                    {"role": "assistant", "content": assistant_msg_1}
                ]
            })
            
            if draft and critique:
                user_msg_2 = format_prompt(prompt, draft, z_initial)
                assistant_msg_2 = format_response(critique, final_code)
                discrete_data.append({
                    "messages": [
                        {"role": "user", "content": user_msg_2},
                        {"role": "assistant", "content": assistant_msg_2}
                    ]
                })
                
            # 2. Continuous latent format: Direct Prompt -> Solution mapping
            # (Reasoning is delegated to latent recurrence, not text thoughts)
            continuous_data.append({
                "task_id": record.get("task_id", ""),
                "prompt": prompt,
                "solution": final_code
            })

    # Save discrete dataset
    os.makedirs(os.path.dirname(discrete_output), exist_ok=True)
    with open(discrete_output, 'w') as f:
        for item in discrete_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(discrete_data)} discrete conversational samples to {discrete_output}")
    
    # Save continuous dataset
    os.makedirs(os.path.dirname(continuous_output), exist_ok=True)
    with open(continuous_output, 'w') as f:
        for item in continuous_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(continuous_data)} clean continuous latent samples to {continuous_output}")
    
    # Also update default data/train.jsonl with continuous_data if output is standard
    train_fallback = os.path.join(os.path.dirname(continuous_output), "continuous_train.jsonl")
    print("Formatting complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_humaneval.jsonl", help="Input JSONL file")
    parser.add_argument("--discrete-output", default="data/discrete_train.jsonl", help="Output JSONL for discrete LoRA training")
    parser.add_argument("--continuous-output", default="data/continuous_train.jsonl", help="Output JSONL for continuous latent training")
    args = parser.parse_args()
    main(args.input, args.discrete_output, args.continuous_output)
