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

def main(input_file: str, output_file: str):
    print(f"Reading from {input_file}...")
    
    formatted_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record['prompt']
            traj = record['trajectory']
            
            draft = traj['draft']
            critique = traj['critique']
            final_code = traj['final_code']
            
            # Step 1: Initial state to Naive Draft
            # Model must output an initial thought (we'll just use a generic one or the draft itself) and then the draft.
            # To simplify, we'll train it to do Draft -> Critique -> Final.
            
            # Sub-trajectory 1: Input -> Thought -> Draft
            # We don't have an initial thought from the Teacher, so let's simulate a generic thought for Draft 1.
            z_initial = "I need to write an initial naive solution to this problem."
            user_msg_1 = format_prompt(prompt, "", "")
            assistant_msg_1 = format_response(z_initial, draft)
            
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": user_msg_1},
                    {"role": "assistant", "content": assistant_msg_1}
                ]
            })
            
            # Sub-trajectory 2: Input + Draft + Old Thought -> Critique -> Final Code
            user_msg_2 = format_prompt(prompt, draft, z_initial)
            assistant_msg_2 = format_response(critique, final_code)
            
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": user_msg_2},
                    {"role": "assistant", "content": assistant_msg_2}
                ]
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully formatted {len(formatted_data)} conversational turns.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_humaneval.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="data/formatted_train.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    main(args.input, args.output)
