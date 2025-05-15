import argparse
import json
import os
from tqdm import tqdm
from openai import OpenAI
import sys
import multiprocessing as mp

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'
client = OpenAI()

from clair import clair

def process_example(qa):
    ground_truth = qa['gt']
    predicted = qa['answer']
    clair_score = clair([predicted], [ground_truth], model='gpt-4o-mini')
    
    return {
        'image': qa['image'],
        "ground_truth": ground_truth,
        "predicted": predicted,
        "clair": clair_score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the CSV file for logging results.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers.")
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        qa_list = json.load(f)['data']
    
    # assert len(qa_list) ==9350

    if 'vhm' in args.data_path.lower():
        out_dir = 'outputs_gpt_vhm'
    elif 'sky' in args.data_path.lower():
        out_dir = 'outputs_gpt_sky'
    elif 'falcon' in  args.data_path.lower():
        out_dir = 'outputs_gpt_falcon'
    else:
        assert 'not supported'

    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.basename(args.data_path).split('.')[0] + '_gpt.json'
    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path): 
        print(f'{out_path} already exists')
        sys.exit(0)  # Exit cleanly

    with mp.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_example, qa_list), total=len(qa_list)))

    # Write results to output file
    with open(out_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Compute final accuracy
    correct = sum([float(result['clair']) for result in results])
    acc = correct / len(results)
    print(f"Correct: {correct}/{len(results)} = {acc:.4f}")

    with open(args.output_file, 'a') as f:
        f.write(f"{args.data_path}, {acc:.4f}\n")
