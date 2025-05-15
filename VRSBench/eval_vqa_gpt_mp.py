import argparse
import json
import os
from tqdm import tqdm
from openai import OpenAI
import numpy as np
from multiprocessing import Pool, cpu_count

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'
client = OpenAI()


def check_match_with_gpt(question, ground_truth, predicted):
    prompt = f"Question: {question}\nGround Truth Answer: {ground_truth}\nPredicted Answer: {predicted}\nDoes the predicted answer match the ground truth? Answer 1 for match and 0 for not match. Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def process_sample(qa):
    question = qa['question']
    ground_truth = qa['gt'].lower()
    predicted = qa['answer'].lower()

    if ground_truth in predicted:
        match_result = '1'
    elif ground_truth in ['yes', 'no'] + list(map(str, range(100))):
        match_result = '1' if ground_truth == predicted else '0'
    elif 'correct' not in qa or qa['correct'] not in ['1', '0']:
        try:
            match_result = check_match_with_gpt(question, ground_truth, predicted)
        except Exception as e:
            match_result = '0'  # Fallback on failure
    else:
        match_result = qa['correct']

    return {
        'image': qa['image'],
        # "type": qa['type'],
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": match_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the file logging final accuracy.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel processes to use.")
    args = parser.parse_args()

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


    with open(args.data_path, 'r') as f:
        qa_list = json.load(f)['data']

    print('number of anns:', len(qa_list))
    # assert len(qa_list) ==37409
    
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_sample, qa_list), total=len(qa_list)))

    # Save results
    with open(out_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Calculate accuracy
    correct = sum([int(result['correct']) for result in results if result['correct'] in ['1', '0']])
    acc = correct / len(results)
    print(f"Correct: {correct}/{len(results)} = {acc:.4f}")

    with open(args.output_file, 'a') as f:
        f.write(f"{args.data_path}, {acc:.4f}\n")
