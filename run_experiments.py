import pandas as pd
import random
import llm
#from analyze_results import run_analysis
import argparse
import os
import shutil
import json
from tqdm import tqdm
from models import get_hf_model
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash, write_jsonl
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default='hf_starcoderbase_1B',help="LLM model name")
parser.add_argument("-d", "--dataset", default='humaneval', help="Dataset", choices=['humaneval', 'mbpp'])
parser.add_argument("-p", "--num_prompts", default=-1, help="Number of prompts to test or list of prompt numbers")
parser.add_argument("-n", "--num_runs", default=1, help="Number of runs for prompt")
parser.add_argument("-g", "--delta_grouping", default=None, help="Grouping for generating delta: permutations or combinations")
parser.add_argument("-e", "--evaluation", default='evalplus', help="Evaluate using evalplus or runtime")


parser.add_argument("-t", "--temperature", type=float, default=0.8)
parser.add_argument("--max_len", type=int, default=2048)
parser.add_argument("--greedy_decode", type=bool, default=True)
parser.add_argument("--decoding_style", type=str, default='sampling')
parser.add_argument("--save_embds", default=False, type=bool)

args = parser.parse_args()

# Function to parse the num_prompts argument
def parse_num_prompts(arg):
    if arg.startswith('[') and arg.endswith(']'):
        return json.loads(arg)
    else:
        return int(arg)


# Check if the folder 'generated_code_files' exists and delete if it does
print('Deleting generated_code_files folder...')
if os.path.exists('generated_code_files'):
    shutil.rmtree('generated_code_files')

# Get dataset problems and expected output
if args.dataset == 'mbpp':
     problems = get_mbpp_plus()
     dataset_hash = get_mbpp_plus_hash()
     expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)
elif args.dataset == 'humaneval':
    problems = get_human_eval_plus()
    dataset_hash = get_human_eval_plus_hash()
    expected_output = get_groundtruth(problems, dataset_hash, [])

# Add expected output to problems
for problem in problems:
    expected_output_keys = [*expected_output[problem]]
    for key in expected_output_keys:
        problems[problem][key] = expected_output[problem][key]

# Create df from problems
df = pd.DataFrame.from_dict(problems, orient='index')
df = df.reset_index(drop=True)

# Determine prompt numbers to test
if args.num_prompts == -1:
    prompt_numbers = list(range(len(problems)))
elif isinstance(args.num_prompts, list):
    # Use the provided list of prompt numbers
    prompt_numbers = args.num_prompts
else:
    # Pick random prompt numbers to test
    prompt_numbers = []
    while len(prompt_numbers) < args.num_prompts:
        prompt_number = random.randint(0, len(df) - 1)
        if prompt_number not in prompt_numbers:
            prompt_numbers.append(prompt_number)

#print('Prompt numbers to test:', prompt_numbers)


if 'hf' in args.model:
    model, tokenizer, generation_config = get_hf_model(args.model,
                                    args.temperature,
                                    args.max_len,
                                    args.greedy_decode,
                                    args.decoding_style)

# Call the function from llm.py with the necessary arguments
print("Running llm tests...")
final_results = []

for prompt_number in tqdm(prompt_numbers, desc="Prompts completed"):
    if args.evaluation == 'runtime':
        llm.run_llm_tests(args.model, args.dataset, prompt_number, args.num_runs, args.delta_grouping, df)
    else:
        if args.save_embds:
            pass
        else:
            final_results+=llm.gen_hf_model_output(model, tokenizer, generation_config,
                                                args.dataset, prompt_number, args.num_runs, args.delta_grouping, df, args.max_len)

result_file = f'{args.dataset}_generated_code.jsonl'
write_jsonl(result_file, final_results)


# run analysis
#print("Running analysis...")
#run_analysis(len(prompt_numbers), args_dict['num_runs'], args_dict['evaluation'])