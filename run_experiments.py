import pandas as pd
import random
import llm
from analyze_results import run_analysis
import argparse
import os
import shutil
import json
from tqdm import tqdm
from adapters import humaneval_adapter
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="LLM model name")
parser.add_argument("-d", "--dataset", help="Dataset: mbpp or humaneval")
parser.add_argument("-p", "--num_prompts", help="Number of prompts to test or list of prompt numbers")
parser.add_argument("-n", "--num_runs", help="Number of runs for prompt")
parser.add_argument("-g", "--delta_grouping", help="Grouping for generating delta: permutations or combinations")
args = parser.parse_args()

# Function to parse the num_prompts argument
def parse_num_prompts(arg):
    if arg.startswith('[') and arg.endswith(']'):
        return json.loads(arg)
    else:
        return int(arg)

args_dict = {
    'model': args.model,
    'dataset': args.dataset,
    'num_prompts': parse_num_prompts(args.num_prompts),
    'num_runs': int(args.num_runs),
    'delta_method': args.delta_grouping
}

# Check if the folder 'generated_code_files' exists and delete if it does
print('Deleting generated_code_files folder...')
if os.path.exists('generated_code_files'):
    shutil.rmtree('generated_code_files')

# Get dataset problems and expected output
if args_dict['dataset'] == 'mbpp':
     problems = get_mbpp_plus()
     dataset_hash = get_mbpp_plus_hash()
     expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)
elif args_dict['dataset'] == 'humaneval':
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
if isinstance(args_dict['num_prompts'], list):
    # Use the provided list of prompt numbers
    prompt_numbers = args_dict['num_prompts']
else:
    # Pick random prompt numbers to test
    prompt_numbers = []
    while len(prompt_numbers) < args_dict['num_prompts']:
        prompt_number = random.randint(0, len(df) - 1)
        if prompt_number not in prompt_numbers:
            prompt_numbers.append(prompt_number)

print('Prompt numbers to test:', prompt_numbers)

# Call the function from llm.py with the necessary arguments
print("Running llm tests...")
for prompt_number in tqdm(prompt_numbers, desc="Prompts completed"):
    llm.run_llm_tests(args_dict['model'], args_dict['dataset'], prompt_number, args_dict['num_runs'], args_dict['delta_method'], df)

# run analysis
print("Running analysis...")
run_analysis(len(prompt_numbers), args_dict['num_runs'])