import pandas as pd
import random
from llm import *
# import llm
from analyze_results import run_analysis
import argparse
import os
from generate_test_cases import run_generate_new_tests
from delta_loader import *
import shutil
import json
import logging
import models
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="OpenAI GPT API model name")
parser.add_argument("-d", "--dataset", help="Dataset", choices=['humaneval', 'mbpp'])
parser.add_argument("-p", "--num_prompts", help="Number of prompts to test or list of prompt numbers")
parser.add_argument("-n", "--num_runs", help="Number of runs for prompt")
parser.add_argument("-t", "--test_type", help="Type of test to run: new or original")
parser.add_argument("-g", "--delta_group", help="Groupings for generating delta: permutations or combinations")

# TODO: do we need a version for MBPP?
#Evalplus Evaluation params
parser.add_argument("--evalplus-exec", default=True)
parser.add_argument("--samples", default=None)
parser.add_argument("--base-only", default=True)
parser.add_argument("--parallel", default=None, type=int)
parser.add_argument("--i-just-wanna-run", action="store_true")
parser.add_argument("--test-details", action="store_true")
parser.add_argument("--min-time-limit", default=0.2, type=float)
parser.add_argument("--gt-time-limit-factor", default=4.0, type=float)
parser.add_argument("--mini", action="store_true")

args = parser.parse_args()

logger.info(locals()['args'])

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
    'test_type': args.test_type,
    'delta_group': args.delta_groups
}

if args_dict['dataset'] == "mbpp":
    all_deltas, test_cases = get_mbpp_deltas('mbpp.jsonl', args.num_prompts, args.test_type)
elif args_dict['dataset'] == "humaneval":
    all_deltas, test_cases = get_humaneval_deltas("HumanEval.jsonl.gz", args.num_prompts, args.test_type, args.evalplus_exec)
else:
    raise NotImplementedError

progress_bar = tqdm(range(len(all_deltas)*args.num_runs*6))
    
# # Original JSON file
# df_orig = pd.read_json('mbpp.jsonl', lines=True)

# Check if the folder 'generated_code_files' exists and delete if it does
print('Deleting generated_code_files folder...')
if os.path.exists('generated_code_files'):
    shutil.rmtree('generated_code_files')

# # Check if the 'mbpp_new_test_cases.jsonl' exists
# if not os.path.exists('mbpp_new_test_cases.jsonl'):
#     print('Generating new test cases...')
#     df = run_generate_new_tests(df_orig)
#     df.to_json('mbpp_new_test_cases.jsonl', orient='records', lines=True)
# else:
#     print('New test cases already exist. Skipping generation...')
#     df = pd.read_json('mbpp_new_test_cases.jsonl', lines=True)

# ===== TODO: combine logic =====
# logger.info("Running llm tests...")
# final_results = []
# for prompt_index, deltas in all_deltas.items():

#     if not args.evalplus_exec:
#         output_directory = os.path.join(args.dataset, 'generated_code_files', f'prompt_{prompt_index}')
#         os.makedirs(output_directory, exist_ok=True)
#         all_results = []

#     for run_index in range(args.num_runs):
#         for i, delta in enumerate(deltas[:1]):
#             generated_output = models.generate_model_output(delta, args.model)
#             if 'OpenAI' in args.model:
#                 generated_code = extract_python_code(generated_output)
#             elif 'hf' in args.model:
#                 #generated_code = extract_python_code_hf(generated_output)
#                 generated_code = generated_output
#             else:
#                 raise NotImplementedError
            
#             if args.evalplus_exec:
#                 final_results.append(dict(task_id = f'HumanEval/{prompt_index}', completion = generated_code))
#             else:
#                 file_name = os.path.join(output_directory, f'delta_{run_index}_{i}.py')
#                 with open(file_name, 'w') as file:
#                     file.write(generated_code)
#                 test_result, error_type = run_test_cases_for_file(file_name, test_cases[prompt_index])  # Updated to receive a tuple
#                 delta_key = f'delta_{i}'
            
#                 all_results.append({
#                     'prompt': prompt_index,
#                     'Delta': delta_key,
#                     'Pass/Fail': test_result,
#                     'Error Type': error_type,
#                     'Run Index': run_index + 1,
#                     'Code': generated_code
#                 })
#             progress_bar.update(1)
    
#     if not args.evalplus_exec:
#         final_results.append(all_results)
# 
== combine logic =====

# Determine prompt numbers to test
if isinstance(args_dict['num_prompts'], list):
    # Use the provided list of prompt numbers
    prompt_numbers = args_dict['num_prompts']
else:
    # Pick random prompt numbers to test
    prompt_numbers = []
    while len(prompt_numbers) < args_dict['num_prompts']:
        prompt_number = random.randint(0, len(df) - 1)
        if prompt_number not in prompt_numbers and df['new_test_list'][prompt_number] != 'Error generating new test cases':
            prompt_numbers.append(prompt_number)

print('Prompt numbers to test:', prompt_numbers)

# Call the function from llm.py with the necessary arguments
print("Running llm tests...")
for prompt_number in prompt_numbers:
    llm.run_llm_tests(args_dict['model'], prompt_number, args_dict['num_runs'], args_dict['test_type'], args_dict['delta_group'], df)
    print('Prompts completed:', round((prompt_numbers.index(prompt_number) + 1) / len(prompt_numbers) * 100, 2), '%')

# run analysis
print("Running analysis...")
run_analysis(len(prompt_numbers), args_dict['num_runs'])
