import pandas as pd
import random
import llm
import analyze_results as ar
import argparse
import sys
import os
from generate_test_cases import run_generate_new_tests
import shutil

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="OpenAI GPT API model name")
parser.add_argument("-p", "--num_prompts", help="Number of prompts to test")
parser.add_argument("-n", "--num_runs", help="Number of runs for prompt")
parser.add_argument("-t", "--test_type", help="Type of test to run: new or original")
parser.add_argument("-d", "--delta_method", help="Method for generating delta: permutations or combinations")
args = parser.parse_args()

args_dict = {
    'model': args.model,
    'num_prompts': int(args.num_prompts),
    'num_runs': int(args.num_runs),
    'test_type': args.test_type,
    'delta_method': args.delta_method
}

# Original JSON file
df_orig = pd.read_json('mbpp.jsonl', lines=True)

# Check if the folder 'generated_code_files' exists and delete if it does
print('Deleting generated_code_files folder...')
if os.path.exists('generated_code_files'):
    shutil.rmtree('generated_code_files')

# Check if the 'mbpp_new_test_cases.jsonl' exists
if not os.path.exists('mbpp_new_test_cases.jsonl'):
    print('Generating new test cases...')
    df = run_generate_new_tests(df_orig)
    df.to_json('mbpp_new_test_cases.jsonl', orient='records', lines=True)
else:
    print('New test cases already exist. Skipping generation...')
    df = pd.read_json('mbpp_new_test_cases.jsonl', lines=True)

# pick random prompt numbers to test
counter = 0
prompt_numbers = []
while counter < args_dict['num_prompts']:
    prompt_number = random.randint(0, len(df) - 1)
    if prompt_number not in prompt_numbers and df['new_test_list'][prompt_number] != 'Error generating new test cases':
        prompt_numbers.append(prompt_number)
        counter += 1

print('Prompt numbers to test:', prompt_numbers)

# Call the function from llm.py with the necessary arguments
print("Running llm tests...")
for prompt_number in prompt_numbers:
    llm.run_llm_tests(args_dict['model'], prompt_number, args_dict['num_runs'], args_dict['test_type'], args_dict['delta_method'], df)

# run analysis
print("Running analysis...")
ar.run_analysis()