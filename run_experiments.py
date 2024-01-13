import pandas as pd
import random
from llm import *
#import analyze_results as ar
import argparse
import sys
import os
from generate_test_cases import run_generate_new_tests
from delta_loader import *
import shutil
import logging
import models

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default='hf_incoder_1B',help="OpenAI GPT API model name", type=str)
parser.add_argument("-d", "--dataset", default="HumanEval",help="Dataset", choices=['HumanEval', 'MBPP'])
parser.add_argument("-p", "--num_prompts", default=1, help="Number of prompts to test", type=int)
parser.add_argument("-n", "--num_runs", default=1, help="Number of runs for prompt", type=int)
parser.add_argument("-t", "--test_type", default='original', help="Type of test to run: new or original", choices=['new', 'original'])
args = parser.parse_args()

# args_dict = {
#     'model': args.model,
#     'num_prompts': args.num_prompts,
#     'num_runs': args.num_runs,
#     'test_type': args.test_type
# }

logger.info(locals()['args'])

if args.dataset == "MBPP":
    all_deltas, test_cases = get_mbpp_deltas('mbpp.jsonl', args.num_prompts, args.test_type)
elif args.dataset == "HumanEval":
    all_deltas, test_cases = get_humaneval_deltas("HumanEval.jsonl.gz", args.num_prompts, args.test_type)
else:
    raise NotImplementedError


logger.info("Running llm tests...")
#for prompt_number in prompt_numbers:
#    llm.run_llm_tests(args_dict['model'], prompt_number, args_dict['num_runs'], args_dict['test_type'], df)
final_results = []
#for idx, prompt_args in enumerate(df.to_dict(orient='records')):
    #prompt_result = llm.run_llm_tests(args.model, args.num_runs, args.test_type, args.dataset, idx, **prompt_args)
    #all_results.append(prompt_result)
for prompt_index, deltas in all_deltas.items():

    #output_directory = f'generated_code_files/prompt_{prompt_index}'
    output_directory = os.path.join(args.dataset, 'generated_code_files', f'prompt_{prompt_index}')
    os.makedirs(output_directory, exist_ok=True)

    #results = {f'delta_{i}': [] for i in range(len(deltas))}

    all_results = []
    for run_index in range(args.num_runs):
        #print(f"Run {run_index + 1} of {num_runs} for prompt {prompt_index}")
        for i, delta in enumerate(deltas):
            #print(f"Generating code for delta {i + 1} of {len(deltas)}")
            generated_output = models.generate_model_output(delta, args.model)
            if 'OpenAI' in args.model:
                generated_code = extract_python_code(generated_output)
            elif 'hf' in args.model:
                generated_code = generated_output
            else:
                raise NotImplementedError
            #file_name = f"{output_directory}/delta_{run_index}_{i}.py"
            file_name = os.path.join(output_directory, f'delta_{run_index}_{i}.py')
            with open(file_name, 'w') as file:
                file.write(generated_code)
            #print(f"Running test(s) for delta {i + 1} of {len(deltas)}")
            test_result, error_type = run_test_cases_for_file(file_name, test_cases[prompt_index])  # Updated to receive a tuple
            delta_key = f'delta_{i}'
            #results[delta_key].append((test_result, error_type))  # Store as a tuple
            
            #pass_fail = 'Pass' if test_result == 'Pass' else 'Fail'
            all_results.append({
                'prompt': prompt_index,
                'Delta': delta_key,
                'Pass/Fail': test_result,
                'Error Type': error_type,
                'Run Index': run_index + 1,
                'Code': generated_code
            })
    
    final_results.append(all_results)

# Analyze the results
logger.info("Running analysis...")
#ar.run_analysis()