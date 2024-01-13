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

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default='hf_incoder_1B',help="OpenAI GPT API model name", type=str)
parser.add_argument("-d", "--dataset", default="HumanEval",help="Dataset", choices=['HumanEval', 'MBPP'])
parser.add_argument("-p", "--num_prompts", default=1, help="Number of prompts to test", type=int)
parser.add_argument("-n", "--num_runs", default=1, help="Number of runs for prompt", type=int)
parser.add_argument("-t", "--test_type", default='original', help="Type of test to run: new or original", choices=['new', 'original', 'evalplus'])
## 'new' for MBPP is not working
args = parser.parse_args()


logger.info(locals()['args'])

if args.dataset == "MBPP":
    all_deltas, test_cases = get_mbpp_deltas('mbpp.jsonl', args.num_prompts, args.test_type)
elif args.dataset == "HumanEval":
    all_deltas, test_cases = get_humaneval_deltas("HumanEval.jsonl.gz", args.num_prompts, args.test_type)
else:
    raise NotImplementedError


progress_bar = tqdm(range(len(all_deltas)*args.num_runs))

logger.info("Running llm tests...")
final_results = []
for prompt_index, deltas in all_deltas.items():

    output_directory = os.path.join(args.dataset, 'generated_code_files', f'prompt_{prompt_index}')
    os.makedirs(output_directory, exist_ok=True)

    all_results = []
    for run_index in range(args.num_runs):
        for i, delta in enumerate(deltas):
            generated_output = models.generate_model_output(delta, args.model)
            if 'OpenAI' in args.model:
                generated_code = extract_python_code(generated_output)
            elif 'hf' in args.model:
                generated_code = generated_output
            else:
                raise NotImplementedError
            file_name = os.path.join(output_directory, f'delta_{run_index}_{i}.py')
            with open(file_name, 'w') as file:
                file.write(generated_code)
            test_result, error_type = run_test_cases_for_file(file_name, test_cases[prompt_index])  # Updated to receive a tuple
            delta_key = f'delta_{i}'
            
            all_results.append({
                'prompt': prompt_index,
                'Delta': delta_key,
                'Pass/Fail': test_result,
                'Error Type': error_type,
                'Run Index': run_index + 1,
                'Code': generated_code
            })
            progress_bar.update(1)
    
    final_results.append(all_results)

logger.info("Running analysis...")
#ar.run_analysis()