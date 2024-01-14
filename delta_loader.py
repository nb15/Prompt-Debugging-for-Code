import pandas as pd
import os
import shutil
import random
from generate_test_cases import run_generate_new_tests
from humaneval_utils import read_problems
from llm import *

def get_mbpp_dataframe(data_dir='mbpp.jsonl', num_prompts=-1):
    # Original JSON file
    df_orig = pd.read_json(data_dir, lines=True)

    # Check if the folder 'generated_code_files' exists and delete if it does
    #logger.info('Deleting generated_code_files folder...')
    if os.path.exists('generated_code_files'):
        shutil.rmtree('generated_code_files')

    # Check if the 'mbpp_new_test_cases.jsonl' exists
    if not os.path.exists('mbpp_new_test_cases.jsonl'):
        #logger.info('Generating new test cases...')
        df = run_generate_new_tests(df_orig)
        df.to_json('mbpp_new_test_cases.jsonl', orient='records', lines=True)
    else:
        #logger.info('New test cases already exist. Skipping generation...')
        df = pd.read_json('mbpp_new_test_cases.jsonl', lines=True)
    
    df = df.reset_index(drop=True)
    
    if num_prompts!=-1:
        counter = 0
        prompt_numbers = []
        while counter < num_prompts:
            prompt_number = random.randint(0, len(df) - 1)
            if prompt_number not in prompt_numbers and df['new_test_list'][prompt_number] != 'Error generating new test cases':
                prompt_numbers.append(prompt_number)
                counter += 1
        df = df.loc[prompt_numbers].reset_index(drop=True)
    return df


def get_mbpp_deltas(data_dir='mbpp.jsonl', num_prompts=-1, test_type = 'Original'):
    df = get_mbpp_dataframe(data_dir, num_prompts)    
    all_deltas, test_cases = {}, ()

    for idx, prompt_args in enumerate(df.to_dict(orient='records')):
        all_deltas[idx] = process_mbpp_deltas(test_type=test_type, **prompt_args)
        test_cases[idx] = all_deltas[idx][-1]

    return all_deltas, test_cases


def get_humaneval_deltas(data_dir="HumanEval.jsonl.gz", num_prompts=-1, test_type = 'original', evalplus_exec=False):
    if test_type == 'evalplus':
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
    else:
        problems = read_problems(data_dir)

    problems = [i for _, i in problems.items()]
    if num_prompts!=-1:
        problems = random.sample(problems, num_prompts)

    all_deltas = {prompt['task_id'].split('/')[1]: process_humaneval_deltas(test_type=test_type, **prompt) for prompt in problems}
    
    if evalplus_exec:
        return all_deltas, None

    if test_type == 'evalplus':
        from evalplus.data import get_human_eval_plus_hash
        from evalplus.evaluate import get_groundtruth

        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(problems, dataset_hash, [])

        test_cases = {prompt['task_id'].split('/')[1]: process_humaneval_plus_testcases(
                                                            expected_output = expected_output[prompt['task_id']['plus']],
                                                            **prompt) for prompt in problems}
    else:    
        test_cases = {prompt['task_id'].split('/')[1]: process_humaneval_testcases(**prompt) for prompt in problems}

    return all_deltas, test_cases


