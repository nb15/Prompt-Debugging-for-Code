from . import general_adapter
import itertools
import pandas as pd
from typing import Iterable, Dict
import gzip
import json
import random

def read_problems(evalset_file: str) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def extract_humaneval_docstring(code, function_header, stop_words):
    text = code.split(function_header)[1].strip()
    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0]
    return text.strip().replace('"', '')

def extract_humaneval_test_list(test, entry_point):
    test_list = [i.strip() for i in test.split('\n') if 'assert' in i]
    test_list = [i.replace('candidate', entry_point) for i in test_list]
    return test_list

def extract_humaneval_plus_test_list(entry_point, plus_input, expected_output):
    def prepare_input(inp):
        return ', '.join([str(i) for i in inp])
    test_list = [f'assert {entry_point}({prepare_input(i)}) == {str(j)}' for i,j in zip(plus_input, expected_output)]
    return test_list

def generate_deltas(df, prompt_index, delta_method, test_type):
    """
    Generate deltas based on the provided DataFrame, prompt index, and delta method.

    :param df: DataFrame containing the necessary data.
    :param prompt_index: The index of the prompt in the DataFrame.
    :param delta_method: Method for generating deltas ('permutations' or 'combinations').
    :return: A tuple containing the list of deltas and a dictionary with delta components info.
    """
    if test_type == 'evalplus':
        df = df[['prompt', 'entry_point', 'test', 'plus_input', 'plus']].copy()
        plus_input = df.iloc[prompt_index]['plus_input']
        expected_output = df.iloc[prompt_index]['plus']
    else:
        df = df[['prompt', 'entry_point', 'test']].copy()
        test = str(df.iloc[prompt_index]['test'])

    # Extracting and ensuring the data types
    prompt = str(df.iloc[prompt_index]['prompt'])
    entry_point = str(df.iloc[prompt_index]['entry_point'])

    function_header = str(general_adapter.extract_function_header(prompt, entry_point))
    docstring = extract_humaneval_docstring(prompt, function_header, ['Example', 'example', 'For example', 'For Example', '>>>', '>>', f'\n{entry_point}'])
    if test_type == 'evalplus':
        test_list = extract_humaneval_plus_test_list(entry_point, plus_input, expected_output)
        test_list = random.sample(test_list, 5) # Randomly sample 5 test cases
    else:
        test_list = None if test_type == 'new' else extract_humaneval_test_list(test, entry_point)

    # Define delta components as a dictionary
    delta_components = {
        'docstring': docstring,
        'function_header': function_header,
        'test_list': str(test_list)
    }

    # Choose between permutations and combinations
    delta_generator = itertools.permutations if delta_method == 'permutations' else itertools.combinations

    # Generate all permutations or combinations of the deltas
    delta_elements = ['docstring', 'function_header', 'test_list']
    all_deltas = []
    for r in range(1, len(delta_elements) + 1):
        all_deltas.extend(delta_generator(delta_elements, r))

    deltas = []
    delta_components_info = {}  # To store components information
    for delta in all_deltas:
        delta_key = '\n'.join([delta_components[element] for element in delta])
        deltas.append(delta_key)
        delta_components_info[delta_key] = ', '.join(delta)  # Store the components for each delta

    return deltas, delta_components_info, test_list