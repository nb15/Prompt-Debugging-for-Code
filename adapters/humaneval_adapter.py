from . import general_adapter
import itertools
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

def extract_humaneval_examples(code, function_header, start_words):
    text = code.split(function_header)[1].strip()
    examples_text = ""
    recording = False

    for line in text.split('\n'):
        if any(start_word in line for start_word in start_words):
            recording = True  
        elif recording and (line.strip() == '' or line.strip().startswith('"""')):
            break
        if recording:
            examples_text += line + '\n'
    return examples_text.strip()

def extract_humaneval_docstring(code, function_header, stop_words):
    text = code.split(function_header)[1].strip()
    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0]
    return text.strip().replace('"', '')

def extract_humaneval_test_list(entry_point, plus_input, expected_output):
    def prepare_input(inp):
        return ', '.join([str(i) for i in inp])
    test_list = [f'assert {entry_point}({prepare_input(i)}) == {str(j)}' for i,j in zip(plus_input, expected_output)]
    return test_list

def generate_deltas(df, prompt_index, delta_method):
    """
    Generate deltas based on the provided DataFrame, prompt index, and delta method.

    :param df: DataFrame containing the necessary data.
    :param prompt_index: The index of the prompt in the DataFrame.
    :param delta_method: Method for generating deltas ('permutations' or 'combinations').
    :return: A tuple containing the list of deltas and a dictionary with delta components info.
    """
    df = df[['prompt', 'entry_point', 'test', 'plus_input', 'plus']].copy()
    plus_input = df.iloc[prompt_index]['plus_input']
    expected_output = df.iloc[prompt_index]['plus']

    # Extracting and ensuring the data types
    prompt = str(df.iloc[prompt_index]['prompt'])
    entry_point = str(df.iloc[prompt_index]['entry_point'])

    function_header = str(general_adapter.extract_function_header(prompt, entry_point))
    docstring = extract_humaneval_docstring(prompt, function_header, ['Example', 'example', 'For example', 'For Example', '>>>', '>>', f'\n{entry_point}'])
    examples = extract_humaneval_examples(prompt, function_header, ['Example', 'example', 'For example', 'For Example', '>>>', '>>', f'\n{entry_point}'])
    #test_list = extract_humaneval_test_list(entry_point, plus_input, expected_output)
    nomalized_function_header = function_header.replace(entry_point, 'func')

    return [f'{prompt}',
            f'{function_header}\n{examples}',
            f'{docstring}\nCreate a function named {entry_point}\n{examples}',
            f'{nomalized_function_header}\n{docstring}'
            f'{docstring}\n{examples}\n{function_header}',
            f'{docstring}\n{function_header}\n{examples}',
        ]

    # Define delta components as a dictionary
    delta_components = {
        'docstring': docstring,
        'function_header': function_header,
        'examples': examples
    }

    # Choose between permutations and combinations
    delta_generator = itertools.permutations if delta_method == 'permutations' else itertools.combinations

    # Generate all permutations or combinations of the deltas
    delta_elements = ['docstring', 'function_header', 'examples']
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