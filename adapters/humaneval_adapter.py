from . import general_adapter
import itertools
import pandas as pd

def extract_humaneval_docstring(code, function_header, stop_words):
    text = code.split(function_header)[1].strip()
    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0]
    return text.strip().replace('"', '')

def extract_humaneval_test_list(test, function_header):
    test_list = [i.strip() for i in test.split('\n') if 'assert' in i]
    # parse function header between 'def ' and '('
    function_name = function_header.split('def ')[1].split('(')[0]
    # replace 'candidate' with function name for every test in test_list
    test_list = [i.replace('candidate', function_name) for i in test_list]
    return test_list

def generate_deltas(df, prompt_index, delta_method, test_type):
    """
    Generate deltas based on the provided DataFrame, prompt index, and delta method.

    :param df: DataFrame containing the necessary data.
    :param prompt_index: The index of the prompt in the DataFrame.
    :param delta_method: Method for generating deltas ('permutations' or 'combinations').
    :return: A tuple containing the list of deltas and a dictionary with delta components info.
    """
    df = df[['prompt', 'entry_point', 'test']].copy()

    # Extracting and ensuring the data types
    prompt = str(df.iloc[prompt_index]['prompt'])
    entry_point = str(df.iloc[prompt_index]['entry_point'])
    test = str(df.iloc[prompt_index]['test'])

    function_header = str(general_adapter.extract_function_header(prompt, entry_point))
    docstring = extract_humaneval_docstring(prompt, function_header, ['Example', 'example', 'For example', 'For Example', '>>>', '>>', f'\n{entry_point}'])
    # test_list = None if test_type == 'new' else prompt.split(docstring)[1].strip()
    test_list = None if test_type == 'new' else extract_humaneval_test_list(test, function_header)

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