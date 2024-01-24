import ast
import itertools
from . import general_adapter

def parse_function_inputs(input_str):
    # Get the substring before the first '=='
    input_str = input_str.split('==')[0].strip()

    # Extract the substring inside the outermost parentheses
    start_idx = input_str.find('(')
    end_idx = input_str.rfind(')')
    params_str = input_str[start_idx + 1:end_idx]

    # Use ast.parse to safely evaluate the structure and extract parameters
    tree = ast.parse(f"f({params_str})")

    # Extract the arguments from the function call
    args = tree.body[0].value.args
    
    # Convert the AST nodes back to Python objects
    inputs = [ast.literal_eval(arg) for arg in args]

    return inputs

def extract_mbpp_plus_examples(prompt, start_word):
    prompt = prompt.replace('"""', '')
    start_pos = prompt.find(start_word)
    examples = prompt[start_pos:].strip()
    return examples

def extract_mbpp_plus_docstring(prompt, stop_word):
    prompt = prompt.replace('"""', '')
    stop_pos = prompt.find(stop_word)
    docstring = prompt[:stop_pos].strip()
    return docstring

def extract_mbpp_plus_test_list(entry_point, plus_input, expected_output):
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
        df = df[['prompt', 'entry_point', 'plus_input', 'plus']].copy()
        plus_input = df.iloc[prompt_index]['plus_input']
        expected_output = df.iloc[prompt_index]['plus']
        prompt = str(df.iloc[prompt_index]['prompt'])
        entry_point = str(df.iloc[prompt_index]['entry_point'])
    else:
        df = df[['text', 'code', 'test_list']].copy()
        docstring = str(df.iloc[prompt_index]['text'])
        code = str(df.iloc[prompt_index]['code'])
        function_header = str(general_adapter.extract_function_header(code))

    # Extracting and ensuring the data types
    docstring = extract_mbpp_plus_docstring(prompt, 'assert')
    examples = extract_mbpp_plus_examples(prompt, 'assert')
    if test_type == 'evalplus':
        test_list = extract_mbpp_plus_test_list(entry_point, plus_input, expected_output)
    else:
        test_list = df.iloc[prompt_index]['test_list']

    # Define delta components as a dictionary
    delta_components = {
        'docstring': docstring,
        'examples': examples
    }

    # Choose between permutations and combinations
    delta_generator = itertools.permutations if delta_method == 'permutations' else itertools.combinations

    # Generate all permutations or combinations of the deltas
    delta_elements = ['docstring', 'examples']
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