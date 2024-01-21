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

def process_mbpp_deltas(test_type, text, code, test_list, new_test_list, **kwargs):
    
    function_header = general_adapter.extract_function_header(code=code)
    test_list = new_test_list if test_type == 'new' else test_list

    deltas = [
        f"{text}\n{function_header}\n{test_list}",
        text,
        f"{text}\n{function_header}",
        function_header,
        f"{function_header}\n{test_list}",
        test_list
    ]
    return deltas

def generate_deltas(df, prompt_index, delta_method, test_type):
    """
    Generate deltas based on the provided DataFrame, prompt index, and delta method.

    :param df: DataFrame containing the necessary data.
    :param prompt_index: The index of the prompt in the DataFrame.
    :param delta_method: Method for generating deltas ('permutations' or 'combinations').
    :return: A tuple containing the list of deltas and a dictionary with delta components info.
    """
    df = df[['text', 'code', 'test_list', 'new_test_list']].copy()
    df['function_header'] = df['code'].apply(general_adapter.extract_function_header)

    # Extracting and ensuring the data types
    docstring = str(df.iloc[prompt_index]['text'])
    code = str(df.iloc[prompt_index]['code'])
    function_header = str(general_adapter.extract_function_header(code))
    test_list = df.iloc[prompt_index]['new_test_list'] if test_type == 'new' else df.iloc[prompt_index]['test_list']

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
