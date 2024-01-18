import ast
import general_adapter

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