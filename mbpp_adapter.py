import ast

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