import pandas as pd
import os
import importlib.util
import timeout_decorator
import models

# Dictionary for storing arguments
args_dict = {
    'model': '',
    'prompt': '',
    'num_runs': '',
    'test_type': ''
}

def extract_function_header(code):
    """
    Extract the function header from a block of code.

    :param code: A string representing a block of Python code.
    :return: The first line starting with 'def' indicating a function definition, or an empty string if not found.
    """
    for line in code.split('\n'):
        if line.strip().startswith('def'):
            return line
    return ''

def extract_python_code(gpt_output):
    """
    Extract Python code from a GPT output block.

    :param gpt_output: A string representing GPT output containing a Python code block.
    :return: Extracted Python code as a string.
    """
    code_block = []
    in_code_block = False
    for line in gpt_output.split('\n'):
        if line.strip() == '```python':
            in_code_block = True
        elif line.strip() == '```' and in_code_block:
            break
        elif in_code_block:
            code_block.append(line)
    return '\n'.join(code_block)

# Define a timeout duration for test cases
TEST_CASE_TIMEOUT = 30  # Example: 30 seconds

def run_single_test_case(module, test):
    """
    Run a single test case.

    :param module: The Python module in which the test will be executed.
    :param test: A string representing the test case to be executed.
    """
    exec(test, globals(), module.__dict__)

@timeout_decorator.timeout(TEST_CASE_TIMEOUT)
def run_test_case_with_timeout(module, test):
    """
    Execute a test case with a specified timeout.

    :param module: The Python module in which the test will be executed.
    :param test: A string representing the test case to be executed.
    """
    run_single_test_case(module, test)

def run_test_cases(module, test_list):
    """
    Run a list of test cases for a given module, handling timeouts and assertions.

    :param module: The Python module to test.
    :param test_list: A list of test case strings to be executed.
    :return: True if all tests pass, False otherwise.
    """
    for test in test_list:
        try:
            run_test_case_with_timeout(module, test)
        except AssertionError:
            return False
        except timeout_decorator.TimeoutError:
            print("Test case timed out. Moving to next test case.")
    return True

def run_test_cases_for_file(file_path, test_list):
    """
    Run test cases for a Python file.

    :param file_path: Path to the Python file to be tested.
    :param test_list: A list of test case strings to be executed.
    :return: A tuple with the test result ('Pass' or 'Fail') and an error type if applicable.
    """
    try:
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if run_test_cases(module, test_list):
            return ('Pass', None)
        else:
            # If the test cases run but fail, it's a semantic error
            return ('Fail', 'Semantic Error (Test Case)')
    except SyntaxError:
        return ('Fail', 'Syntax Error')
    except AssertionError:
        # If an assertion fails, it's considered a semantic error
        return ('Fail', 'Semantic Error (Assertion)')
    except timeout_decorator.TimeoutError:
        return ('Fail', 'Timeout')
    except MemoryError:
        return ('Fail', 'Resource Error')
    except ImportError:
        return ('Fail', 'Dependency Error')
    except EnvironmentError:
        return ('Fail', 'Environment Error')
    except Exception as e:
        return ('Fail', f'Runtime Error - {e.__class__.__name__}')

def main(df):
    """
    Main function to run tests on a DataFrame of code and test cases.

    :param df: A DataFrame containing code and test cases.
    """
    df = df[['text', 'code', 'test_list', 'new_test_list']].copy()
    df['function_header'] = df['code'].apply(extract_function_header)

    prompt_index = int(args_dict['prompt']) if args_dict['prompt'] else 0
    num_runs = int(args_dict['num_runs']) if args_dict['num_runs'] else 1

    output_directory = f'generated_code_files/prompt_{prompt_index}'
    os.makedirs(output_directory, exist_ok=True)

    docstring, code, function_header = df.iloc[prompt_index][['text', 'code', 'function_header']]
    test_list = df.iloc[prompt_index]['new_test_list'] if args_dict['test_type'] == 'new' else df.iloc[prompt_index]['test_list']

    deltas = [f"{docstring}\n{function_header}\n{test_list}", docstring, f"{docstring}\n{function_header}", function_header, f"{function_header}\n{test_list}", test_list]
    results = {f'delta_{i}': [] for i in range(len(deltas))}

    for run_index in range(num_runs):
        print(f"Run {run_index + 1} of {num_runs} for prompt {prompt_index}")
        for i, delta in enumerate(deltas):
            print(f"Generating code for delta {i + 1} of {len(deltas)}")
            generated_output = models.generate_model_output(delta, args_dict['model'])
            generated_code = extract_python_code(generated_output)
            file_name = f"{output_directory}/delta_{run_index}_{i}.py"
            with open(file_name, 'w') as file:
                file.write(generated_code)
            print(f"Running test(s) for delta {i + 1} of {len(deltas)}")
            results[f'delta_{i}'].append(run_test_cases_for_file(file_name, test_list))

    all_results = []
    for delta_key, test_results in results.items():
        for run_index, (test_result, error_type) in enumerate(test_results):
            file_name = f"{output_directory}/delta_{run_index}_{delta_key.split('_')[1]}.py"
            with open(file_name, 'r') as file:
                code = file.read()
            all_results.append({
                'Delta': delta_key,
                'Pass/Fail': 'Pass' if test_result == 'Pass' else 'Fail',
                'Error Type': error_type,
                'Run Index': run_index + 1,
                'Code': code
            })

    results_df = pd.DataFrame(all_results)
    results_df['Passed In All Runs'] = results_df.groupby('Delta')['Pass/Fail'].transform(lambda x: 'Yes' if all(x == 'Pass') else 'No')

    csv_filename = f'{output_directory}/results_prompt_{prompt_index}.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"CSV file created: {csv_filename}")

def run_llm_tests(model, prompt_num, num_runs, test_type, df):
    """
    Run the Language Learning Model (LLM) tests.

    :param model: The model to use for generating code.
    :param prompt_num: The prompt number to test.
    :param num_runs: Number of times to run the tests.
    :param test_type: The type of test to run ('new' or other).
    :param df: A DataFrame containing code and test cases.
    """
    args_dict.update({'model': model, 'prompt': prompt_num, 'num_runs': num_runs, 'test_type': test_type})
    main(df)

if __name__ == '__main__':
    pass