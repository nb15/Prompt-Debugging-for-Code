import pandas as pd
import traceback
import subprocess
from adapters import mbpp_adapter
import value_generator

# List of prompts that are problematic for parsing
blacklisted_prompts = [366, 485, 532, 600, 768, 911, 926]

def generate_test_case_values(test_list):
    """
    Generate test case values for a given list of tests.

    :param test_list: List of tests for which values need to be generated.
    :return: Dynamically regenerated values for the test cases.
    """
    test_inputs = [mbpp_adapter.parse_function_inputs(test) for test in test_list]
    return value_generator.regenerate_values_dynamically(test_inputs)

def generate_test_case(test_case_values, example_test_case, code_solution):
    """
    Generate a test case based on the provided values, test case, and code solution.

    :param test_case_values: Values to be used in the test case.
    :param example_test_case: The example test case string.
    :param code_solution: The code solution as a string.
    :return: Formatted test case assertion as a string.
    """
    test_case_values = [f"'{x}'" if isinstance(x, str) else str(x) for x in test_case_values]
    function_name = example_test_case.split('assert')[1].split('(')[0].strip()

    print_statement = f"print({function_name}({', '.join(test_case_values)}))"

    with open('test_case.py', 'w') as file:
        file.write(code_solution + '\n')
        file.write(print_statement)

    output = subprocess.run(['python', 'test_case.py'], capture_output=True).stdout.decode().strip()

    test_case = f"assert {function_name}({', '.join(test_case_values)}) == {output}"
    return test_case

def generate_new_tests(df):
    """
    Generate new tests for a dataframe containing code and test cases.

    :param df: DataFrame with test cases and code.
    :return: DataFrame updated with new tests.
    """
    if 'new_test_list' not in df.columns:
        df['new_test_list'] = [[] for _ in range(len(df))]

    for i, (test_list, code) in enumerate(zip(df['test_list'], df['code'])):
        if i in blacklisted_prompts:
            continue

        print(f'Generating new test cases for prompt {i+1}')

        try:
            test_case_values = generate_test_case_values(test_list)
        except:
            print('Error generating new test case values')
            traceback.print_exc()
            continue
        
        new_test_list = []
        for idx, test in enumerate(test_list):
            try:
                test_case = generate_test_case(test_case_values[idx], test, code)
                new_test_list.append(test_case)
            except:
                print('Error generating new test case')
                traceback.print_exc()

        df.at[i, 'new_test_list'] = new_test_list

    return df

def run_generate_new_tests(df):
    """
    Wrapper function to run the process of generating new tests.

    :param df: DataFrame with code and test cases.
    :return: DataFrame updated with new tests.
    """
    return generate_new_tests(df)