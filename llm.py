import pandas as pd
import os
import importlib.util
import timeout_decorator  # Import the timeout_decorator package
import models

args_dict = {
    'model': '',
    'prompt': '',
    'num_runs': '',
    'test_type': ''
}

def extract_function_header(code, entry_point=None):
    lines = code.split('\n')
    for line in lines:
        if line.strip().startswith('def'):
            if entry_point:
                if entry_point in line:
                    return line
            else:
                return line
    return ''


def extract_humaneval_docstring(code, function_header, stop_words):
    text = code.split(function_header)[1].strip()
    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0]
    return text

def extract_python_code(gpt_output):
    lines = gpt_output.split('\n')
    code_block = []
    in_code_block = False
    for line in lines:
        if line.strip() == '```python':
            in_code_block = True
            continue
        elif line.strip() == '```' and in_code_block:
            break
        elif in_code_block:
            code_block.append(line)
    return '\n'.join(code_block)

def extract_python_code_hf(model_output):
    function_start_idx = model_output.find('def')
    if function_start_idx != -1:
        return model_output[function_start_idx:]
    
    function_start_idx = model_output.find('class')
    if function_start_idx != -1:
        return model_output[function_start_idx:]
    return ''

# Define a timeout duration for your test cases (in seconds)
TEST_CASE_TIMEOUT = 30  # Example: 30 seconds

def run_single_test_case(module, test):
    """
    Function to run a single test case.
    """
    exec(test, globals(), module.__dict__)

@timeout_decorator.timeout(TEST_CASE_TIMEOUT)
def run_test_case_with_timeout(module, test):
    """
    Wrapper function to execute run_single_test_case with a timeout.
    """
    run_single_test_case(module, test)

def run_test_cases(module, test_list):
    """
    Run each test case for the given module and handle timeouts.
    """
    for test in test_list:
        try:
            run_test_case_with_timeout(module, test)
        except AssertionError:
            return False
        except timeout_decorator.TimeoutError:
            print("Test case timed out. Moving to next test case.")
            continue  # Continue to the next test case after a timeout
    return True

def run_test_cases_for_file(file_path, test_list):
    try:
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        test_results = run_test_cases(module, test_list)
        return ('Pass', None) if test_results else ('Fail', 'Semantic')
    except SyntaxError as se:
        return ('Fail', 'Syntax')
    except Exception as e:
        # Handle other types of exceptions here, if necessary
        return ('Fail', 'Other Exception')


def main(df):
    # Load the DataFrame
    df = df[['text', 'code', 'test_list', 'new_test_list']]
    df = df.copy()
    df['function_header'] = df['code'].apply(extract_function_header)

    prompt_index = int(args_dict['prompt']) if args_dict['prompt'] else 0
    num_runs = int(args_dict['num_runs']) if args_dict['num_runs'] else 1

    # Directory for each prompt
    output_directory = f'generated_code_files/prompt_{prompt_index}'
    os.makedirs(output_directory, exist_ok=True)

    # Getting the prompt details
    docstring = df.iloc[prompt_index]['text']
    code = df.iloc[prompt_index]['code']
    function_header = df.iloc[prompt_index]['function_header']
    if args_dict['test_type'] == 'new':
        test_list = df.iloc[prompt_index]['new_test_list']
    else:
        test_list = df.iloc[prompt_index]['test_list']

    deltas = [
        f"{docstring}\n{function_header}\n{test_list}",
        docstring,
        f"{docstring}\n{function_header}",
        function_header,
        f"{function_header}\n{test_list}",
        test_list
    ]

    results = {f'delta_{i}': [] for i in range(6)}  # Assuming 6 different deltas

    for run_index in range(num_runs):
        print(f"Run {run_index + 1} of {num_runs} for prompt {prompt_index}")
        for i, delta in enumerate(deltas):
            print(f"Generating code for delta {i + 1} of {len(deltas)}")
            delta_key = f'delta_{i}'
            generated_output = models.generate_model_output(delta, args_dict['model'])
            generated_code = extract_python_code(generated_output)
            file_name = f"{output_directory}/delta_{run_index}_{i}.py"
            with open(file_name, 'w') as file:
                file.write(generated_code)
            print(f"Running test(s) for delta {i + 1} of {len(deltas)}")
            test_result, error_type = run_test_cases_for_file(file_name, test_list)  # Updated to receive a tuple
            results[delta_key].append((test_result, error_type))  # Store as a tuple

    all_results = []
    for delta_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        test_results = results[delta_key]
        for run_index, (test_result, error_type) in enumerate(test_results):  # Unpack the tuple
            pass_fail = 'Pass' if test_result == 'Pass' else 'Fail'
            file_name = f"{output_directory}/delta_{run_index}_{delta_key.split('_')[1]}.py"
            with open(file_name, 'r') as file:
                code = file.read()
            all_results.append({
                'Delta': delta_key,
                'Pass/Fail': pass_fail,
                'Error Type': error_type,  # Add error type
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
    Run the ILLM tests.
    """
    args_dict['model'] = model
    args_dict['prompt'] = prompt_num
    args_dict['num_runs'] = num_runs
    args_dict['test_type'] = test_type
    main(df)
    

def process_humaneval_deltas(test_type, prompt, entry_point, **kwargs):
    
    function_header = extract_function_header(prompt, entry_point)
    text = extract_humaneval_docstring(prompt, function_header, ['Example', 'example', 'For example', 'For Example', '>>>', '>>', f'\n{entry_point}(', f'  {entry_point}('])
    test_list = prompt.split(text)[1].strip().replace('"', '')
    text = text.strip().replace('"', '')
    
    deltas = [
        f"{text}\n{test_list}\n{function_header}",
        text,
        f"{text}\n{function_header}",
        function_header,
        f"{test_list}\n{function_header}",
        #test_list
    ]
    return deltas

def process_humaneval_plus_testcases(expected_output, test, entry_point,**kwargs):
    def prepare_input(inp):
        return ', '.join([str(i) for i in inp])
    test = [f'assert {entry_point}({prepare_input(i)}) == {str(j)}' for i,j in zip(test, expected_output)]
    return test

def process_humaneval_testcases(test, entry_point, **kwargs):
    test = [i.strip() for i in test.split('\n') if 'assert' in i]
    test = [i.replace('candidate', entry_point) for i in test]
    return test

def process_mbpp_deltas(test_type, text, code, test_list, new_test_list, **kwargs):
    
    function_header = extract_function_header(code=code)
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


if __name__ == '__main__':
    main()