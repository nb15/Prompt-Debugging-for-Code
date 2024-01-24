import os
import pandas as pd
from reports import generate_prompt_report
from reports import generate_global_report
import os
import jsonlines


def runtime_evaluation(base_path, total_prompts, runs_per_prompt):
    """
    Process each CSV file in each folder for individual report generation and aggregate data for global report.

    :param base_path: The base directory containing the prompt directories.
    :return: A DataFrame containing aggregated data from all CSV files.
    """
    all_data = []
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith('prompt_'):
                full_path = os.path.join(root, dir_name)
                for file_name in os.listdir(full_path):
                    if file_name.endswith('.csv'):
                        csv_file_path = os.path.join(full_path, file_name)
                        df = pd.read_csv(csv_file_path)
                        prompt_name = dir_name.split('_')[-1]
                        pdf_file_path = os.path.join(full_path, f'prompt_{prompt_name}_analysis.pdf')
                        generate_prompt_report.generate_report(df, pdf_file_path, prompt_name)
                        print(f'Generated PDF report for prompt {prompt_name} at {pdf_file_path}')
                        df['Prompt'] = dir_name  # Adding a column to identify the prompt
                        all_data.append(df)
    # Generate global report
    global_data = pd.concat(all_data, ignore_index=True)
    global_report_path = os.path.join(base_path, 'global_analysis_report.pdf')
    generate_global_report.generate_report(global_data, global_report_path, total_prompts, runs_per_prompt)
    print(f'Generated global analysis report at {global_report_path}')

def evalplus_evaluation(base_path):
    """
    Process each JSONL file in each 'prompt_' prefixed folder, aggregate data into a global JSONL file,
    rename certain columns, and retain only specified columns.

    :param base_path: The base directory containing the prompt directories.
    """
    all_data = []
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith('prompt_'):
                full_path = os.path.join(root, dir_name)
                for file_name in os.listdir(full_path):
                    if file_name.endswith('.jsonl'):
                        jsonl_file_path = os.path.join(full_path, file_name)
                        with jsonlines.open(jsonl_file_path) as reader:
                            for obj in reader:
                                # Create a new object with only task_id and solution
                                new_obj = {}
                                if 'Task ID' in obj:
                                    new_obj['task_id'] = obj['Task ID']
                                elif 'task_id' in obj:
                                    new_obj['task_id'] = obj['task_id']

                                if 'Code' in obj:
                                    new_obj['solution'] = obj['Code']
                                elif 'solution' in obj:
                                    new_obj['solution'] = obj['solution']

                                if new_obj:
                                    all_data.append(new_obj)
    
    # Writing the processed data to samples.jsonl
    global_jsonl_path = os.path.join(base_path, 'samples.jsonl')
    with jsonlines.open(global_jsonl_path, mode='w') as writer:
        writer.write_all(all_data)
    print(f'Aggregated prompt results to {global_jsonl_path}')

def run_analysis(total_prompts, runs_per_prompt, evaluation):
    """
    Run the analysis on the generated code files and produce reports.
    """
    base_folder_path = 'generated_code_files'
    if evaluation == 'runtime':
        runtime_evaluation(base_folder_path, total_prompts, runs_per_prompt)
    elif evaluation == 'evalplus':
        evalplus_evaluation(base_folder_path)