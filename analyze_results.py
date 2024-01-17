import os
import pandas as pd
from generate_prompt_report import generate_pdf_report
from generate_global_report import generate_global_pdf_report

def process_and_aggregate_csv_files(base_path, perform_individual_analysis):
    """
    Process each CSV file in each folder for individual report generation and aggregate data for global report.

    :param base_path: The base directory containing the prompt directories.
    :param perform_individual_analysis: Boolean to decide whether to perform individual analysis.
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
                        if perform_individual_analysis:
                            prompt_name = dir_name.split('_')[-1]
                            pdf_file_path = os.path.join(full_path, f'prompt_{prompt_name}_analysis.pdf')
                            generate_pdf_report(df, pdf_file_path, prompt_name)
                            print(f'Generated PDF report for prompt {prompt_name} at {pdf_file_path}')
                        df['Prompt'] = dir_name  # Adding a column to identify the prompt
                        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def run_analysis(total_prompts, runs_per_prompt):
    """
    Run the analysis on the generated code files and produce reports.
    """
    base_folder_path = 'generated_code_files'
    # Process and aggregate CSV files
    global_data = process_and_aggregate_csv_files(base_folder_path, True)

    # Generate global report
    global_report_path = os.path.join(base_folder_path, 'global_analysis_report.pdf')
    generate_global_pdf_report(global_data, global_report_path, total_prompts, runs_per_prompt)
    print(f'Generated global analysis report at {global_report_path}')