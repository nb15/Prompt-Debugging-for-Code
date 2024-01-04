import os
import pandas as pd
from generate_prompt_report import generate_pdf_report

def process_csv_in_folder(folder_path, perform_individual_analysis):
    """
    Process each CSV file in a folder and generate a PDF report.

    :param folder_path: Path of the folder containing CSV files.
    :param perform_individual_analysis: Boolean to decide whether to perform individual analysis.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(csv_file_path)
            if perform_individual_analysis:
                prompt_name = os.path.basename(folder_path).split('_')[-1]
                pdf_file_path = os.path.join(folder_path, f'prompt_{prompt_name}_analysis.pdf')
                generate_pdf_report(df, pdf_file_path, prompt_name)
                print(f'Generated PDF report for prompt {prompt_name} at {pdf_file_path}')

def run_analysis():
    """
    Run the analysis on the generated code files and produce reports.
    """
    base_folder_path = 'generated_code_files'
    for root, dirs, _ in os.walk(base_folder_path):
        for dir_name in dirs:
            if dir_name.startswith('prompt_'):
                full_path = os.path.join(root, dir_name)
                process_csv_in_folder(full_path, True)