import os
import pandas as pd
from tabulate import tabulate

args_dict = {'type': ''}

def analyze_csv_data(df):
    pass_fail_counts = df['Pass/Fail'].value_counts()
    total_passes = pass_fail_counts.get('Pass', 0)
    total_fails = pass_fail_counts.get('Fail', 0)

    # New: Analyze errors by type
    error_type_counts = df[df['Pass/Fail'] == 'Fail']['Error Type'].value_counts()

    delta_analysis = df.groupby('Delta')['Pass/Fail'].value_counts().unstack(fill_value=0)
    # Ensure 'Pass' and 'Fail' columns exist
    delta_analysis['Pass'] = delta_analysis.get('Pass', pd.Series(index=delta_analysis.index, data=[0]*len(delta_analysis)))
    delta_analysis['Fail'] = delta_analysis.get('Fail', pd.Series(index=delta_analysis.index, data=[0]*len(delta_analysis)))
    # Calculate pass_ratio
    delta_analysis['pass_ratio'] = delta_analysis['Pass'] / (delta_analysis['Pass'] + delta_analysis['Fail'])

    code_analysis = df.groupby('Code')['Pass/Fail'].value_counts().unstack(fill_value=0)
    # Ensure 'Pass' and 'Fail' columns exist
    code_analysis['Pass'] = code_analysis.get('Pass', pd.Series(index=code_analysis.index, data=[0]*len(code_analysis)))
    code_analysis['Fail'] = code_analysis.get('Fail', pd.Series(index=code_analysis.index, data=[0]*len(code_analysis)))
    # Calculate pass_ratio
    code_analysis['pass_ratio'] = code_analysis['Pass'] / (code_analysis['Pass'] + code_analysis['Fail'])

    analysis_results = {
        'total_passes': total_passes,
        'total_fails': total_fails,
        'error_type_counts': error_type_counts,  # New
        'delta_analysis': delta_analysis,
        'deltas_passed_all_runs': df[df['Passed In All Runs'] == 'Yes']['Delta'].unique(),
        'code_analysis': code_analysis
    }
    return analysis_results

def format_analysis_results(analysis_results):
    formatted_output = ""

    pass_fail_data = [["Total Passes", analysis_results['total_passes']],
                      ["Total Fails", analysis_results['total_fails']]]
    formatted_output += tabulate(pass_fail_data, headers=[], tablefmt="grid") + "\n\n"

    # New: Add error type counts to the formatted output
    error_type_data = [[error_type, count] for error_type, count in analysis_results['error_type_counts'].items()]
    formatted_output += "Error Type Counts:\n"
    formatted_output += tabulate(error_type_data, headers=["Error Type", "Count"], tablefmt="grid") + "\n\n"

    formatted_output += "Delta Analysis (Pass/Fail Ratio for each Delta):\n"
    formatted_output += tabulate(analysis_results['delta_analysis'], headers='keys', tablefmt="grid") + "\n\n"

    formatted_output += "Deltas Passed in All Runs:\n"
    deltas_data = [[delta] for delta in analysis_results['deltas_passed_all_runs']]
    formatted_output += tabulate(deltas_data, headers=[], tablefmt="grid") + "\n\n"

    formatted_output += "Code Snippets Analysis (Pass/Fail Rates):\n"
    formatted_output += tabulate(analysis_results['code_analysis'], headers='keys', tablefmt="grid")

    return formatted_output

def overall_analysis(dfs):
    combined_df = pd.concat(dfs, ignore_index=True)
    overall_analysis_results = analyze_csv_data(combined_df)
    return format_analysis_results(overall_analysis_results)

def process_csv_in_folder(folder_path, perform_individual_analysis):
    csv_file_path = None
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, file_name)
            break

    if csv_file_path is not None:
        print(f"Processing CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        if perform_individual_analysis:
            analysis_results = analyze_csv_data(df)
            formatted_results = format_analysis_results(analysis_results)
            print(formatted_results)
        return df
    else:
        print(f"No CSV file found in {folder_path}.")
        return None

def main():
    perform_overall_analysis = False
    perform_individual_analysis = False
    
    if args_dict['type'] == 'overall':
        perform_overall_analysis = True
    else:
        perform_individual_analysis = True

    base_folder_path = 'generated_code_files'
    all_dataframes = []
    for root, dirs, files in os.walk(base_folder_path):
        for dir_name in dirs:
            if dir_name.startswith('prompt_'):
                full_path = os.path.join(root, dir_name)
                df = process_csv_in_folder(full_path, perform_individual_analysis)
                if df is not None:
                    all_dataframes.append(df)

    if perform_overall_analysis and all_dataframes:
        print("Performing overall analysis across all CSV files...")
        overall_results = overall_analysis(all_dataframes)
        print(overall_results)
    elif not all_dataframes:
        print("No CSV files were found for analysis.")

def run_analysis(type):
    args_dict['type'] = type
    main()

if __name__ == "__main__":
    main()