import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def analyze_csv_data(df, run_index):
    """
    Analyze CSV data for a specific run index.

    :param df: DataFrame containing the test results.
    :param run_index: The index of the run to analyze.
    :return: A dictionary containing analysis results.
    """
    run_df = df[df['Run Index'] == run_index]
    total_passes = (run_df['Pass/Fail'] == 'Pass').sum()
    total_fails = (run_df['Pass/Fail'] == 'Fail').sum()
    error_type_counts = run_df[run_df['Pass/Fail'] == 'Fail']['Error Type'].value_counts()

    # Ensure both 'Pass' and 'Fail' columns exist after unstacking
    delta_analysis = run_df.pivot_table(index='Delta', columns='Pass/Fail', aggfunc='size', fill_value=0)
    delta_analysis['Pass'] = delta_analysis.get('Pass', pd.Series(index=delta_analysis.index, data=[0]*len(delta_analysis)))
    delta_analysis['Fail'] = delta_analysis.get('Fail', pd.Series(index=delta_analysis.index, data=[0]*len(delta_analysis)))

    error_type_mapping = run_df[run_df['Pass/Fail'] == 'Fail'].groupby('Delta')['Error Type'].first()
    delta_analysis['Details'] = delta_analysis.index.map(error_type_mapping)

    return {
        'total_passes': total_passes,
        'total_fails': total_fails,
        'error_type_counts': error_type_counts,
        'delta_analysis': delta_analysis
    }

def calculate_overall_stats(df):
    """
    Calculate overall statistics from the DataFrame.

    :param df: DataFrame containing the test results.
    :return: A dictionary containing overall statistics.
    """
    all_runs_delta_analysis = df.groupby('Delta')['Pass/Fail'].value_counts().unstack(fill_value=0)
    all_runs_delta_analysis['Pass'] = all_runs_delta_analysis.get('Pass', pd.Series(index=all_runs_delta_analysis.index, data=[0]*len(all_runs_delta_analysis)))
    all_runs_delta_analysis['Fail'] = all_runs_delta_analysis.get('Fail', pd.Series(index=all_runs_delta_analysis.index, data=[0]*len(all_runs_delta_analysis)))

    deltas_passed_in_all_runs = all_runs_delta_analysis[all_runs_delta_analysis['Fail'] == 0].index.tolist()
    deltas_failed_in_all_runs = all_runs_delta_analysis[all_runs_delta_analysis['Pass'] == 0].index.tolist()

    error_type_counts_overall = df[df['Pass/Fail'] == 'Fail']['Error Type'].value_counts()

    return {
        'deltas_passed_in_all_runs': deltas_passed_in_all_runs,
        'deltas_failed_in_all_runs': deltas_failed_in_all_runs,
        'delta_analysis_overall': all_runs_delta_analysis,
        'error_type_counts_overall': error_type_counts_overall
    }

def generate_pdf_report(df, file_path, prompt_name):
    """
    Generate a PDF report from the DataFrame.

    :param df: DataFrame containing the test results.
    :param file_path: Path where the PDF report will be saved.
    :param prompt_name: Name of the prompt for which the report is generated.
    """
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    elements, styleSheet = [], getSampleStyleSheet()
    elements.append(Paragraph(f"<b>Prompt {prompt_name} Analysis Report</b>", styleSheet['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Overall Statistics</b>", styleSheet['Heading2']))
    elements.append(Spacer(1, 12))

    overall_stats = calculate_overall_stats(df)
    create_tables_for_analysis(overall_stats, elements, styleSheet, overall=True)

    for run_index in df['Run Index'].unique():
        elements.append(PageBreak() if run_index > 0 else Spacer(1, 12))
        elements.append(Paragraph(f"<b>Run {run_index}</b>", styleSheet['Heading2']))
        analysis_results = analyze_csv_data(df, run_index)
        create_tables_for_analysis(analysis_results, elements, styleSheet)

    doc.build(elements)

def create_tables_for_analysis(analysis_results, elements, styleSheet, overall=False):
    """
    Create tables for analysis results and append them to the document elements.

    :param analysis_results: Dictionary containing analysis results.
    :param elements: List of elements to which the tables will be added.
    :param styleSheet: StyleSheet used for formatting the report.
    :param overall: Boolean indicating whether the analysis is overall or per run.
    """
    def create_table(data, title=None):
        if title:
            elements.append(Paragraph(title, styleSheet['Heading2']))
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)

    # Generate tables based on whether the analysis is overall or per run
    if overall:
        create_table([["Delta"]] + [[delta] for delta in analysis_results['deltas_passed_in_all_runs']], "Deltas Passed in All Runs")
        create_table([["Delta"]] + [[delta] for delta in analysis_results['deltas_failed_in_all_runs']], "Deltas Failed in All Runs")
        create_table([["Error Type", "Count"]] + [[et, c] for et, c in analysis_results['error_type_counts_overall'].items()], "Error Type Counts (Overall)")
    else:
        create_table([["Statistic", "Count"], ["Total Passes", analysis_results['total_passes']], ["Total Fails", analysis_results['total_fails']]], "Total Passes and Fails")
        create_table([["Error Type", "Count"]] + [[et, c] for et, c in analysis_results['error_type_counts'].items()], "Error Type Counts")
        create_table([["Delta", "Result", "Details"]] + [[delta, "Pass" if analysis_results['delta_analysis'].loc[delta, 'Pass'] > 0 else "Fail", analysis_results['delta_analysis'].loc[delta, 'Details'] if analysis_results['delta_analysis'].loc[delta, 'Pass'] == 0 else ""] for delta in analysis_results['delta_analysis'].index], "Delta Analysis")

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
                print(f"Analysis PDF generated: {pdf_file_path}")

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