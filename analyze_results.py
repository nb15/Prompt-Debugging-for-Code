import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def analyze_csv_data(df, run_index):
    run_df = df[df['Run Index'] == run_index]

    # Counting total passes and fails
    total_passes = (run_df['Pass/Fail'] == 'Pass').sum()
    total_fails = (run_df['Pass/Fail'] == 'Fail').sum()

    # Counting error types for failures
    error_type_counts = run_df[run_df['Pass/Fail'] == 'Fail']['Error Type'].value_counts()

    # Analyzing deltas
    delta_analysis = run_df.pivot_table(index='Delta', columns='Pass/Fail', aggfunc='size', fill_value=0)
    delta_analysis['Pass'] = delta_analysis.get('Pass', 0)
    delta_analysis['Fail'] = delta_analysis.get('Fail', 0)

    # Create a column for Error Type details (only for Fail)
    error_type_mapping = run_df[run_df['Pass/Fail'] == 'Fail'].groupby('Delta')['Error Type'].first()
    delta_analysis['Details'] = delta_analysis.index.map(error_type_mapping)

    return {
        'total_passes': total_passes,
        'total_fails': total_fails,
        'error_type_counts': error_type_counts,
        'delta_analysis': delta_analysis
    }

def calculate_overall_stats(df):
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
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    elements = []
    styleSheet = getSampleStyleSheet()

    title = Paragraph(f"<b>Prompt {prompt_name} Analysis Report</b>", styleSheet['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Overall Stats Section Header
    elements.append(Paragraph("<b>Overall Statistics</b>", styleSheet['Heading2']))
    elements.append(Spacer(1, 12))

    # Add Overall Stats
    overall_stats = calculate_overall_stats(df)
    create_tables_for_analysis(overall_stats, elements, styleSheet, overall=True)

    # Individual Run Stats
    run_indices = df['Run Index'].unique()
    for idx, run_index in enumerate(run_indices):
        if idx > 0:
            elements.append(PageBreak())  # Page break before each run section, except the first one
        elements.append(Paragraph(f"<b>Run {run_index}</b>", styleSheet['Heading2']))
        analysis_results = analyze_csv_data(df, run_index)
        create_tables_for_analysis(analysis_results, elements, styleSheet)

    doc.build(elements)

def create_tables_for_analysis(analysis_results, elements, styleSheet, overall=False):
    def create_table(data, title=None, scale_down=False):
        if title:
            elements.append(Paragraph(title, styleSheet['Heading2']))
        t = Table(data)
        style = [('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                 ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                 ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                 ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                 ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                 ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                 ('GRID', (0, 0), (-1, -1), 1, colors.black)]
        if scale_down:
            style.append(('FONTSIZE', (0, 0), (-1, -1), 8))
        t.setStyle(TableStyle(style))
        elements.append(t)

    if overall:
        # Tables for Overall Stats
        # Deltas Passed in All Runs
        deltas_passed_data = [["Delta"]] + [[delta] for delta in analysis_results['deltas_passed_in_all_runs']]
        create_table(deltas_passed_data, title="Deltas Passed in All Runs")

        # Deltas Failed in All Runs
        deltas_failed_data = [["Delta"]] + [[delta] for delta in analysis_results['deltas_failed_in_all_runs']]
        create_table(deltas_failed_data, title="Deltas Failed in All Runs")

        # Error Type Counts (Overall)
        error_type_data_overall = [["Error Type", "Count"]] + \
                                  [[error_type, count] for error_type, count in analysis_results['error_type_counts_overall'].items()]
        create_table(error_type_data_overall, title="Error Type Counts (Overall)")

        # Delta Analysis (Overall)
        delta_data_overall = [["Delta", "Pass", "Fail", "Pass Ratio"]] + \
                             [[delta] + values[['Pass', 'Fail']].tolist() + [values['Pass'] / (values['Pass'] + values['Fail'])] for delta, values in analysis_results['delta_analysis_overall'].iterrows()]
        create_table(delta_data_overall, title="Delta Analysis (Overall)")

    else:
        # Tables for Individual Run Stats
        # Total Passes and Fails (Fixed Formatting)
        pass_fail_data = [["Statistic", "Count"],
                          ["Total Passes", analysis_results['total_passes']],
                          ["Total Fails", analysis_results['total_fails']]]
        create_table(pass_fail_data, title="Total Passes and Fails")

        # Error Type Counts
        error_type_data = [["Error Type", "Count"]] + \
                          [[error_type, count] for error_type, count in analysis_results['error_type_counts'].items()]
        create_table(error_type_data, title="Error Type Counts")

        # Updated Delta Analysis Table
        delta_analysis_data = [["Delta", "Result", "Details"]]
        for delta in analysis_results['delta_analysis'].index:  # Use the delta values from the DataFrame
            result = "Pass" if analysis_results['delta_analysis'].loc[delta, 'Pass'] > 0 else "Fail"
            details = analysis_results['delta_analysis'].loc[delta, 'Details'] if result == "Fail" else ""
            delta_analysis_data.append([delta, result, details])
        create_table(delta_analysis_data, title="Delta Analysis")

def process_csv_in_folder(folder_path, perform_individual_analysis):
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
    base_folder_path = 'generated_code_files'
    for root, dirs, files in os.walk(base_folder_path):
        for dir_name in dirs:
            if dir_name.startswith('prompt_'):
                full_path = os.path.join(root, dir_name)
                process_csv_in_folder(full_path, True)