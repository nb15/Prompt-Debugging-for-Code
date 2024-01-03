import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet

def analyze_csv_data(df):
    pass_fail_counts = df['Pass/Fail'].value_counts()
    total_passes = pass_fail_counts.get('Pass', 0)
    total_fails = pass_fail_counts.get('Fail', 0)

    # Ensure 'Error Type' column exists and handle cases where there are no 'Fail' entries
    if 'Error Type' in df.columns and 'Fail' in pass_fail_counts:
        error_type_counts = df[df['Pass/Fail'] == 'Fail']['Error Type'].value_counts()
    else:
        error_type_counts = pd.Series(dtype='int64')

    delta_analysis = df.groupby('Delta')['Pass/Fail'].value_counts().unstack(fill_value=0)
    delta_analysis = delta_analysis.reindex(columns=['Pass', 'Fail'], fill_value=0)
    delta_analysis['pass_ratio'] = delta_analysis['Pass'] / (delta_analysis['Pass'] + delta_analysis['Fail'])
    delta_analysis = delta_analysis.reset_index()

    code_analysis = df.groupby(['Code', 'Pass/Fail']).size().unstack(fill_value=0)
    code_analysis = code_analysis.reindex(columns=['Pass', 'Fail'], fill_value=0)
    code_analysis['Pass Ratio'] = code_analysis['Pass'] / (code_analysis['Pass'] + code_analysis['Fail'])
    code_analysis = code_analysis.reset_index()

    analysis_results = {
        'total_passes': total_passes,
        'total_fails': total_fails,
        'error_type_counts': error_type_counts,
        'delta_analysis': delta_analysis,
        'deltas_passed_all_runs': df[df['Passed In All Runs'] == 'Yes']['Delta'].unique(),
        'code_analysis': code_analysis
    }
    return analysis_results

def format_analysis_results(analysis_results):
    formatted_output = {
        "Summary": "",
        "Error Type Counts": pd.DataFrame(analysis_results['error_type_counts'].items(), columns=["Error Type", "Count"]),
        "Delta Analysis": analysis_results['delta_analysis'],
        "Deltas Passed in All Runs": pd.DataFrame(analysis_results['deltas_passed_all_runs'], columns=["Delta"]),
        "Code Snippets Analysis": analysis_results['code_analysis']
    }

    summary_data = [["Total Passes", analysis_results['total_passes']],
                    ["Total Fails", analysis_results['total_fails']]]
    formatted_output["Summary"] = "\n".join([f"{row[0]}: {row[1]}" for row in summary_data])

    return formatted_output

def overall_analysis(dfs):
    combined_df = pd.concat(dfs, ignore_index=True)
    overall_analysis_results = analyze_csv_data(combined_df)
    return overall_analysis_results

def save_analysis_to_pdf(analysis_data, file_name, graph_file_path=None, heatmap_file_path=None):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    elements = []

    for title, data in analysis_data.items():
        elements.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
        if isinstance(data, pd.DataFrame):
            table_data = [data.columns.tolist()] + data.values.tolist()
            t = Table(table_data)
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                   ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                   ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke)]))
            elements.append(t)
        else:
            elements.append(Paragraph(data, styles['Normal']))

    if graph_file_path and os.path.exists(graph_file_path):
        img = Image(graph_file_path)
        img_width = 400
        img_height = img.imageHeight * img_width / img.imageWidth
        img.drawWidth = img_width
        img.drawHeight = img_height
        elements.append(img)

    if heatmap_file_path and os.path.exists(heatmap_file_path):
        img = Image(heatmap_file_path)
        img_width = 400
        img_height = img.imageHeight * img_width / img.imageWidth
        img.drawWidth = img_width
        img.drawHeight = img_height
        elements.append(img)

    doc.build(elements)
    print(f"Analysis saved to {file_name}")

def save_delta_performance_graph(delta_analysis, file_path):
    plt.figure(figsize=(10, 6))
    plt.bar(delta_analysis['Delta'], delta_analysis['pass_ratio'], color='blue')
    plt.xlabel('Delta')
    plt.ylabel('Pass Ratio')
    plt.title('Delta Performance Analysis')
    plt.savefig(file_path)
    plt.close()

def transform_data_for_heatmap(all_dataframes):
    heatmap_data_list = []

    for df in all_dataframes:
        # Convert 'Pass/Fail' to numeric values, e.g., 1 for Pass, 0 for Fail
        df['Pass/Fail'] = df['Pass/Fail'].map({'Pass': 1, 'Fail': 0})

        # Calculate pass rate for each delta in the DataFrame
        pass_rate = df.pivot_table(index='Delta', values='Pass/Fail', aggfunc='mean')
        heatmap_data_list.append(pass_rate.T)

    # Concatenate all the pass rate DataFrames
    heatmap_data = pd.concat(heatmap_data_list, ignore_index=True)

    heatmap_data.columns = [f'delta_{i}' for i in range(len(heatmap_data.columns))]
    return heatmap_data

def save_heatmap(heat_data, file_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(heat_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Delta Correlation Heatmap')
    plt.savefig(file_path)
    plt.close()

def main():
    base_folder_path = 'generated_code_files'
    all_dataframes = []
    csv_files_count = 0

    for root, dirs, files in os.walk(base_folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files_count += 1
                file_path = os.path.join(root, file)
                print(f"Processing CSV file: {file_path}")
                df = pd.read_csv(file_path)
                analysis_results = analyze_csv_data(df)
                formatted_results = format_analysis_results(analysis_results)
                print(formatted_results["Summary"])

                # Save graph for each CSV file
                graph_file_path = os.path.splitext(file_path)[0] + "_delta_performance.png"
                save_delta_performance_graph(analysis_results['delta_analysis'], graph_file_path)

                # Save analysis to PDF for each CSV file
                pdf_file_name = os.path.splitext(file_path)[0] + "_analysis.pdf"
                save_analysis_to_pdf(formatted_results, pdf_file_name, graph_file_path=graph_file_path)
                
                all_dataframes.append(df)

    heatmap_file_path = None
    if csv_files_count > 1:
        print("Performing overall analysis across all CSV files...")

        # Perform overall analysis
        overall_results = overall_analysis(all_dataframes)
        formatted_overall_results = format_analysis_results(overall_results)
        print(formatted_overall_results["Summary"])

        # Generate heatmap for overall analysis
        transformed_heatmap_data = transform_data_for_heatmap(all_dataframes)
        heatmap_file_path = os.path.join(base_folder_path, "delta_correlation_heatmap.png")
        save_heatmap(transformed_heatmap_data, heatmap_file_path)

        # Save overall analysis to PDF including heatmap
        overall_pdf_file_name = os.path.join(base_folder_path, "overall_analysis.pdf")
        save_analysis_to_pdf(formatted_overall_results, overall_pdf_file_name, heatmap_file_path=heatmap_file_path)
    
    elif csv_files_count == 0:
        print("No CSV files were found for analysis.")

def run_analysis():
    main()

if __name__ == "__main__":
    main()