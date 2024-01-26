import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def create_deltas_key_table(df, elements, styleSheet):
    """
    Create a table for the Deltas Key section in the PDF document.

    :param df: DataFrame containing the test results.
    :param elements: List of elements to which the table will be added.
    :param styleSheet: StyleSheet used for formatting the report.
    """
    deltas_components = df[['Delta', 'Components']].drop_duplicates().sort_values(by='Delta')
    data = [['Delta', 'Components']] + deltas_components.values.tolist()

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(table)

def generate_deltas_tree(df):
    """
    Generate a centered three-level binary tree image with corrected edges and pass ratios for each delta.
    """
    def parse_components(components_str):
        """Parse the components string into a list of individual components."""
        return components_str.split(", ")

    # Identifying deltas for each component and combination of components
    components_set = set()
    for components in df['Components']:
        components_set.update(parse_components(components))

    component_to_deltas = {component: [] for component in components_set}
    for index, row in df.iterrows():
        for component in parse_components(row['Components']):
            component_to_deltas[component].append(row['Delta'])

    # Finding the root, level 2, and level 3 nodes
    root_node = None
    level_2_nodes = {}
    level_3_nodes = []

    for delta, components in df.set_index('Delta')['Components'].items():
        parsed_components = parse_components(components)
        if len(parsed_components) == len(components_set):
            root_node = delta
        elif len(parsed_components) == 2:
            level_2_nodes[delta] = parsed_components
        elif len(parsed_components) == 1:
            level_3_nodes.append(delta)

    # Calculate pass ratios for each delta
    delta_pass_ratios = df.groupby('Delta')['Pass/Fail'].value_counts().unstack(fill_value=0)

    # Ensure both 'Pass' and 'Fail' columns exist
    if 'Pass' not in delta_pass_ratios.columns:
        delta_pass_ratios['Pass'] = 0
    if 'Fail' not in delta_pass_ratios.columns:
        delta_pass_ratios['Fail'] = 0

    delta_pass_ratios['Pass Ratio'] = delta_pass_ratios['Pass'] / (delta_pass_ratios['Pass'] + delta_pass_ratios['Fail'])

    # Create the graph
    G = nx.DiGraph()

    # Adding nodes with pass ratios and edges based on dynamically identified levels
    G.add_node(root_node)
    for node, components in level_2_nodes.items():
        G.add_node(node)
        G.add_edge(root_node, node)
        for component in components:
            for delta in component_to_deltas[component]:
                if delta in level_3_nodes:
                    G.add_node(delta)
                    G.add_edge(node, delta)

    # Using Graphviz to position nodes
    pos = graphviz_layout(G, prog='dot')  # This will position the nodes in a hierarchical structure

    # Drawing the tree
    plt.figure(figsize=(6, 4))  # Adjust figure size to your preference
    labels = {node: f"{node}\n{delta_pass_ratios.loc[node, 'Pass Ratio']:.2f}" for node in G.nodes()}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=3500, node_color='skyblue', font_size=10, edge_color='black', arrowsize=20)

    # Saving the image
    img_data = BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    plt.close()
    img_data.seek(0)
    return img_data

def generate_bar_graph(data, title, xlabel, ylabel, y_min=0):
    """
    Generate a bar graph from the given data.

    :param data: Data to be plotted in the bar graph.
    :param title: Title of the graph.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param y_min: Minimum value for the y-axis.
    :return: BytesIO object containing the image data of the plot.
    """
    plt.figure(figsize=(6, 4))
    data.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(bottom=y_min)
    plt.tight_layout()
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    plt.close()
    img_data.seek(0)
    return img_data

def add_plot_to_pdf(elements, img_data):
    """
    Add a plot to the PDF document elements.

    :param elements: List of elements to which the plot will be added.
    :param img_data: BytesIO object containing the image data of the plot.
    """
    img = Image(img_data)
    elements.append(img)

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

def calculate_variance(df):
    """
    Calculate the variance in pass counts for each delta across all runs.
    
    :param df: DataFrame containing the test results.
    :return: A Series with the variance for each delta.
    """
    # Pivot table to count 'Pass' for each delta per run
    delta_run_counts = df.pivot_table(index='Delta', columns='Run Index', values='Pass/Fail',
                                      aggfunc=lambda x: (x == 'Pass').sum(), fill_value=0)
    
    # Calculate variance for each delta's pass counts
    variance = delta_run_counts.var(axis=1, ddof=0)  # ddof=0 for population variance
    
    return variance

def calculate_overall_stats(df):
    """
    Calculate overall statistics from the DataFrame.

    :param df: DataFrame containing the test results.
    :return: A dictionary containing overall statistics.
    """
    all_runs_delta_analysis = df.groupby('Delta')['Pass/Fail'].value_counts().unstack(fill_value=0)
    all_runs_delta_analysis['Pass'] = all_runs_delta_analysis.get('Pass', pd.Series(index=all_runs_delta_analysis.index, data=[0]*len(all_runs_delta_analysis)))
    all_runs_delta_analysis['Fail'] = all_runs_delta_analysis.get('Fail', pd.Series(index=all_runs_delta_analysis.index, data=[0]*len(all_runs_delta_analysis)))

    # Calculate pass ratio for each delta
    all_runs_delta_analysis['Pass Ratio'] = all_runs_delta_analysis['Pass'] / (all_runs_delta_analysis['Pass'] + all_runs_delta_analysis['Fail'])

    # Calculate variance for each delta
    delta_variance = calculate_variance(df)
    all_runs_delta_analysis['Variance'] = delta_variance

    # Identify deltas that passed in all runs or failed in all runs
    deltas_passed_in_all_runs = all_runs_delta_analysis[all_runs_delta_analysis['Fail'] == 0].index.tolist()
    deltas_failed_in_all_runs = all_runs_delta_analysis[all_runs_delta_analysis['Pass'] == 0].index.tolist()

    # Identify deltas that passed in some runs but not all
    deltas_passed_in_some_runs = all_runs_delta_analysis[(all_runs_delta_analysis['Pass Ratio'] > 0) & (all_runs_delta_analysis['Pass Ratio'] < 1)].index.tolist()

    error_type_counts_overall = df[df['Pass/Fail'] == 'Fail']['Error Type'].value_counts()

    return {
        'deltas_passed_in_all_runs': deltas_passed_in_all_runs,
        'deltas_failed_in_all_runs': deltas_failed_in_all_runs,
        'deltas_passed_in_some_runs': deltas_passed_in_some_runs,  # New table data
        'delta_analysis_overall': all_runs_delta_analysis,
        'error_type_counts_overall': error_type_counts_overall
    }

def generate_report(df, file_path, prompt_name):
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

    elements.append(Paragraph("<b>Overall Analysis</b>", styleSheet['Heading1']))
    elements.append(Spacer(1, 12))

    overall_stats = calculate_overall_stats(df)
    create_tables_for_analysis(overall_stats, elements, styleSheet, overall=True)

    # Add the binary tree image
    tree_image = generate_deltas_tree(df)
    add_plot_to_pdf(elements, tree_image)

    elements.append(PageBreak())
    elements.append(Paragraph("<b>Individual Run Analysis</b>", styleSheet['Heading1']))
    elements.append(Spacer(1, 12))

    for run_index in df['Run Index'].unique():
        elements.append(Paragraph(f"<b>Run {run_index}</b>", styleSheet['Heading2']))
        analysis_results = analyze_csv_data(df, run_index)
        create_tables_for_analysis(analysis_results, elements, styleSheet)
        elements.append(Spacer(1, 12))
        if run_index < df['Run Index'].max():
            elements.append(PageBreak())

    elements.append(PageBreak())
    elements.append(Paragraph("<b>Deltas Key</b>", styleSheet['Heading1']))
    # Create Deltas Key table
    create_deltas_key_table(df, elements, styleSheet)

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
        create_table([["Delta"]] + [[delta] for delta in analysis_results['deltas_passed_in_some_runs']], "Deltas Passed in Some Runs")
        create_table([["Delta"]] + [[delta] for delta in analysis_results['deltas_failed_in_all_runs']], "Deltas Failed in All Runs")
        create_table([["Error Type", "Count"]] + [[et, c] for et, c in analysis_results['error_type_counts_overall'].items()], "Error Type Counts (Overall)")

        # Adding the Delta Analysis (Overall) table
        delta_data = [["Delta", "Pass", "Fail", "Pass Ratio", "Variance"]]
        for delta, row in analysis_results['delta_analysis_overall'].iterrows():
            total = row['Pass'] + row['Fail']
            pass_ratio = row['Pass'] / total if total > 0 else 0
            variance = row['Variance']
            delta_data.append([delta, row['Pass'], row['Fail'], f"{pass_ratio:.2f}", f"{variance:.2f}"])
        create_table(delta_data, "Delta Analysis (Overall)")

        elements.append(Spacer(1, 12))

        # Generating and adding the bar graphs
        if not analysis_results['error_type_counts_overall'].empty:
            error_graph_data = generate_bar_graph(analysis_results['error_type_counts_overall'], "Error Type Counts", "Error Types", "Counts")
            add_plot_to_pdf(elements, error_graph_data)
        else:
            no_error_note = Paragraph("NOTE: No 'Fail' entries were found across all runs, hence no error type count plot was generated.", styleSheet['BodyText'])
            elements.append(no_error_note)

        delta_graph_data = generate_bar_graph(analysis_results['delta_analysis_overall']['Pass'], "Delta Analysis - Pass Count", "Deltas", "Pass Count")
        add_plot_to_pdf(elements, delta_graph_data)

    else:
        create_table([["Statistic", "Count"], ["Total Passes", analysis_results['total_passes']], ["Total Fails", analysis_results['total_fails']]], "Total Passes and Fails")
        create_table([["Error Type", "Count"]] + [[et, c] for et, c in analysis_results['error_type_counts'].items()], "Error Type Counts")
        create_table([["Delta", "Result", "Details"]] + [[delta, "Pass" if analysis_results['delta_analysis'].loc[delta, 'Pass'] > 0 else "Fail", analysis_results['delta_analysis'].loc[delta, 'Details'] if analysis_results['delta_analysis'].loc[delta, 'Pass'] == 0 else ""] for delta in analysis_results['delta_analysis'].index], "Delta Analysis")