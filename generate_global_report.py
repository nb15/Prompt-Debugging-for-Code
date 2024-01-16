import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def calculate_global_stats(df):
    """
    Calculate global statistics from the aggregated DataFrame.

    :param df: DataFrame containing the test results from all prompts.
    :return: A dictionary containing global statistics.
    """
    global_delta_analysis = df.groupby('Delta')['Pass/Fail'].value_counts().unstack(fill_value=0)
    global_delta_analysis['Pass'] = global_delta_analysis.get('Pass', pd.Series(index=global_delta_analysis.index, data=[0]*len(global_delta_analysis)))
    global_delta_analysis['Fail'] = global_delta_analysis.get('Fail', pd.Series(index=global_delta_analysis.index, data=[0]*len(global_delta_analysis)))
    global_delta_analysis['Pass Ratio'] = global_delta_analysis['Pass'] / (global_delta_analysis['Pass'] + global_delta_analysis['Fail'])
    return global_delta_analysis

def create_deltas_key_table(df):
    """
    Create a table for the Deltas Key section in the PDF document.

    :param df: DataFrame containing the test results.
    :return: Table object for the Deltas Key section.
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
    return table

def error_frequency_table(df):
    """
    Create a table for the frequency of each type of distinct error.

    :param df: DataFrame containing the test results.
    :return: Table object.
    """
    error_freq = df[df['Pass/Fail'] == 'Fail']['Error Type'].value_counts().reset_index()
    error_freq.columns = ['Error Type', 'Frequency']
    data = [['Error Type', 'Frequency']] + error_freq.values.tolist()

    return create_table(data, "Error Frequency Table")

def most_common_error_per_delta_table(df):
    """
    Create a table showing the most common error for each delta.

    :param df: DataFrame containing the test
    results.
    :return: Table object.
    """
    most_common_error = df[df['Pass/Fail'] == 'Fail'].groupby('Delta')['Error Type'].agg(lambda x: x.value_counts().index[0]).reset_index()
    most_common_error.columns = ['Delta', 'Most Common Error']
    data = [['Delta', 'Most Common Error']] + most_common_error.values.tolist()

    return create_table(data, "Most Common Error per Delta Table")

def create_table(data, title=None):
    """
    Create a generic table from the provided data.

    :param data: Data to be displayed in the table.
    :param title: Title of the table (optional).
    :return: Table object.
    """
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
    return t

def create_global_stats_table(global_stats):
    """
    Create a table from global statistics data.

    :param global_stats: A DataFrame containing global statistics.
    :return: Table object.
    """
    data = [["Delta", "Pass", "Fail", "Pass Ratio"]] + [
        [delta, row['Pass'], row['Fail'], f"{row['Pass Ratio']:.2f}"]
        for delta, row in global_stats.iterrows()
    ]
    return create_table(data, "Global Statistics")

def generate_global_pdf_report(df, file_path):
    """
    Generate a global PDF report from the aggregated DataFrame.

    :param df: DataFrame containing the aggregated test results from all prompts.
    :param file_path: Path where the global PDF report will be saved.
    """
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    elements, styleSheet = [], getSampleStyleSheet()

    elements.append(Paragraph("<b>Global Analysis Report</b>", styleSheet['Title']))
    elements.append(Spacer(1, 12))

    # Global Statistics Table
    elements.append(Paragraph(f"Delta Performance", styleSheet['Heading2']))
    global_stats = calculate_global_stats(df)
    global_stats_table = create_global_stats_table(global_stats)
    elements.append(global_stats_table)
    elements.append(Spacer(1, 12))

    # Error Frequency Table
    elements.append(Paragraph(f"Error Frequency", styleSheet['Heading2']))
    error_freq_table = error_frequency_table(df)
    elements.append(error_freq_table)
    elements.append(Spacer(1, 12))

    # Most Common Error per Delta Table
    elements.append(Paragraph(f"Most Common Error per Delta", styleSheet['Heading2']))
    common_error_delta_table = most_common_error_per_delta_table(df)
    elements.append(common_error_delta_table)
    elements.append(Spacer(1, 12))

    # Deltas Key Table
    deltas_key_table = create_deltas_key_table(df)
    elements.append(PageBreak())
    elements.append(Paragraph("<b>Deltas Key</b>", styleSheet['Heading1']))
    elements.append(deltas_key_table)

    doc.build(elements)