
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

    # Calculate pass ratio for each delta
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

def calculate_global_stats(df):
    """
    Calculate global statistics from the aggregated DataFrame.

    :param df: DataFrame containing the test results from all prompts.
    :return: A dictionary containing global statistics.
    """
    global_delta_analysis = df.groupby('Delta')['Pass/Fail'].value_counts().unstack(fill_value=0)
    global_delta_analysis['Pass'] = global_delta_analysis.get('Pass', pd.Series(index=global_delta_analysis.index, data=[0]*len(global_delta_analysis)))
    global_delta_analysis['Fail'] = global_delta_analysis.get('Fail', pd.Series(index=global_delta_analysis.index, data=[0]*len(global_delta_analysis)))

    # Calculate pass ratio for each delta
    global_delta_analysis['Pass Ratio'] = global_delta_analysis['Pass'] / (global_delta_analysis['Pass'] + global_delta_analysis['Fail'])

    return global_delta_analysis

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

    global_stats = calculate_global_stats(df)

    # Create table for global statistics
    global_stats_table_data = [["Delta", "Pass", "Fail", "Pass Ratio"]] + [
        [delta, row['Pass'], row['Fail'], f"{row['Pass Ratio']:.2f}"] for delta, row in global_stats.iterrows()
    ]

    global_stats_table = Table(global_stats_table_data)
    global_stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(global_stats_table)

    deltas_key_table = create_deltas_key_table(df)
    elements.append(PageBreak())
    elements.append(Paragraph("<b>Deltas Key</b>", styleSheet['Heading1']))
    elements.append(deltas_key_table)

    # Build the PDF
    doc.build(elements)