import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
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

def calculate_delta_pass_frequency(df, total_prompts, runs_per_prompt):
    """
    Calculate the frequency of each delta passing in all runs for each prompt.

    :param df: DataFrame containing the test results.
    :param total_prompts: Total number of prompts.
    :param runs_per_prompt: Number of runs per prompt.
    :return: DataFrame with delta and its corresponding pass frequency.
    """
    pass_counts = df[df['Pass/Fail'] == 'Pass'].groupby(['Prompt', 'Delta']).size()
    all_pass = pass_counts[pass_counts == runs_per_prompt].reset_index().groupby('Delta').size()
    delta_frequency = pd.DataFrame(all_pass, columns=['Frequency']).reset_index()
    return delta_frequency

def create_delta_pass_frequency_table(df, total_prompts, runs_per_prompt):
    """
    Create a table for the frequency of each delta passing in all runs for each prompt.

    :param df: DataFrame containing the test results.
    :param total_prompts: Total number of prompts.
    :param runs_per_prompt: Number of runs per prompt.
    :return: Table object.
    """
    delta_frequency = calculate_delta_pass_frequency(df, total_prompts, runs_per_prompt)
    data = [['Delta', 'Frequency']] + delta_frequency.values.tolist()

    return create_table(data, "Delta Pass Frequency Table")

def calculate_delta_fail_frequency(df, total_prompts, runs_per_prompt):
    """
    Calculate the frequency of each delta failing in all runs for each prompt.

    :param df: DataFrame containing the test results.
    :param total_prompts: Total number of prompts.
    :param runs_per_prompt: Number of runs per prompt.
    :return: DataFrame with delta and its corresponding fail frequency.
    """
    fail_counts = df[df['Pass/Fail'] == 'Fail'].groupby(['Prompt', 'Delta']).size()
    all_fail = fail_counts[fail_counts == runs_per_prompt].reset_index().groupby('Delta').size()
    delta_frequency = pd.DataFrame(all_fail, columns=['Frequency']).reset_index()
    return delta_frequency

def create_delta_fail_frequency_table(df, total_prompts, runs_per_prompt):
    """
    Create a table for the frequency of each delta failing in all runs for each prompt.

    :param df: DataFrame containing the test results.
    :param total_prompts: Total number of prompts.
    :param runs_per_prompt: Number of runs per prompt.
    :return: Table object.
    """
    delta_frequency = calculate_delta_fail_frequency(df, total_prompts, runs_per_prompt)
    data = [['Delta', 'Frequency']] + delta_frequency.values.tolist()

    return create_table(data, "Delta Fail Frequency Table")

def calculate_delta_pass_some_runs_frequency(df, total_prompts, runs_per_prompt):
    """
    Calculate the frequency of each delta passing in some runs for each prompt.

    :param df: DataFrame containing the test results.
    :param total_prompts: Total number of prompts.
    :param runs_per_prompt: Number of runs per prompt.
    :return: DataFrame with delta and its corresponding frequency of passing in some runs.
    """
    pass_counts = df[df['Pass/Fail'] == 'Pass'].groupby(['Prompt', 'Delta']).size()
    some_pass = pass_counts[(pass_counts > 0) & (pass_counts < runs_per_prompt)].reset_index().groupby('Delta').size()
    delta_frequency = pd.DataFrame(some_pass, columns=['Frequency']).reset_index()
    return delta_frequency

def create_delta_pass_some_runs_frequency_table(df, total_prompts, runs_per_prompt):
    """
    Create a table for the frequency of each delta passing in some runs for each prompt.

    :param df: DataFrame containing the test results.
    :param total_prompts: Total number of prompts.
    :param runs_per_prompt: Number of runs per prompt.
    :return: Table object.
    """
    delta_frequency = calculate_delta_pass_some_runs_frequency(df, total_prompts, runs_per_prompt)
    data = [['Delta', 'Frequency']] + delta_frequency.values.tolist()

    return create_table(data, "Delta Pass in Some Runs Frequency Table")

# Function to create a plot and return it as an Image object
def create_plot_image(plt_function):
    buffer = BytesIO()
    plt_function()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return Image(buffer)

# Function to plot Delta Performance
def delta_performance_plot(global_stats):
    def plt_function():
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(global_stats.index, global_stats['Pass'], label='Pass', color='green')
        ax1.bar(global_stats.index, global_stats['Fail'], bottom=global_stats['Pass'], label='Fail', color='red')
        ax1.set_xlabel('Delta')
        ax1.set_ylabel('Count')
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(global_stats.index, global_stats['Pass Ratio'], label='Pass Ratio', color='blue', marker='o')
        ax2.set_ylabel('Pass Ratio')
        ax2.legend(loc='upper right')
        plt.tight_layout()
    return create_plot_image(plt_function)

# Function to plot Delta Frequency
def delta_frequency_plot(df, title):
    def plt_function():
        plt.figure(figsize=(6, 4))
        plt.bar(df['Delta'], df['Frequency'])
        plt.xlabel('Delta')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.tight_layout()  
    return create_plot_image(plt_function)

# Function to plot Error Frequency
def error_frequency_plot(df):
    def plt_function():
        plt.figure(figsize=(6, 4))
        plt.barh(df['Error Type'], df['Frequency'])
        plt.xlabel('Frequency')
        plt.ylabel('Error Type')
        plt.title('Error Frequency')
        plt.tight_layout()  
    return create_plot_image(plt_function)

def generate_global_pdf_report(df, file_path, total_prompts, runs_per_prompt):
    """
    Generate a global PDF report from the aggregated DataFrame.

    :param df: DataFrame containing the aggregated test results from all prompts.
    :param file_path: Path where the global PDF report will be saved.
    """
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    elements, styleSheet = [], getSampleStyleSheet()

    elements.append(Paragraph("<b>Global Analysis Report</b>", styleSheet['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Tables</b>", styleSheet['Heading1']))
    elements.append(Spacer(1, 12))

    # Calculate global statistics
    global_stats = calculate_global_stats(df)

    # Global Statistics Table
    elements.append(Paragraph(f"Delta Performance ({total_prompts} Prompts, {runs_per_prompt} Runs Each)", styleSheet['Heading2']))
    global_stats_table = create_global_stats_table(global_stats)
    elements.append(global_stats_table)
    elements.append(Spacer(1, 12))

    # Delta Pass Frequency Table
    elements.append(Paragraph(f"Frequency of Deltas that Passed in All Runs", styleSheet['Heading2']))
    delta_pass_freq_table = create_delta_pass_frequency_table(df, total_prompts, runs_per_prompt)
    elements.append(delta_pass_freq_table)
    elements.append(Spacer(1, 12))

    # Delta Pass in Some Runs Frequency Table
    elements.append(Paragraph(f"Frequency of Deltas that Passed in Some Runs", styleSheet['Heading2']))
    delta_pass_some_runs_freq_table = create_delta_pass_some_runs_frequency_table(df, total_prompts, runs_per_prompt)
    elements.append(delta_pass_some_runs_freq_table)
    elements.append(Spacer(1, 12))

    # Delta Fail Frequency Table
    elements.append(Paragraph(f"Frequency of Deltas that Failed in All Runs", styleSheet['Heading2']))
    delta_fail_freq_table = create_delta_fail_frequency_table(df, total_prompts, runs_per_prompt)
    elements.append(delta_fail_freq_table)
    elements.append(Spacer(1, 12))

    # Error Frequency Table
    elements.append(Paragraph(f"Frequency of Error Types", styleSheet['Heading2']))
    error_freq_table = error_frequency_table(df)
    elements.append(error_freq_table)
    elements.append(Spacer(1, 12))

    # Most Common Error per Delta Table
    elements.append(Paragraph(f"Most Common Error Type per Delta", styleSheet['Heading2']))
    common_error_delta_table = most_common_error_per_delta_table(df)
    elements.append(common_error_delta_table)
    elements.append(Spacer(1, 12))

    elements.append(PageBreak())
    elements.append(Paragraph("<b>Plots</b>", styleSheet['Heading1']))

    # Delta Performance Plot
    elements.append(Paragraph(f"Delta Performance Plot", styleSheet['Heading2']))
    delta_performance_plot_image = delta_performance_plot(global_stats)
    elements.append(delta_performance_plot_image)

    # Delta Pass in All Runs Frequency Plot
    elements.append(PageBreak())
    elements.append(Paragraph(f"Frequency of Deltas that Passed in All Runs Plot", styleSheet['Heading2']))
    delta_pass_freq_df = calculate_delta_pass_frequency(df, total_prompts, runs_per_prompt)
    delta_pass_freq_plot_image = delta_frequency_plot(delta_pass_freq_df, "Delta Pass in All Runs Frequency")
    elements.append(delta_pass_freq_plot_image)
    elements.append(Spacer(1, 12))

    # Delta Pass in Some Runs Frequency Plot
    elements.append(PageBreak())
    elements.append(Paragraph(f"Frequency of Deltas that Passed in Some Runs Plot", styleSheet['Heading2']))
    delta_pass_some_runs_freq_df = calculate_delta_pass_some_runs_frequency(df, total_prompts, runs_per_prompt)
    delta_pass_some_runs_freq_plot_image = delta_frequency_plot(delta_pass_some_runs_freq_df, "Delta Pass in Some Runs Frequency")
    elements.append(delta_pass_some_runs_freq_plot_image)
    elements.append(Spacer(1, 12))

    # Delta Fail in All Runs Frequency Plot
    elements.append(PageBreak())
    elements.append(Paragraph(f"Frequency of Deltas that Failed in All Runs Plot", styleSheet['Heading2']))
    delta_fail_freq_df = calculate_delta_fail_frequency(df, total_prompts, runs_per_prompt)
    delta_fail_freq_plot_image = delta_frequency_plot(delta_fail_freq_df, "Delta Fail in All Runs Frequency")
    elements.append(delta_fail_freq_plot_image)
    elements.append(Spacer(1, 12))

    # Error Frequency Plot
    elements.append(PageBreak())
    elements.append(Paragraph(f"Frequency of Error Types Plot", styleSheet['Heading2']))
    error_freq_df = df[df['Pass/Fail'] == 'Fail']['Error Type'].value_counts().reset_index()
    error_freq_df.columns = ['Error Type', 'Frequency']
    error_freq_plot_image = error_frequency_plot(error_freq_df)
    elements.append(error_freq_plot_image)
    elements.append(Spacer(1, 12))

    # Deltas Key Table
    deltas_key_table = create_deltas_key_table(df)
    elements.append(PageBreak())
    elements.append(Paragraph("<b>Deltas Key</b>", styleSheet['Heading1']))
    elements.append(deltas_key_table)

    doc.build(elements)