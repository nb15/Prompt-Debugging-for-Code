# PromptDebuggingForCode

This project is designed to run experiments for debugging code prompts using various large language models (LLMs). Requires Python 3.12 or later.

## Installation

To install the required packages, use the following command:

```pip install -r requirements.txt```

This will install all the necessary packages required for this project.

## File Descriptions and Process Flow

### Main Execution File
- **run_experiments.py**: The primary script that users need to execute. It integrates various components of the project to run and analyze experiments.

### Utility and Supporting Files
- **generate_test_cases.py**: Generates test cases for the experiments. Utilizes `mbpp_adapter.py` and `value_generator.py` for preparing test data.
- **llm.py**: Handles operations related to large language models, such as extracting and processing Python code from model outputs.
- **mbpp_adapter.py**: Provides functionality for parsing function inputs, aiding in test case generation in `generate_test_cases.py`.
- **models.py**: Manages interactions with various models, including OpenAI and Google's Generative AI models. It's crucial for selecting and running different LLMs during experiments.
- **value_generator.py**: Generates dynamic values, primarily used in test case generation in `generate_test_cases.py`.
- **analyze_results.py**: Analyzes the results post-experiment. It processes data to count successes and failures and performs other analyses.

### Process Flow
1. **Preparation Phase**: Test cases are prepared using `generate_test_cases.py`, in conjunction with `mbpp_adapter.py` and `value_generator.py`.
2. **Experiment Phase**: `run_experiments.py` orchestrates the experiment process, leveraging `llm.py` for LLM interactions and `models.py` to select and use different LLMs.
3. **Analysis Phase**: After experiments are conducted, `analyze_results.py` is used to analyze and summarize the outcomes.