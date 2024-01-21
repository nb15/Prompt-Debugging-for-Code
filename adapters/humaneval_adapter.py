from . import general_adapter

def extract_humaneval_docstring(code, function_header, stop_words):
    text = code.split(function_header)[1].strip()
    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0]
    return text.strip().replace('"', '')

def process_humaneval_deltas(test_type, prompt, entry_point, **kwargs):
    
    function_header = general_adapter.extract_function_header(prompt, entry_point)
    text = extract_humaneval_docstring(prompt, function_header, ['Example', 'example', 'For example', 'For Example', '>>>', '>>', f'\n{entry_point}'])
    if test_type == 'original':
        test_list = prompt.split(text)[1].strip()
    else:
        raise NotImplementedError
    
    deltas = [
        f"{text}\n{function_header}\n{test_list}",
        text,
        f"{text}\n{function_header}",
        function_header,
        f"{function_header}\n{test_list}",
        test_list
    ]
    return deltas

def process_humaneval_testcases(test, **kwargs):
    test = [i.strip() for i in test.split('\n') if 'assert' in i]
    return test