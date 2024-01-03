from openai import OpenAI
import google.generativeai as genai
import google_api_key
from llama_cpp import Llama

genai.configure(api_key = google_api_key.GOOGLE_API_KEY)

# Models mapping
model_dict = {
    'OpenAI_3.5T': 'gpt-3.5-turbo',
    'OpenAI_4': 'gpt-4',
    'OpenAI_4T': 'gpt-4-1106-preview',
    'Google_GemP': 'gemini-pro',
    'Mistral_7B': 'mistral-7b-v0.2-ggml-model-f16.bin',
    'CodeLlama_7B_Python': 'codellama-7b-python-ggml-model-f16.bin',
    'CodeLlama_13B_Python': 'codellama-13b-python-ggml-model-f16.bin'
}

def generate_openai_output(delta, llm):
    question = f"{delta}"
    client = OpenAI()
    response = client.chat.completions.create(
        model=llm,
        messages=[{'role': 'user', 'content': question}],
        max_tokens=150,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    return answer

def generate_google_output(delta, llm):
    question = f"{delta}"
    model=llm
    client = genai.GenerativeModel(model)
    response = client.generate_content(question)
    try:
        answer = response.text
    except:
        print(response.prompt_feedback)
    return answer

def generate_local_llm_output(delta, llm):
    question = f"{delta}"
    client = Llama(model_path=llm)
    response = client(
        question,
        max_tokens=150,
        echo=True
    )
    answer = response["choices"][0]["text"]
    return answer

def generate_model_output(delta, model):
    llm = model_dict[model]
    if model.startswith('OpenAI'):
        return generate_openai_output(delta, llm)
    elif model.startswith('Google'):
        return generate_google_output(delta, llm)
    elif model.startswith('Mistral') or model.startswith('CodeLlama'):
        return generate_local_llm_output(delta, llm)
    else:
        return "Invalid model"