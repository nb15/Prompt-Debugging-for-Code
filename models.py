from openai import OpenAI
import google.generativeai as genai
import google_api_key
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

genai.configure(api_key = google_api_key.GOOGLE_API_KEY)

# Models mapping
model_dict = {
    'OpenAI_3.5T': 'gpt-3.5-turbo',
    'OpenAI_4': 'gpt-4',
    'OpenAI_4T': 'gpt-4-1106-preview',
    'Google_GemP': 'gemini-pro',
    'Mistral_7B': 'mistral-7b-v0.2-ggml-model-f16.bin',
    'CodeLlama_7B_Python': 'codellama-7b-python-ggml-model-f16.bin',
    'CodeLlama_13B_Python': 'codellama-13b-python-ggml-model-f16.bin',
    'hf_incoder_1B': "facebook/incoder-1B",
    'hf_starcoderbase_1B': 'bigcode/starcoderbase-1b',
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

def get_hf_model(model):
    llm = model_dict[model]
    tokenizer = AutoTokenizer.from_pretrained(llm)
    model = AutoModelForCausalLM.from_pretrained(llm)
    
    if 'incoder' in llm:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.padding_side = "right"
    else:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.half()
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_huggingface_output(delta, llm, model = None, tokenizer = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(llm)
    
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(llm)
    
        if 'incoder' in llm:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            tokenizer.padding_side = "right"
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        model.to(device)
        model.eval()
    
    if torch.cuda.is_available():
        model.half()

    input_ids = tokenizer([delta], return_tensors="pt").input_ids.to(device)
    current_length = input_ids.flatten().size(0)

    max_len = 1024
    model_output = model.generate(input_ids=input_ids,
                                max_length = max_len,
                                num_return_sequences = 1,
                                return_dict_in_generate = True,
                                output_scores = False,
                                top_p = 0.95,
                                temperature = 0.2,
                                num_beams=1,
                                )
    
    #model_output_seq = model_output["sequences"][0, current_length:]
    model_output_seq = model_output["sequences"][0]

    def incoder_post_process(stop_words, trg_prediction):
        for stop_word in stop_words:
            if stop_word in trg_prediction:
                trg_prediction = trg_prediction.split(stop_word)[0]
        return trg_prediction
    
    def starcoder_postprocess(stop_words, generated_preds, prompt):
        for stop_word in stop_words:
            if stop_word in generated_preds:
                generated_preds = generated_preds.split(stop_word)[0]
        return prompt + generated_preds

    if 'incoder' in llm:
        num_non_pad = model_output_seq.ne(1).sum()
        model_output_seq = model_output_seq[:num_non_pad]
        preds = tokenizer.decode(model_output_seq)
        processed_preds = incoder_post_process(["</code>", "# SOLUTION END"], preds)
    else:
        stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>", "<|endoftext|>"]
        num_non_pad = model_output_seq.ne(tokenizer.eos_token_id).sum()
        model_output_seq = model_output_seq[:num_non_pad]
        preds = tokenizer.decode(model_output_seq[current_length:])
        processed_preds = starcoder_postprocess(stop_words, preds, delta)
        
    
    return preds, processed_preds
    

def generate_model_output(delta, model):
    llm = model_dict[model]
    if model.startswith('OpenAI'):
        return generate_openai_output(delta, llm)
    elif model.startswith('Google'):
        return generate_google_output(delta, llm)
    elif model.startswith('Mistral') or model.startswith('CodeLlama'):
        return generate_local_llm_output(delta, llm)
    elif model.startswith('hf'):
        return generate_huggingface_output(delta, llm)
    else:
        return "Invalid model"