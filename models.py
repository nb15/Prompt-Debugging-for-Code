#from openai import OpenAI
#import google.generativeai as genai
#import google_api_key
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import BartForConditionalGeneration
import torch, os

#genai.configure(api_key = google_api_key.GOOGLE_API_KEY)

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
    'hf_wizardcoder_15B': 'WizardLM/WizardCoder-15B-V1.0',
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


def generate_wizardcoder_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION


def get_hf_model(llm, temperature, max_len, greedy_decode, decoding_style, load_8bit=True):
    model_name = model_dict[llm]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            #load_in_8bit = load_8bit,
            #torch_dtype = torch.float16,
            torch_dtype = "auto",
            device_map = "auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )        
    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()
    
    model.eval()
    model = torch.compile(model)

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False if greedy_decode else True,
        temperature=temperature,
        max_length=max_len,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.95
    )

    return model, tokenizer, generation_config



def generate_wizardcode_output(delta, model, tokenizer, generation_config, max_len):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prompt = delta.replace('    ', '\t')
    prompt = [generate_wizardcoder_prompt(prompt)]
    
    encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    temp = model(**encoding, output_hidden_states=True)
    model_out = model.generate(
        **encoding,
        generation_config = generation_config
    )

    decoded_seq = tokenizer.batch_decode(model_out, skip_special_tokens = True)[0]
    trucated_seq = decoded_seq.split("### Response:")[1]
    trucated_seq = trucated_seq.replace('\t', '    ')
    raw_seq = decoded_seq.replace('\t', '    ')

    return trucated_seq, raw_seq

def generate_huggingface_output(delta, llm):
    tokenizer = AutoTokenizer.from_pretrained(llm)
    model = AutoModelForCausalLM.from_pretrained(llm)
    if 'incoder' in llm:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.padding_side = "right"
    else:
        raise NotImplementedError
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.half()
    model.to(device)
    model.eval()

    input_ids = tokenizer([delta], return_tensors="pt").input_ids.to(device)
    current_length = input_ids.flatten().size(0)

    max_len = 256
    model_output = model.generate(input_ids=input_ids,
                                max_length = max_len,
                                num_return_sequences = 1,
                                return_dict_in_generate = True,
                                output_scores = False,
                                top_p = 0.95,
                                temperature = 0.001,
                                num_beams=1,
                                )
    
    model_output_seq = model_output["sequences"][0, current_length:]

    def incoder_post_process(stop_words, trg_prediction):
        for stop_word in stop_words:
            if stop_word in trg_prediction:
                trg_prediction = trg_prediction.split(stop_word)[0]
        return trg_prediction

    if 'incoder' in llm:
        num_non_pad = model_output_seq.ne(1).sum()
        model_output_seq = model_output_seq[:num_non_pad]
        preds = tokenizer.decode(model_output_seq)
        preds = incoder_post_process(["</code>", "# SOLUTION END"], preds)
    else:
        raise NotImplementedError
    
    return preds
    

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