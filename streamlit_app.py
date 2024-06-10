import streamlit as st
import replicate
import os
import torch
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import transformers
    import peft
    import huggingface_hub
except ImportError:
    install("transformers")
    install("peft")
    install("huggingface_hub")

# Import necessary libraries after installation
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Log in to Hugging Face
hf_token = st.secrets["HF_TOKEN"]
login(hf_token)
# App title
st.set_page_config(page_title="MindMate 🧠")

# Base model and adapter model paths
base_model = 'meta-llama/Llama-2-7b-chat-hf'
adapter_model = "Mental-Health-Chatbot"  # Path to your adapter model directory

@st.cache_resource
def load_model_and_tokenizer(base_model, adapter_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(model, adapter_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer(base_model, adapter_model)
# Replicate Credentials
with st.sidebar:
    st.title('MindMate 🧠')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Parameters')
    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('Max Length', min_value=32, max_value=128, value=100, step=8)


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    input_ids = tokenizer.encode(f"{string_dialogue} {prompt_input} Assistant: ", return_tensors="pt").to(device)
    output = model.generate(input_ids, max_new_tokens=max_length, temperature=temperature, top_p=top_p, repetition_penalty=1.0)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("Assistant: ")[-1].strip()  # Extract the relevant part of the response
    return response

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(st.session_state.messages[-1]["content"])
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})