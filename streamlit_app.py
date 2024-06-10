import streamlit as st
import torch
import subprocess
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to install required packages
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

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Log in to Hugging Face
hf_token = st.secrets["HF_TOKEN"]
login(hf_token)

# App title
st.set_page_config(page_title="MindMate 🧠")
logger.info("Streamlit app started")

# Base model and adapter model paths
base_model = 'meta-llama/Llama-2-7b-chat-hf'
adapter_model = "Mental-Health-Chatbot"  # Path to your adapter model directory

# Load model and tokenizer with caching for performance
@st.cache_resource
def load_model_and_tokenizer(base_model, adapter_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading model on device: {device}")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(model, adapter_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer(base_model, adapter_model)
logger.info("Model and tokenizer loaded successfully")

# Sidebar for API and parameters
with st.sidebar:
    st.title('MindMate 🧠')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    
    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('Max Length', min_value=32, max_value=128, value=100, step=8)

    st.button('Clear Chat History', on_click=lambda: logger.info("Chat history cleared"))

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    logger.info("Chat history cleared")
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_llama2_response(prompt_input):
    logger.info(f"Generating response for input: {prompt_input}")
    # Prepare the input dialogue
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.\n\n"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    # Add the latest user input
    string_dialogue += f"User: {prompt_input}\n\nAssistant: "

    # Tokenize the input text
    input_ids = tokenizer.encode(string_dialogue, return_tensors="pt").to(device)
    logger.info("Input text tokenized")

    # Generate the output text
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, top_p=top_p, repetition_penalty=1.0)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Generated response: {response}")

    return response

# User-provided prompt
if prompt := st.chat_input(disabled=not st.secrets["HF_TOKEN"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    logger.info(f"User input: {prompt}")

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(st.session_state.messages[-1]["content"])
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    logger.info("Assistant response added to session state")
