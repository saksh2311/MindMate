import streamlit as st
import os
import torch
import subprocess
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Load model and tokenizer (ensure GPU usage if available)
@st.cache_resource
def load_model_and_tokenizer(base_model, adapter_model, hf_token):
    from huggingface_hub import login
    # Log in to Hugging Face
    login(hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on device: {device}")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(model, adapter_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    logger.info("Model and tokenizer loaded correctly")
    return model, tokenizer, device

def main():
    # Initialize model and tokenizer
    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    adapter_model = "Mental-Health-Chatbot"  # Path to your adapter model directory

    hf_token = st.secrets["HF_TOKEN"]
    model, tokenizer, device = load_model_and_tokenizer(base_model, adapter_model, hf_token)

    # Streamlit UI
    st.title("MindMate 🧠")

    st.write("## Chat with the MindMate Assistant")
    st.write("Enter your message below and get a response from the assistant.")

    prompt = st.text_input("Enter your message:")
    if st.button("Get Response"):
        if prompt:
            response = generate_llama2_response(model, tokenizer, device, prompt)
            st.write("### Assistant Response:")
            st.write(response)
        else:
            st.warning("Please enter a message to get a response.")
            
@st.cache
def generate_llama2_response(model, tokenizer, device, prompt_input):
    input_ids = tokenizer.encode(prompt_input, return_tensors="pt").to(device)
    logger.info("Input text tokenized")
    output = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Generated response: {response}")
    return response

# Entry point for the script
if __name__ == "__main__":
    main()
