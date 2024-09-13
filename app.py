import streamlit as st
from huggingface_hub import InferenceClient
import torch
from transformers import pipeline
import re
import time

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

# This was adapted from Professor's sample code with modifications for compatibility with Streamlit 
def respond(
    message,
    system_message="You are a bot that paraphrases text.",
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    use_local_model=False,
):
    global stop_inference
    stop_inference = False  # Reset cancellation flag

    if use_local_model:
        # Local inference
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": message})

        response = ""
        for output in pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        ):
            if stop_inference:
                return "Inference cancelled."
            token = output['generated_text'][-1]['content']
            response += token
        
        return response

    else:
        # API-based inference
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": message})

        response = ""
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if stop_inference:
                return "Inference cancelled."
            token = message_chunk.choices[0].delta.content
            response += token
        
        return response


def cancel_inference():
    global stop_inference
    stop_inference = True

# This was written by ChatGPT with the prompt "Create a Streamlit interface for a text paraphraser."
def main():
    st.markdown('''<h3 style="text-align:center;">Text Paraphraser</h3>''', unsafe_allow_html=True)

    system_message = st.text_input("System message", "You are a bot that paraphrases text.")
    use_local_model = st.checkbox("Use local model", value=False)
    st.write("<center>", use_local_model, "</center>", unsafe_allow_html=True)

    max_tokens = st.slider('Max new tokens', 1, 2048, 512)
    temperature = st.slider('Temperature', 0.1, 1.0, 0.7)
    top_p = st.slider('Top-p (nucleus sampling)', 0.1, 1.0, 0.95)

    input_txt = st.text_area("Enter the text to paraphrase:", "", height=150)

    paraphrased_txt = None

    if st.button("Submit"):
        input_txt = re.sub(r'\n+', ' ', input_txt)  # Clean the input text
        
        start_time = time.time()  # Start the stopwatch
        
        with st.spinner("Processing..."):
            if st.button("Cancel"):
                cancel_inference()
            paraphrased_txt = respond(input_txt, system_message=system_message, max_tokens=max_tokens, temperature=temperature, top_p=top_p, use_local_model=use_local_model)
        
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        
        if paraphrased_txt:
            st.success(f"Text successfully paraphrased in {elapsed_time:.2f} seconds!")
        else:
            st.error("Failed to paraphrase the text.")

    if paraphrased_txt:
        st.text_area("Paraphrased Text:", paraphrased_txt, height=150)

if __name__ == "__main__":
    main()
