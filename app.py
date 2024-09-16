import streamlit as st
import re
import time
from inference import respond, cancel_inference

# This was written by ChatGPT with the prompt "Create a Streamlit interface for a text paraphraser."
def main():
    st.markdown('''<h3 style="text-align:center;">Rephrasely</h3>''', unsafe_allow_html=True)
    system_message = st.text_input("System message", "You are a bot that paraphrases text.")

    use_local_model = st.checkbox("Use local model", value=False)

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
