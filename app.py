import streamlit as st
from huggingface_hub import InferenceClient
import torch
from transformers import pipeline
import re

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

def respond(
    message,
    system_message="You are a friendly Chatbot who paraphrases text.",
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
        print(f"Paraphrased Text (Local Model): {response}")  # Debug log
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
        print(f"Paraphrased Text (API Model): {response}")  # Debug log
        return response


def cancel_inference():
    global stop_inference
    stop_inference = True

def main():
    st.markdown('''<h3>Paraphraser</h3>''', unsafe_allow_html=True)

    system_message = st.text_area("System message", "You are a friendly Chatbot who paraphrases text.")
    use_local_model = st.checkbox("Use local model", value=False)

    max_tokens = st.slider('Max new tokens', 1, 2048, 512)
    temperature = st.slider('Temperature', 0.1, 4.0, 0.7)
    top_p = st.slider('Top-p (nucleus sampling)', 0.1, 1.0, 0.95)

    input_txt = st.text_area("Enter the text to paraphrase:", "", height=150)

    paraphrased_txt = None

    if st.button("Submit"):
        input_txt = re.sub(r'\n+', ' ', input_txt)  # Clean the input text
        
        with st.spinner("Processing..."):
            paraphrased_txt = respond(input_txt, system_message=system_message, max_tokens=max_tokens, temperature=temperature, top_p=top_p, use_local_model=use_local_model)

        if paraphrased_txt:
            st.success("Text successfully paraphrased!")
        else:
            st.error("Failed to paraphrase the text.")

    # Debug log the paraphrased text
    print(f"Paraphrased Text for Debug: {paraphrased_txt}")  # Debug log

    if paraphrased_txt:
        st.text_area("Paraphrased Text:", paraphrased_txt, height=150)

    if st.button("Cancel Inference"):
        cancel_inference()


if __name__ == "__main__":
    main()

# import streamlit as st
# from huggingface_hub import InferenceClient
# import torch
# from transformers import pipeline
# import re

# # Inference client setup
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
# pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# # Global flag to handle cancellation
# stop_inference = False

# def respond(
#     message,
#     # history: list[tuple[str, str]],
#     system_message="You are a friendly Chatbot who paraphrases text.",
#     # max_tokens=512,
#     temperature=0.7,
#     # top_p=0.95,
#     use_local_model=False,
# ):
#     global stop_inference
#     stop_inference = False  # Reset cancellation flag

#     # Initialize history if it's None
#     # if history is None:
#     #     history = []

#     if use_local_model:
#         # local inference 
#         messages = [{"role": "system", "content": system_message}]
#         # for val in history:
#         #     if val[0]:
#         #         messages.append({"role": "user", "content": val[0]})
#         #     if val[1]:
#         #         messages.append({"role": "assistant", "content": val[1]})
#         messages.append({"role": "user", "content": message})

#         response = ""
#         for output in pipe(
#             messages,
#             # max_new_tokens=max_tokens,
#             temperature=temperature,
#             do_sample=True,
#             # top_p=top_p,
#         ):
#             if stop_inference:
#                 response = "Inference cancelled."
#                 yield [(message, response)]
#                 return
#             token = output['generated_text'][-1]['content']
#             response += token
#         return response
#             # yield [(message, response)]  # Yield history + new response

#     else:
#         # API-based inference 
#         messages = [{"role": "system", "content": system_message}]
#         # for val in history:
#         #     if val[0]:
#         #         messages.append({"role": "user", "content": val[0]})
#         #     if val[1]:
#         #         messages.append({"role": "assistant", "content": val[1]})
#         messages.append({"role": "user", "content": message})

#         response = ""
#         for message_chunk in client.chat_completion(
#             messages,
#             # max_tokens=max_tokens,
#             stream=True,
#             temperature=temperature,
#             # top_p=top_p,
#         ):
#             if stop_inference:
#                 response = "Inference cancelled."
#                 yield [(message, response)]
#                 return
#             if stop_inference:
#                 response = "Inference cancelled."
#                 break
#             token = message_chunk.choices[0].delta.content
#             response += token
#             # yield [(message, response)]  # Yield history + new response
#         return response


# def cancel_inference():
#     global stop_inference
#     stop_inference = True


# def main():
#     st.markdown('''<h3>Paraphraser</h3>''', unsafe_allow_html=True)

#     # Number of sentences and temperature sliders
#     n_sents = st.slider('Select the number of sentences to process', 5, 30, 10)
#     temperature = st.slider('Enter the temperature', 0.0, 1.0, 0.7)  # Adjusted default temperature to 0.7
#     sys_message = st.text_area("Modify the system message if you want:", "You are a friendly Chatbot who paraphrases text.", height=10)
#     local_model = st.checkbox("Use local model", value=False)

#     input_txt = st.text_area("Enter the text. (Ensure the text is grammatically correct and has punctuations at the right places):", "", height=150)

#     if st.button("Submit") and input_txt:
#         with st.status("Processing...", expanded=True) as status:
#             input_txt = re.sub(r'\n+', ' ', input_txt)

#             paraphrased_txt = ""
#             paraphrase_error = None

#             # Paraphrasing start
#             try:
#                 st.info("Rewriting the text. This takes time.", icon="ℹ️")

#                 # Iterate over the generator to collect responses
#                 for resp in respond(input_txt, system_message=sys_message, temperature=temperature, use_local_model=local_model):
#                     paraphrased_txt += resp  # Accumulate the generated tokens

#             except Exception as e:
#                 paraphrased_txt = None
#                 paraphrase_error = str(e)
            
#             # Paraphrasing end
#             if paraphrased_txt is not None:
#                 st.success("Successfully rewrote the text.", icon="✅")
#                 st.text_area("Paraphrased Text:", paraphrased_txt, height=150)  # Display the paraphrased output
#                 print(paraphrased_txt)
#             else:
#                 st.error(f"Encountered an error: {paraphrase_error}", icon="🚨")

#             if paraphrase_error is None:
#                 status.update(label="Done", state="complete", expanded=False)
#             else:
#                 status.update(label="Error", state="error", expanded=False)


# if __name__ == "__main__":
#     main()
