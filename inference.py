from huggingface_hub import InferenceClient
import torch
from transformers import pipeline

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

# This was adapted from Yang's code and modified to not keep track of history and made compatible with Streamlit
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