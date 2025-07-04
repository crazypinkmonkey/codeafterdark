from llama_cpp import Llama

model_path = "C:/Users/Aditya/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
llm = Llama(model_path=model_path, n_ctx=512)

prompt = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nWhat is the capital of India?<|end|>\n<|assistant|>"

output = llm(prompt)

print(output["choices"][0]["text"].strip())