from transformers import AutoTokenizer, AutoModel
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

model_name = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.to(device)

text = "Hello world"

inputs = tokenizer(text, return_tensors="pt")

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
