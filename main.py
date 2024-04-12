import torch
from transformers import AutoTokenizer
from src.modeling_llama import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("/home/qingyu_yin/model/llama-2-7b-hf").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("/home/qingyu_yin/model/llama-2-7b-hf")

text = "Based on our previous analysis, the role of the self-attention mechanism within the deep layers is significant, as they possess the capability to aggregate information across various positions."

inputs = tokenizer(text, return_tensors="pt").to("cuda")

inputs["labels"] = inputs["input_ids"]
model.model.skip_list = [24, 25, 26, 27, 28]
model.model.skip_from = 1
print(model(**inputs).loss)