import torch
from transformers import AutoTokenizer
from src.modeling_qwen2 import Qwen2ForCausalLM

model = Qwen2ForCausalLM.from_pretrained("/home/qingyu_yin/model/Qwen1.5-1.8B").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("/home/qingyu_yin/model/Qwen1.5-1.8B")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
inputs["labels"] = inputs["input_ids"]

model.model.skip_list = [21, 22, 23, 24]
model.model.skip_from = "linear"

outputs = model.forward(**inputs)

print(outputs.loss)