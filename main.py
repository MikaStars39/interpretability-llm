import torch
from transformers import AutoTokenizer
from src.modeling_llama import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "/home/qingyu_yin/model/llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
    )
tokenizer = AutoTokenizer.from_pretrained("/home/qingyu_yin/model/llama-2-7b-hf")

prompt = "Large Language Models (LLMs) are deep learning models that can efficiently process and understand \
    natural language text. One challenge in the application of LLMs is their enormous number of parameters. "
inputs = tokenizer(prompt, return_tensors="pt").to("cuda").to(torch.float16)
inputs["labels"] = inputs["input_ids"]
outputs = model.forward(**inputs)
base_loss = outputs.loss

for skip in range(1, 32):
    from src.functions import test_qa
    
    model.model.skip_list = [skip]
    model.model.skip_from = "direct"

    outputs = model.forward(**inputs)

    print(outputs.loss - base_loss)