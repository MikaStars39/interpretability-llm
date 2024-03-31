# import gpt2-small

from transformers import AutoTokenizer
from datasets import load_dataset
import time
import torch

from src.modify_gptneo import GPTNeoForCausalLM
from src.utils import draw_rv_figure


model = GPTNeoForCausalLM.from_pretrained("/code/gpt-neo-1.3B", output_attentions=True).to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("/code/gpt-neo-1.3B")

dataset = load_dataset("emozilla/pg19-test")
print(dataset)
itp_all = []
with torch.no_grad():
    for text in iter(dataset['test']):

        inputs = tokenizer(text['text'], return_tensors="pt", max_length=1024, truncation=True).to("cuda")
        outputs, itp_output= model(**inputs)

        # # decode
        # decoded = tokenizer.decode(outputs.logits[0].argmax(-1))

        # itp_output
        if len(itp_all) == 0:
            itp_all = itp_output
            for layer in range(0, len(itp_all)):
                for item in range(0, len(itp_all[0])):
                    itp_all[layer][item] = itp_output[layer][item] / len(dataset)
        else:
            for layer in range(0, len(itp_all)):
                for item in range(0, len(itp_all[0])):
                    itp_all[layer][item] += itp_output[layer][item] / len(dataset)

# [input, att_output, att_weight, residual, mlp_output]
# layer item batch length hidden_dim

# attention
draw_rv_figure(
    hidden_state = itp_output,
    heads_num = 16,
    heads_dim = 2048 // 16,
    layers = 24,
    fig_path = f"figure/figure-{time.ctime(time.time())}",
    type_a = 0,
    type_b = 4,
    )