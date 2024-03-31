# import gpt2-small
import time
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
from matplotlib import pyplot as plt

from src.modify_gptneo import GPTNeoForCausalLM
from src.utils import draw_rv_figure, draw_attention_maps

@dataclass
class Patching_Config:
    heads: list
    head_dim: int
    layers: list
    position: list


def tiny_case(
    model,
    patching_config,
    patching_type):
    
    # head for retention 0, 2, 6, 7, 9, 12

    # corrupted input
    text_corrupt = "Tell me who ate the apple. Context: Today, Mary came to my home and ate an apple. So the person who ate apple was"
    with torch.no_grad():
        inputs = tokenizer(text_corrupt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
        outputs, itp_output, corrupt_storage = model(**inputs, patching_type=patching_type + "_corrupt", patching_config=patching_config)

    draw_attention_maps(outputs.attentions, f"figure/attention-{time.ctime(time.time())}")
    exit()

    target_id = tokenizer.convert_tokens_to_ids("Alice")
    corrupt_id = tokenizer.convert_tokens_to_ids("Mary")
    
    # decode outputs
    # print(tokenizer.decode(outputs.logits[0].argmax(-1)))

    # clean input
    text_clean = "Tell me who ate the apple. Context: Today, Alice came to my home and ate an apple. So the person who ate apple was"
    with torch.no_grad():
        inputs = tokenizer(text_clean, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
        outputs, itp_output, _ = model(**inputs, patching_type=patching_type + "_clean", patching_storage=corrupt_storage, patching_config=patching_config)

    # print(tokenizer.decode(outputs.logits[0].argmax(-1)))

    logits = torch.softmax(outputs.logits, dim=-1)
        # 获取最后一个位置的 logits
    last_logits = logits[:, -1, :]
    # print("Alice", logits[0, -1, target_id], "Mary", logits[0, -1, corrupt_id] )
    
    return logits[0, -1, corrupt_id].item() - logits[0, -1, target_id].item()


model = GPTNeoForCausalLM.from_pretrained("/code/gpt-neo-1.3B", output_attentions=True).to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("/code/gpt-neo-1.3B")
gpt_neo_retention_heads = [3, 4, 5, 10, 11 ,13, 14]
# gpt_neo_retention_heads = [0, 1, 2, 6, 7, 8, 9, 12]
hotmap = []
for layer in tqdm(range(24)):
    temp = []
    for position in range(28):
        patching_config = Patching_Config(
            heads=gpt_neo_retention_heads,
            head_dim= 2048 // 16,
            layers = [layer],
            position = position
        ) 
        result = tiny_case(model, patching_config, "layer")
        temp.append(result)
    hotmap.append(temp)

plt.imshow(torch.tensor(hotmap).detach().numpy(), cmap='Blues')
plt.title(f'Attention')
plt.savefig(f"figure/replace-{time.ctime(time.time())}")
print(f"Successfully draw RV figure")