import time
import torch
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import AutoTokenizer

from src.modeling_gptneo import GPTNeoForCausalLM
from kv_test.kv_generation import generate_kv


@torch.no_grad()
def test_kv(args):
    model = GPTNeoForCausalLM.from_pretrained(args.model_name_or_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset = generate_kv()
    acc = 0
    perplexity = 0

    for ids, batch in tqdm(enumerate(dataset)):
        answer, query = batch
        inputs = tokenizer(query, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=inputs["input_ids"].size(1) + 10, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )

        input_ids = inputs["input_ids"] 
        outputs_p = model(input_ids[:, :input_ids.size(1)-1], labels=input_ids[:, 1:input_ids.size(1)])
        loss = outputs_p.loss
        perplexity += loss.item() / 49
         
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if answer == outputs[len(query):len(query)+8]:
            acc = acc + 1
    acc = acc / len(dataset)
    print(acc, perplexity)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="/home/qingyu_yin/model/gpt-neo-1.3B"
    )
    parser.add_argument("--data_name_or_path", type=str, default="kv_test/kv_pairs_100_100.json")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--len", type=int, default=16) 
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    test_kv(args)
    
