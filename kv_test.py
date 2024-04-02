import time
import torch
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from src.modeling_gptneo import GPTNeoForCausalLM
from src.utils import draw_rv_figure, draw_attention_maps

kv_prompt = "I will give you some keys and values. Then I ask you a key, you respones me with its value. "
example = ""

def test_dataset():
    dataset = load_dataset("json", data_files="kv_test/kv_pairs_100_100.json", split="train")
    print(len(next(iter(dataset))["0"][0]))


@torch.no_grad()
def test_kv(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset = load_dataset("json", data_files=args.data_name_or_path, split="train")

    length = len(tokenizer.encode(next(iter(dataset))["0"][0]))
    total_num = args.context_len // length - 4

    acc = [ 0 for _ in range(0, total_num, args.span)]

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for ids, query in tqdm(enumerate(dataloader)):
        query = dataset[f"{ids}"][0]
        input_text = kv_prompt
        for each_query_id in range(total_num):
            each_query = query[each_query_id]
            input_text += each_query + "; "
        for position in range(0, total_num, args.span):
            prompt = "Question: the key is " + query[position][5:21] + ", so the value is"
            input_text_now = input_text + prompt

            inputs = tokenizer(input_text_now, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs["input_ids"], max_length=inputs["input_ids"].size(1) + 16, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
            outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if outputs[len(input_text_now)+1:len(input_text_now)+17] == query[position][-16:]:
                acc[position // args.span] += 1
        print(acc)
    
    acc = torch.tensor(acc) / len(dataset)
    print(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="/home/qingyu_yin/model/gpt-neo-1.3B"
    )
    parser.add_argument("--data_name_or_path", type=str, default="kv_test/kv_pairs_100_100.json")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    test_kv(args)
    # test_dataset()
