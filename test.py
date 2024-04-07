import time
import torch
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def test_kv(args, model, skip = None):
    from kv_test.kv_generation import generate_kv

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

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

        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(outputs[len(query):len(query)+8])
        if answer == outputs[len(query):len(query)+8]:
            acc = acc + 1
    acc = acc / len(dataset)
    print(acc)
    return acc


@torch.no_grad()
def test_ppl(args, model):
    from src.dataset import load_pg19
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 

    dataloader = load_pg19(batch_size=args.batch_size)

    all_loss = []

    for ids, batch in tqdm(enumerate(dataloader)):
        inputs, labels = batch
        outputs = model(inputs, labels=inputs)
        all_loss.append(float(torch.exp(outputs.loss)))

    print(torch.tensor(all_loss).mean())


@torch.no_grad()
def test_mmlu(args, model):
    from src.dataset import load_mmlu

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataloader = load_mmlu(model_type=args.tokenizer)
    acc = 0
    for ids, batch in tqdm(enumerate(dataloader)):
        query, answer = batch
        outputs = model.generate(
            query, 
            max_length=query.size(1) + 1, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        # print(answer, outputs[0, -1])
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if answer == outputs[0, -1]:
            acc = acc + 1
    acc = acc / 285
    print(acc)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="/home/qingyu_yin/model/gpt-neo-1.3B"
    )
    parser.add_argument("--tokenizer", type=str, default="/home/qingyu_yin/model/gpt-neo-1.3B")
    parser.add_argument("--data_name_or_path", type=str, default="kv_test/kv_pairs_100_100.json")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--len", type=int, default=16) 
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    from src.modeling_gptneo import GPTNeoForCausalLM
    model = GPTNeoForCausalLM.from_pretrained(args.model_name_or_path).to("cuda")
    args.tokenizer = "/home/qingyu_yin/model/gpt-neo-1.3B"
    baseline_a = []
    baseline_m = []
    ours_a = []
    ours_m = []
    for i in range(22, 1, -1):
        model.transformer.skip_list.append(i)  # Set skip_from to the current value of i
        print(model.transformer.skip_list)
        print("Ours:")
        ours_a.append(test_kv(args, model))
        ours_m.append(test_mmlu(args, model))
        print("####")
        print("baseline:")
        model.transformer.skip_from = None
        baseline_a.append(test_kv(args, model))
        baseline_m.append(test_mmlu(args, model))
        model.transformer.skip_from = 24
        print("####")

    # test_mmlu(args, model)
    # test_kv(args, model)
    # test_mmlu(args, model)

    # from src.modeling_llama import LlamaForCausalLM
    # args.model_name_or_path = args.tokenizer
    # model = LlamaForCausalLM.from_pretrained(args.model_name_or_path).to("cuda")
    # model = GPTNeoForCausalLM.from_pretrained(args.model_name_or_path).to("cuda")

    print(ours_a, ours_m, baseline_a, baseline_m)

    # test_kv(args, model)
    test_mmlu(args, model)

    

    
