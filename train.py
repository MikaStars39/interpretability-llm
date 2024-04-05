# import gpt2-small

from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import time
import torch
import argparse
import os

from src.modify_gptneo import GPTNeoForCausalLM
from src.dataset import load_wikitext

def train(args):
    model = GPTNeoForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_format(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"] 
        return tokenized_inputs

    dataset = load_wikitext(data_type="test")
    tokenized_dataset = dataset.map(tokenize_and_format, batched=True)

    training_args = TrainingArguments(
        output_dir="./results/" + str(time.time()),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate = args.learning_rate,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="/home/qingyu_yin/model/gpt-neo-125m"
    )
    parser.add_argument("--data_name_or_path", type=str, default="kv_test/kv_pairs_100_100.json")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4) 
    parser.add_argument("--warmup_steps", type=int, default=200)
    args = parser.parse_args()

    os.system("export TOKENIZERS_PARALLELISM=false")

    train(args)

