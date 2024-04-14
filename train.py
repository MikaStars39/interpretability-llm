# import gpt2-small
import time
import torch
import argparse
import os

from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.distributed import barrier

from src.modeling_qwen2 import Qwen2ForCausalLM


def train(args):
    model_name = "/home/qingyu_yin/model/Qwen1.5-1.8B"
    
    model_config = AutoConfig.from_pretrained(model_name)
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=torch.float16
    )

    model.model.skip_list = [20]
    model.model.skip_from = "linear"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=args.context_len,
        padding_side="right",
        use_fast=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    def tokenize_and_format(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"] 
        return tokenized_inputs

    dataset = load_dataset(args.data_name_or_path, "wikitext-103-v1", split="test")
    dataset = dataset.map(tokenize_and_format, batched=True, num_proc=16)

    if rank == 0:
        barrier()

    print(dataset)

    training_args = TrainingArguments(
        output_dir="./results/" + str(time.time()),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        warmup_steps=args.warmup_steps,
        learning_rate = args.learning_rate,
        logging_steps=1,
        fp16=True,
        fp16_backend="amp",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="/home/qingyu_yin/model/gpt-neo-125m"
    )
    parser.add_argument("--data_name_or_path", type=str, default="/home/qingyu_yin/data/wikitext")
    parser.add_argument("--model_type", type=str, default="Auto")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4) 
    parser.add_argument("--warmup_steps", type=int, default=200)
    args = parser.parse_args()

    os.system("export TOKENIZERS_PARALLELISM=true")

    train(args)

