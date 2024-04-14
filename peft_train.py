
from datetime import datetime
# from typing import Optional
import os
import torch
from torch import nn
from dataclasses import dataclass
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    GPT2LMHeadModel,
    TrainingArguments,
)
from src.modeling_qwen2 import Qwen2ForCausalLM

@dataclass
class TrainingArgs:
    model_max_length: int = 1024

def train(args):

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = "/home/qingyu_yin/model/Qwen1.5-1.8B"
    
    model_config = AutoConfig.from_pretrained(model_name)
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

     

    args = TrainingArguments(
        output_dir="cache",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        save_steps=1000,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        max_grad_norm=1,
        logging_steps=1,
        fp16=True,
        report_to="wandb",
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=100,
    )

    print("model loading")
    # model = GPT2Model.from_pretrained(
    #     "gpt2",
    #     torch_dtype=torch.float16,
    # )
    config = AutoConfig.from_pretrained("src/gpt2")
    model = GPT2LMHeadModelFT.from_pretrained(
        "src/gpt2",
        config=config,
        # torch_dtype=torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))
    print("model loaded")

    # freeze the parameter with no "ft" in name
    for name, param in model.named_parameters():
        if "ft" not in name:
            param.requires_grad = False

    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.dtype, torch.isnan(param).any())

    print("begin training")
    trainer = Trainer(
            args=args,
            model=model, tokenizer=tokenizer,
            train_dataset=dataset,
            data_collator=data_collator,
            eval_dataset=None,
    )
    trainer.train()
    trainer.save_model(output_dir="cache")

if __name__ == "__main__":
    args = TrainingArgs()
    train(args)