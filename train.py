# import gpt2-small

from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import time
import torch

from src.modeling_gptneo import GPTNeoForCausalLM
from src.utils import draw_rv_figure

def tokenize_and_format(examples):
    # 对输入进行编码
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
    
    # 为因果语言模型准备标签，将输入数据偏移一位作为标签
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
    
    return tokenized_inputs


model = GPTNeoForCausalLM.from_pretrained("/code/gpt-neo-1.3B")
tokenizer = AutoTokenizer.from_pretrained("/code/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("/code/minipile", split="test")
tokenized_dataset = dataset.map(tokenize_and_format, batched=True)


# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    warmup_steps=100,
    learning_rate = 5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
)

# 创建训练器并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()


