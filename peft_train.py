import time
import os
import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from functools import partial
from peft import LoraConfig, TaskType, get_peft_model

from src.modeling_llama import LlamaForCausalLM, LlamaConfig

def tokenize_fn(tokenizer, example):
    # 分词
    example = tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)
    print(example)
    return example

def tokenize_fn_2(tokenizer, example):
    context_length = 512
    outputs = tokenizer(
        example["text"],
        truncation=True,
        max_length=context_length,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length), "labels": outputs["input_ids"].view(-1, context_length)}



if __name__ == "__main__":

    model_name = "/home/qingyu_yin/model/gpt-neo-1.3B"
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
        # 加载数据集
    dataset = load_dataset("/home/qingyu_yin/data/wikitext", "wikitext-103-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = dataset.map(partial(tokenize_fn_2, tokenizer), batched=True, num_proc=16)


    # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0, bias="none",)

    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./results/" + str(time.time()),
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        num_train_epochs=1,
        save_steps=100,
        logging_dir="./logs/" + str(time.time()),
        logging_steps=1,
    )

    # 创建 Trainer 并开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # 开始微调
    trainer.train()



