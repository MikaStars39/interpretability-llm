from transformers import GPTNeoForCausalLM, GPTNeoConfig, Trainer, TrainingArguments
from datasets import load_dataset
import time

# 加载预训练模型和配置
model_name = "/home/qingyu_yin/mode/gpt-neo-1.3B"
config = GPTNeoConfig.from_pretrained(model_name, lora=True, lora_alpha=16, lora_r=8)
model = GPTNeoForCausalLM.from_pretrained(model_name, config=config)

# 加载数据集
dataset = load_dataset("/home/qingyu_yin/data/wikitext", "wikitext-2-raw-v1", split="train")

# 预处理数据集（根据需要进行修改）
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results" + str(time.time()),
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_dir="./logs" + str(time.time()),
    logging_steps=10,
)

# 创建 Trainer 并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
