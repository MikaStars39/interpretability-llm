import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class MMLUDataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="dev",
        device="cuda",
    ):
        self.dataset = load_dataset("/home/qingyu_yin/data/mmlu", "all", split=data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device

    def __getitem__(self, index):
        question = self.dataset[index]["question"]
        choices = self.dataset[index]["choices"]
        choices = ", ".join([f"{chr(65+i)}.{choice}" for i, choice in enumerate(choices)])
        example = "Exmaple: get the answer of 1+2. The choices are A.0, B.1, C.2 D.3, So we will choose D. "
        prompt = "Please answer this math question." + example + "The question is: " + question + " The choices are: " + choices + ". So we will choose " + chr(65+self.dataset[index]["answer"])
        prompt = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(self.device)
        answer = prompt[-1].contiguous()
        prompt = prompt[:-1].contiguous()
        return (prompt, answer)

    def __len__(self):
        return len(self.dataset)

class Pg19Dataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="test",
        device="cuda",
        length=1024
    ):
        self.dataset = load_dataset("/home/qingyu_yin/data/pg19-test", "all", split=data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device
        self.length = length

    def __getitem__(self, index):
        text = self.dataset[index]["text"]
        text_ids = self.tokenizer(text, truncation=True, max_length=self.length)["input_ids"]
        input_ids = torch.tensor(text_ids[:len(text_ids)-1]).to(self.device)
        labels = torch.tensor(text_ids[1:len(text_ids)]).to(self.device)
        return (input_ids, labels)

    def __len__(self):
        return len(self.dataset)


def load_pg19(
    batch_size: int = 1,
):
    dataset = Pg19Dataset()
    return DataLoader(dataset, batch_size=batch_size)

def load_mmlu(
    batch_size: int = 1,
    model_type: str = "/home/qingyu_yin/model/gpt-neo-1.3B"
):
    dataset = MMLUDataset(model_type=model_type)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def load_wikitext(
    data_type = "test",
):  
    return load_dataset("/home/qingyu_yin/data/wikitext", 'wikitext-103-raw-v1', split=data_type)
    