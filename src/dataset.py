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
        length=1024,
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
        length=1024,
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


class PIQADataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="validation",
        device="cuda",
        length=1024
    ):
        self.dataset = load_dataset("/home/qingyu_yin/data/piqa", split=data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device
        self.length = length

    def __getitem__(self, index):
        goal = self.dataset[index]["goal"]
        sol1 = self.dataset[index]["sol1"]
        sol2 = self.dataset[index]["sol2"]
        label = self.dataset[index]["label"]
        text = "You should answer a question with two options. Example: How do you eat? 0. use mouth. 1. use leg. We should choose the option 0. Question: "
        text = text + goal + " 0. " + sol1 + " 1. " + sol2 + " We should choose the option " + str(label)
        text_ids = torch.tensor(self.tokenizer(text, truncation=True, max_length=self.length)["input_ids"], device=self.device)
        return (text_ids[:-1].contiguous(), text_ids[-1:].contiguous())

    def __len__(self):
        return len(self.dataset)

class LambadaDataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="test",
        device="cuda",
        length=1024
    ):
        self.dataset = load_dataset("/home/qingyu_yin/data/lambada", split=data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device
        self.length = length

    def __getitem__(self, index):
        text = self.dataset[index]["text"]
        text_ids = torch.tensor(self.tokenizer(text, truncation=True, max_length=self.length)["input_ids"], device=self.device)
        label = text_ids[-1:]
        text_ids = text_ids[:-1]
        return (text_ids, label)

    def __len__(self):
        return len(self.dataset)

class WinograndeDataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="validation",
        device="cuda",
        length=1024
    ):
        self.dataset = load_dataset("/home/qingyu_yin/data/winogrande", "winogrande_debiased", split=data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device
        self.length = length

    def __getitem__(self, index):
        goal = self.dataset[index]["sentence"]
        sol1 = self.dataset[index]["option1"]
        sol2 = self.dataset[index]["option2"]
        label = self.dataset[index]["answer"]
        text = "You should answer a question with two options. Example: Alice was a better teacher than Bob, so _ was a better teacher. 1. Alice 2. Bob We should choose the option 1. Question: "
        text = text + goal + " 1. " + sol1 + " 2. " + sol2 + " We should choose the option " + str(label)
        text_ids = torch.tensor(self.tokenizer(text, truncation=True, max_length=self.length)["input_ids"], device=self.device)
        return (text_ids[:-1].contiguous(), text_ids[-1:].contiguous())

    def __len__(self):
        return len(self.dataset)

class BoolQDataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="validation",
        device="cuda",
        length=1024
    ):
        self.dataset = load_dataset("/home/qingyu_yin/data/winogrande", split=data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device
        self.length = length

    def __getitem__(self, index):
        passage = self.dataset[index]["passage"]
        question = self.dataset[index]["question"]
        label = self.dataset[index]["answer"]
        text = "You should answer a true-or-false question based on the given passage. EXAMPLE: Passage: Alice was a better teacher than Bob, so Bob was a better teacher. Question: Is it correct? Answer: We think it is false. Passage: "
        text = text + passage + " Question: " + question + " We think it is " + str(label)
        text_ids = torch.tensor(self.tokenizer(text, truncation=True, max_length=self.length)["input_ids"], device=self.device)
        return (text_ids[:-1].contiguous(), text_ids[-1:].contiguous())

    def __len__(self):
        return len(self.dataset)
    
# load functions

def load_pg19(
    batch_size: int = 1,
    model_type: str = "/home/qingyu_yin/model/gpt-neo-1.3B",
    data_type: str = "test",
    device: str = "cuda", 
    length: int = 1024,
):
    dataset = Pg19Dataset(
        model_type=model_type,
        data_type=data_type,
        device=device, 
        length=length,
    )
    return DataLoader(dataset, batch_size=batch_size)

def load_mmlu(
    batch_size: int = 1,
    model_type: str = "/home/qingyu_yin/model/gpt-neo-1.3B",
    data_type: str = "dev",
    device: str = "cuda",
    length: int = 1024,
):
    dataset = MMLUDataset(
        model_type=model_type,
        data_type=data_type,
        device=device,
        length=length
        )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def load_wikitext(
    data_type = "test",
):  
    return load_dataset("/home/qingyu_yin/data/wikitext", 'wikitext-103-raw-v1', split=data_type)

def load_piqa(
    batch_size: int = 1,
    model_type: str = "/home/qingyu_yin/model/gpt-neo-1.3B",
    data_type: str = "validation",
    device: str = "cuda",
    length: int = 1024, 
):
    dataset = PIQADataset(
        model_type=model_type,
        data_type=data_type,
        device=device,
        length=length,
    )
    return DataLoader(dataset, batch_size=batch_size)

def load_lambada(
    batch_size: int = 1,
    model_type: str = "/home/qingyu_yin/model/gpt-neo-1.3B",
    data_type: str = "test",
    device: str = "cuda",
    length: int = 1024, 
):
    dataset = LambadaDataset(
        model_type=model_type,
        data_type=data_type,
        device=device,
        length=length,
    )
    return DataLoader(dataset, batch_size=batch_size)

def load_winogrande(
    batch_size: int = 1,
    model_type: str = "/home/qingyu_yin/model/gpt-neo-1.3B",
    data_type: str = "validation",
    device: str = "cuda",
    length: int = 1024, 
):
    dataset = WinograndeDataset(
        model_type=model_type,
        data_type=data_type,
        device=device,
        length=length,
    )
    return DataLoader(dataset, batch_size=batch_size)

def load_boolq(
    batch_size: int = 1,
    model_type: str = "/home/qingyu_yin/model/gpt-neo-1.3B",
    data_type: str = "validation",
    device: str = "cuda",
    length: int = 1024, 
):
    dataset = WinograndeDataset(
        model_type=model_type,
        data_type=data_type,
        device=device,
        length=length,
    )
    return DataLoader(dataset, batch_size=batch_size)

