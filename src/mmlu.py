from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer

class MMLUDataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="dev"
    ):
        self.dataset = load_dataset("/home/qingyu_yin/data/mmlu", "all", split=data_type)
        self.tokenizer = AutoTokenizer(model_type)

    def __getitem__(self, index):
        question = self.dataset[index]["question"]
        choices = self.dataset[index]["choices"]
        choices = ", ".join([f"{chr(65+i)}.{choice}" for i, choice in enumerate(choices)])
        example = "The question is: get the answer of 1+2. The choices are A.0, B.1, C.2 D.3, So we will choose D. "
        prompt = "Please answer this math question." + example + "The question is: " + question + " The choices are: " + choices + " So we will choose "
        return prompt

    def __len__(self):
        return len(self.dataset)

def load_mmlu():
    dataset = MMLUDataset()
    dataloader 

load_mmlu()