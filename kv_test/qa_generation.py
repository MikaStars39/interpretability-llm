import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class QADataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="test",
        device="cuda",
        length=1024,
    ):
        self.dataset = load_dataset("/home/qingyu_yin/data/lambada", split=data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device
        self.length = length
        # self.target_question = "Revue Starlight the Movie (Gekijōban Shōjo☆Kageki Revyū Sutāraito) is a sequel film. "
        # self.target_answer = "The film includes 50 minutes of musical sequences featuring six new revue songs."

    def __getitem__(self, index):
        text = self.dataset[index]["text"][:self.length * 20]
        example = "USER: I like apple. Repeat all content I have just said. AGENT: I like apple. USER: The film includes 50 minutes of musical sequences featuring six new revue songs. Repeat all content I have just said. AGENT: The film includes 50 minutes of musical sequences featuring six new revue songs. USER: "
        text_ids = self.tokenizer(text, truncation=True, max_length=self.length)["input_ids"]
        text_only = torch.tensor(text_ids, device=self.device)
        text_ids = self.tokenizer(example)["input_ids"] + text_ids
        instruction = " Repeat all content I have just said. AGENT: "
        instruction = self.tokenizer(instruction)["input_ids"]
        text_ids += instruction
        text_ids = torch.tensor(text_ids, device=self.device)
        return (text_ids, text_only)

    def __len__(self):
        return len(self.dataset)

def load_qa(
    batch_size: int = 1,
    model_type: str = "/home/qingyu_yin/model/gpt-neo-1.3B",
    data_type: str = "test",
    device: str = "cuda",
    length: int = 1024, 
):
    dataset = QADataset(
        model_type=model_type,
        data_type=data_type,
        device=device,
        length=length,
    )
    return DataLoader(dataset, batch_size=batch_size)

