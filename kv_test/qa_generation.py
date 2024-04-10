import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

def insert_after_first_period(original_string, string_to_insert):
    first_period_index = original_string.find('.')
    
    if first_period_index != -1:
        inserted_string = original_string[:first_period_index] + string_to_insert + original_string[first_period_index:]
    else:
        inserted_string = original_string
    
    return inserted_string

class QADataset(Dataset):
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
        self.target_question = "Revue Starlight the Movie (Gekijōban Shōjo☆Kageki Revyū Sutāraito) is a sequel film. "
        self.target_answer = "The film includes 50 minutes of musical sequences featuring six new revue songs."

    def __getitem__(self, index):
        text = self.dataset[index]["text"][:self.length // 5]
        text = text + (self.target_question + self.target_answer)
        text_ids = self.tokenizer(text, truncation=True, max_length=self.length - 30)["input_ids"]
        question = "Repeat the sentence after this sentence: " + self.target_question + "This sentence is \""
        question = self.tokenizer(question, truncation=True, max_length=self.length - 30)["input_ids"]
        answer = self.tokenizer(self.target_answer, truncation=True, max_length=self.length - 30)["input_ids"]
        text_ids = torch.tensor(text_ids + question, device=self.device)
        answer = torch.tensor(answer, device=self.device)
        return (text_ids, answer)

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

