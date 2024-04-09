from torch.utils.data import Dataset, DataLoader

class MMLUDataset(Dataset):
    def __init__(
        self, 
        model_type="/home/qingyu_yin/model/gpt-neo-1.3B", 
        data_type="validation",
        device="cuda",
        length=1024,
    ):
        super.__init__()
        self.dataset = load_dataset("/home/qingyu_yin/data/pg19-test", "all", split=data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device
        self.target_question = "Revue Starlight the Movie (Gekijōban Shōjo☆Kageki Revyū Sutāraito) is a sequel film continuing from the ending shared by the anime and Rondo Rondo Rondo. "

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

def generate_qa_dataset():

