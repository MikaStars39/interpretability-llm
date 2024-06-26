import torch
from tqdm import tqdm

def compare_lists(list1, list2):
    count = 0
    for element in list1:
        if element in list2:
            count += 1
            list2.remove(element)  # 从list2中移除匹配到的元素，以防止重复计数
    return count

@torch.no_grad()
def test_qa(
    args,
    model,
    tokenizer,
    stop,
):
    from kv_test.qa_generation import load_qa

    dataset = load_qa(
        batch_size = 1,
        model_type = args.tokenizer,
        length= 1024, 
    )
    acc = 0

    for ids, batch in tqdm(enumerate(dataset)):
        if ids >= stop:
            break
        inputs, label = batch
        outputs = model.generate(
            inputs, 
            max_length=inputs.size(-1) + label.size(-1), 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )

        equal_elements = compare_lists(list(outputs[0, -label.size(-1):]), list(label[0]))
        acc += (equal_elements / label.size(-1))
    acc = acc / stop
    return acc

@torch.no_grad()
def test_kv(
    args,
    model,
    tokenizer,
    stop,
):
    from kv_test.kv_generation import generate_kv

    dataset = generate_kv(length=16)
    acc = 0

    for ids, batch in tqdm(enumerate(dataset)):
        label, query = batch
        label = tokenizer(label, return_tensors="pt")["input_ids"].to("cuda")
        inputs = tokenizer(query, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=inputs["input_ids"].size(1) + 10, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )

        # print(outputs[len(query):len(query)+8])
        equal_elements = compare_lists(list(outputs[0, -label.size(-1):]), list(label[0]))
        acc += (equal_elements / label.size(-1))
    acc = acc / len(dataset)
    return acc


@torch.no_grad()
def test_ppl(args, model):
    from src.dataset import load_pg19
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 

    dataloader = load_pg19(batch_size=args.batch_size)

    all_loss = []

    for ids, batch in tqdm(enumerate(dataloader)):
        inputs, labels = batch
        outputs = model(inputs, labels=inputs)
        all_loss.append(float(torch.exp(outputs.loss)))

    return torch.tensor(all_loss).mean()


@torch.no_grad()
def test_mmlu(
    args, 
    model,
    tokenizer,
    stop,
):
    from src.dataset import load_mmlu

    dataloader = load_mmlu(model_type=args.tokenizer)
    acc = 0
    for ids, batch in tqdm(enumerate(dataloader)):
        query, answer = batch
        outputs = model.generate(
            query, 
            max_length=query.size(1) + 1, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        # print(answer, outputs[0, -1])
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if answer == outputs[0, -1]:
            acc = acc + 1
    acc = acc / 285
    return acc

@torch.no_grad()
def test_piqa(
    args, 
    model, 
    tokenizer,
    stop = 199,
):
    from src.dataset import load_piqa

    dataloader = load_piqa(model_type=args.tokenizer)
    acc = 0
    for ids, batch in tqdm(enumerate(dataloader)):
        if ids >= stop:
            break
        query, answer = batch
        outputs = model.generate(
            query, 
            max_length=query.size(1) + 1, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if answer == outputs[0, -1]:
            acc = acc + 1
    acc = acc / stop
    return acc

@torch.no_grad()
def test_lambada(
    args, 
    model, 
    tokenizer,
    stop = 199,
):

    from src.dataset import load_lambada

    dataloader = load_lambada(model_type=args.tokenizer)
    acc = 0
    for ids, batch in tqdm(enumerate(dataloader)):
        if ids >= stop:
            break
        query, answer = batch
        outputs = model.generate(
            query, 
            max_length=query.size(1) + 1, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        # print(answer, outputs[0, -1])
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if answer == outputs[0, -1]:
            acc = acc + 1
    acc = acc / stop
    return acc

@torch.no_grad()
def test_winogrande(
    args, 
    model, 
    tokenizer,
    stop = 199,
):
    from src.dataset import load_winogrande

    dataloader = load_winogrande(model_type=args.tokenizer)
    acc = 0
    for ids, batch in tqdm(enumerate(dataloader)):
        if ids >= stop:
            break
        query, answer = batch
        outputs = model.generate(
            query, 
            max_length=query.size(1) + 1, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        # print(answer, outputs[0, -1])
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if answer == outputs[0, -1]:
            acc = acc + 1
    acc = acc / stop
    return acc

@torch.no_grad()
def test_boolq(
    args, 
    model, 
    tokenizer,
    stop = 199,
):
    from src.dataset import load_boolq

    dataloader = load_boolq(model_type=args.tokenizer)
    acc = 0
    for ids, batch in tqdm(enumerate(dataloader)):
        if ids >= stop:
            break
        query, answer = batch
        outputs = model.generate(
            query, 
            max_length=query.size(1) + 1, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        # print(answer, outputs[0, -1])
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if answer == outputs[0, -1]:
            acc = acc + 1
    acc = acc / stop
    return acc


    
