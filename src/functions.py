import torch
from tqdm import tqdm

@torch.no_grad()
def test_kv(
    args,
    model,
    tokenizer,
    stop,
):
    from kv_test.kv_generation import generate_kv

    dataset = generate_kv(length=99)
    acc = 0
    perplexity = 0

    for ids, batch in tqdm(enumerate(dataset)):
        answer, query = batch
        inputs = tokenizer(query, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=inputs["input_ids"].size(1) + 10, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )

        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(outputs[len(query):len(query)+8])
        if answer == outputs[len(query):len(query)+8]:
            acc = acc + 1
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
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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


    
