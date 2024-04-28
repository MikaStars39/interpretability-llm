import torch
import argparse
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generate_data import generate_sum_expressions

def get_precision(
    precision: str,
):
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    elif precision == "fp32":
        return torch.float32
    else:
        raise ValueError("Not a valid data type")

def build_expression():
    zero, __ = generate_sum_expressions(
        min_terms=1,
        max_terms=3,
        num_range=10,
        need_total_sum=0,
    )

    none_zero, result_none = generate_sum_expressions(
        min_terms=1,
        max_terms=3,
        num_range=10,
        need_total_sum=None,
    )
    
    expression_cpl, result_cpl = generate_sum_expressions(
        min_terms=4,
        max_terms=4,
        num_range=20,
        need_total_sum=None,
    )

    expression_simple, result_simple = generate_sum_expressions(
        min_terms=1,
        max_terms=2,
        num_range=10,
        need_total_sum=None,
        minus=False,
    )
    expression = expression_simple + "(" + zero + ")*(" + expression_cpl + ")+" + "="
    none_expression = expression_simple + "(" + none_zero + ")*(" + expression_cpl + ")+" + "="

    return expression, result_simple, none_expression, result_cpl*result_none+result_simple

def build_code(inputs, range: int = 10):
    input_value = random.randint(0, range)

    if inputs["type"] == "+":
        result = input_value + int(inputs["number"])
    elif inputs["type"] == "-":
        result = input_value - int(inputs["number"])
    elif inputs["type"] == "*":
        result = input_value * int(inputs["number"])
    elif inputs["type"] == "/":
        result = input_value / int(inputs["number"])
    else:
        print(inputs["type"])
        raise ValueError("Not a valid type")
    
    return inputs["code"], input_value, result

@torch.no_grad()
def test_expression(
    model, tokenizer, precision,
    test_len: int = 99,
    shot_num: int = 5,
    generation_len: int = 2,
):

    prompt = "Now you need to calculate the answer of some mathematic equations. Here are some examples: \n"
    instruction = ""
    answer = ""

    count_has_zero = 0
    count_no_zero = 0

    for _ in tqdm(range(test_len)):
        text_has_zero = prompt
        text_no_zero = prompt
        for __ in range(shot_num):
            expression_has_zero, result_has_zero, expression_no_zero, result_no_zero = build_expression()
            text_has_zero += expression_has_zero + str(result_has_zero) + " \n "
            text_no_zero += expression_no_zero + str(result_no_zero) + " \n "
        text_has_zero += instruction
        text_no_zero += instruction
        expression_has_zero, result_has_zero, expression_no_zero, result_no_zero = build_expression()
        text_has_zero += expression_has_zero
        text_no_zero += expression_has_zero

        inputs = tokenizer(text_has_zero, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens = generation_len, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_has_zero):]

        # print(text_has_zero, outputs)

        if str(result_has_zero) in outputs:
            count_has_zero += 1
        
        inputs = tokenizer(text_no_zero, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens = generation_len, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_no_zero):]

        # print(text_no_zero, outputs)

        if str(result_has_zero) in outputs:
            count_no_zero += 1

    print("shot num:", shot_num)
    print(count_has_zero/test_len)
    print(count_no_zero/test_len)

@torch.no_grad()
def test_code(
    model, tokenizer, precision,
    test_len: int = 99,
    shot_num: int = 5,
    generation_len: int = 2,
    data_name_or_path: str = "../data/code.json"
):
    prompt = "Now you need to give me the printed result after running this python code . Here are some examples: \n"
    hint = "The code is: \n"
    instruction = "The input is: "
    answer = ", so the output is: "

    count_has_zero = 0
    # count_no_zero = 0

    dataset = load_dataset("json", data_files=data_name_or_path, split="train")

    for pos in tqdm(range(test_len)):
        text_has_zero = prompt
        real_result = None
        # text_no_zero = prompt
        for idx in range(shot_num):
            input_code, input_value, result = build_code(dataset[pos+idx])
            text_has_zero += hint + input_code + instruction + str(input_value) + answer + str(result) + "\n" + "Here is next example:" + "\n"
            if idx == shot_num-1:
                text_has_zero += instruction
                # text_no_zero += instruction
                input_code, input_value, real_result = build_code(dataset[pos+idx+1])
                text_has_zero += hint + input_code + instruction + str(input_value) + answer

        # print(text_has_zero)

        inputs = tokenizer(text_has_zero, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens = generation_len, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_has_zero):]

        # print(text_has_zero, outputs)

        if str(real_result) in outputs:
            count_has_zero += 1
        
        # inputs = tokenizer(text_no_zero, return_tensors="pt").to("cuda")
        # outputs = model.generate(
        #     inputs["input_ids"], 
        #     max_new_tokens = generation_len, 
        #     num_return_sequences=1, 
        #     pad_token_id=tokenizer.eos_token_id
        #     )
        
        # outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_no_zero):]

        # # print(text_no_zero, outputs)

        # if str(result_has_zero) in outputs:
        #     count_no_zero += 1

    print("shot num:", shot_num)
    print(count_has_zero/test_len)
    # print(count_no_zero/test_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/qingyu_yin/model/llama-2-7b-hf")
    parser.add_argument("--tokenizer", type=str, default="/home/qingyu_yin/model/llama-2-7b-hf")
    parser.add_argument("--data_name_or_path", type=str, default="code")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--len", type=int, default=16) 
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    precision = get_precision(args.precision)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=precision).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.data_name_or_path == "expression":
        test = test_expression
    elif args.data_name_or_path == "code":
        test = test_code
    else:
        raise ValueError("Not a valid data type")

    for shot_num in [1, 2, 4, 8, 16, 32]:
        test(
            model=model,
            tokenizer=tokenizer,
            precision=precision,
            shot_num=shot_num,
        )