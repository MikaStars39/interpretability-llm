import time
import torch
import argparse
from tqdm import tqdm

def load_model_and_tokenizer(args, device="cuda"):
    from transformers import AutoTokenizer
    if "gpt-neo-1.3B" in args.model_name_or_path:
        from src.modeling_gptneo import GPTNeoForCausalLM
        return GPTNeoForCausalLM.from_pretrained(args.model_name_or_path).to(device), AutoTokenizer.from_pretrained(args.tokenizer)
    if "llama-2-7b-hf" in args.model_name_or_path:
        from src.modeling_llama import LlamaForCausalLM
        return LlamaForCausalLM.from_pretrained(args.model_name_or_path).to(device), AutoTokenizer.from_pretrained(args.tokenizer)

# def test_across_rate(args, model):
#     baseline_a = []
#     baseline_m = []
#     ours_a = []
#     ours_m = []
#     for i in range(22, 1, -1):
#         model.transformer.skip_list.append(i)  # Set skip_from to the current value of i
#         print(model.transformer.skip_list)
#         print("Ours:")
#         ours_a.append(test_kv(args, model))
#         ours_m.append(test_mmlu(args, model))
#         print("####")
#         print("baseline:")
#         model.transformer.skip_from = None
#         baseline_a.append(test_kv(args, model))
#         baseline_m.append(test_mmlu(args, model))
#         model.transformer.skip_from = 24
#         print("####")
#     print(ours_a, ours_m, baseline_a, baseline_m)

def single_comparison(
    args,
    task,
    model,
    tokenizer,
    skip_list,
    stop=199,
):
    # print("########")
    # acc = task(args, model, tokenizer, stop)
    # print("baseline:", acc)
    # model.transformer.skip_list=skip_list
    # model.transformer.skip_from = None
    # acc = task(args, model, tokenizer, stop)
    # print("direct:", acc) 
    # model.transformer.skip_list=[27, 26, 25, 28, 24, 29, 23, 21, 22, 20, 19, 18]
    # model.transformer.skip_from = 1
    # acc = task(args, model, tokenizer, stop)
    # print("ffn_skip", acc)
    # model.transformer.skip_list=skip_list
    # model.transformer.skip_from = 2
    # acc = task(args, model, tokenizer, stop)
    # print("ours:", acc)
    print("########")
    acc = task(args, model, tokenizer, stop)
    print("baseline:", acc)
    model.model.skip_list=skip_list
    model.model.skip_from = None
    acc = task(args, model, tokenizer, stop)
    print("direct:", acc) 
    model.model.skip_list=[27, 26, 25, 28, 24, 29, 23, 21, 22,]
    model.model.skip_from = 1
    acc = task(args, model, tokenizer, stop)
    print("ffn_skip", acc)
    model.model.skip_list=skip_list
    model.model.skip_from = 2
    acc = task(args, model, tokenizer, stop)
    print("ours:", acc)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="piqa_mmlu_lambada_boolq_winogrande")
    parser.add_argument("--model_name_or_path", type=str, default="/home/qingyu_yin/model/gpt-neo-1.3B")
    parser.add_argument("--tokenizer", type=str, default="/home/qingyu_yin/model/gpt-neo-1.3B")
    parser.add_argument("--data_name_or_path", type=str, default="kv_test/kv_pairs_100_100.json")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--len", type=int, default=16) 
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    skip_list = [_ for _ in range(1, 31)]

    if "QA" in args.task_name:
        from src.functions import test_qa
        single_comparison(
            args,
            test_qa,
            model,
            tokenizer,
            skip_list,
        )  
    
    if "kv" in args.task_name:
        from src.functions import test_kv
        single_comparison(
            args,
            test_kv,
            model,
            tokenizer,
            skip_list,
        ) 

    if "piqa" in args.task_name:
        from src.functions import test_piqa
        single_comparison(
            args,
            test_piqa,
            model,
            tokenizer,
            skip_list,
        )

    
    if "mmlu" in args.task_name:
        from src.functions import test_mmlu
        single_comparison(
            args,
            test_mmlu,
            model,
            tokenizer,
            skip_list,
        ) 
    
    if "winogrande" in args.task_name:
        from src.functions import test_winogrande
        single_comparison(
            args,
            test_winogrande,
            model,
            tokenizer,
            skip_list,
        ) 
    
    if "boolq" in args.task_name:
        from src.functions import test_boolq
        single_comparison(
            args,
            test_boolq,
            model,
            tokenizer,
            skip_list,
        ) 
    
    if "lambada" in args.task_name:
        from src.functions import test_lambada
        single_comparison(
            args,
            test_lambada,
            model,
            tokenizer,
            skip_list,
        ) 
    

    
