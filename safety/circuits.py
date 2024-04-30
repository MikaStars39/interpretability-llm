import torch
import plotly.express as px
import tqdm
import einops
import transformer_lens.patching as pt
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.hook_points import (
    HookPoint,
) 
from functools import partial
from jaxtyping import Float
from neel_plotly.plot import imshow

from src.draw import draw_attention_heads, save_imshow

def early_stop(
    cache: ActivationCache,
    model: HookedTransformer,
    layer_id: int,
    icl_ans: str,
    real_ans: str,
):
    accum_resid, labels = cache.accumulated_resid(return_labels=True, apply_ln=True)
    last_token_accum = accum_resid[:, 0, -1, :]  # layer, batch, pos, d_model
    W_U = model.W_U
    layers_unembedded = torch.matmul(last_token_accum, W_U)
    print(layers_unembedded.shape)
    sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
    rank_answer = (sorted_indices == model.to_single_token(icl_ans)).nonzero(as_tuple=True)[1]
    print(pd.Series(rank_answer.cpu(), index=labels))
    rank_answer = (sorted_indices == model.to_single_token(real_ans)).nonzero(as_tuple=True)[1]
    print(pd.Series(rank_answer.cpu(), index=labels))


def get_act_block_every_pos(
    model: HookedTransformer,
    corrupted_tokens,
    clean_cache,
    ioi_metric,
):
    every_block_result = pt.get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric)
    figure = imshow(every_block_result, 
    facet_col=0, 
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"], 
    title="Activation Patching Per Block", 
    xaxis="Position", yaxis="Layer", 
    zmax=1, zmin=-1, 
    x= [f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(corrupted_tokens[0]))],
    return_fig=True,
    )
    save_imshow(figure)

def get_all_head(
    model: HookedTransformer,
    corrupted_tokens,
    clean_cache,
    ioi_metric,
):
    every_head_all_pos_act_patch_result = pt.get_act_patch_attn_head_all_pos_every(model, corrupted_tokens, clean_cache, ioi_metric)
    figure = imshow(
        every_head_all_pos_act_patch_result, 
        facet_col=0, facet_labels=["Output", "Query", "Key", "Value", "Pattern"], 
        title="Activation Patching Per Head (All Pos)", 
        xaxis="Head", yaxis="Layer", zmax=1, zmin=-1,
        return_fig=True,
    )
    save_imshow(figure)

def get_single_head(
    model: HookedTransformer,
    corrupted_tokens,
    clean_cache,
    ioi_metric,
):

    attn_head_out_all_pos_act_patch_results = pt.get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, ioi_metric)
    figure = imshow(attn_head_out_all_pos_act_patch_results, 
        yaxis="Layer", 
        xaxis="Head", 
        title="attn_head_out Activation Patching (All Pos)",
        return_fig=True,
        )
    save_imshow(figure)

def patching(model: HookedTransformer):

    clean_prompt = "Calculate this: 0*(1+2-3-4-5)+0+2=2, "
    corrupted_prompt = "Calculate this: 0*(1+2-3-4-5)+0+2=2"
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    def logits_to_logit_diff(logits, correct_answer="1", incorrect_answer="2"):
        # model.to_single_token maps a string value of a single token to the token index for that token
        # If the string is not a single token, it raises an error.
        correct_index = model.to_single_token(correct_answer)
        incorrect_index = model.to_single_token(incorrect_answer)
        return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_logit_diff = logits_to_logit_diff(clean_logits)
    print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

    # draw_attention_heads(model, clean_cache, [f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))])

    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
    corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
    print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")
    
    CLEAN_BASELINE = clean_logit_diff
    CORRUPTED_BASELINE = corrupted_logit_diff
    def ioi_metric(logits):
        return (logits_to_logit_diff(logits) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)
    
    get_single_head(model, corrupted_tokens, clean_cache, ioi_metric)

def run(model: HookedTransformer):
    clean_prompt = "Calculate this: 1+1=2, (1-1)*(1+2+3)+1+2="
    clean_tokens = model.to_tokens(clean_prompt)[:, 1:]
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    text = model.generate(clean_tokens)
    early_stop(clean_cache, model, 0, "2", "3")
    clean_prompt = "Calculate this: (1-1)*(1+2+3)+1+2=3, (4-3)*(1+2-1)+2+0="
    clean_tokens = model.to_tokens(clean_prompt)[:, 1:]
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    text = model.generate(clean_tokens)
    early_stop(clean_cache, model, 0, "2", "4")
    clean_prompt = "Calculate this: (1-1)*(1+2+3)+1+2=3, (4-3)*(1+2-1)+2+0=4, (27-26+1)*(2342-6+324-3)+1+1=2, (34-65+31)*(43-2-41+55-54)+5-2="
    clean_tokens = model.to_tokens(clean_prompt)[:, 1:]
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    text = model.generate(clean_tokens)
    early_stop(clean_cache, model, 0, "3", "5")
    
@torch.no_grad()
def circuits():
    device = utils.get_device()
    model = AutoModelForCausalLM.from_pretrained("/home/qingyu_yin/model/Qwen1.5-1.8B", torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("/home/qingyu_yin/model/Qwen1.5-1.8B")
    model = HookedTransformer.from_pretrained_no_processing(model_name="Qwen1.5-1.8B", hf_model=model, torch_dtype=torch.float16)
    patching(model)

if __name__ == "__main__":
    circuits()