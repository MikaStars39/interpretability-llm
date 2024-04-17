import torch
import plotly.express as px
import tqdm
import einops
import transformer_lens.patching as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import utils, HookedTransformer, FactoredMatrix
from transformer_lens.hook_points import (
    HookPoint,
) 
from functools import partial
from jaxtyping import Float
from datetime import datetime
from neel_plotly.plot import imshow

def save_imshow(fig, filename=None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename == None:
        filename = f"image/{current_time}_image.png"
    fig.write_image(filename)

def draw_attention_heads(model, cache, tokens):
    for layer in tqdm.tqdm(range(model.cfg.n_layers)):
        attention_pattern = cache["pattern", layer, "attn"][0]
        for head in range(model.cfg.n_heads):
            head_attn = attention_pattern[head]
            fig = imshow(
                head_attn, 
                xaxis="Position", yaxis="Position", 
                x=tokens,
                title="Layer " + str(layer) + " Head " + str(head),
                return_fig=True,
            )
            save_imshow(fig, "image/head/" + "Layer" + str(layer) + "Head" + str(head) + ".png")

    
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

def patching(model: HookedTransformer):

    clean_prompt = "0*(1+2-3-4-5)+1+2=3, (2-1+3)*(3+4-2-4-5-2)+1+0=1, (3+2-5)*(3-1-4+2+5+2-4)+1+0="
    corrupted_prompt = "0*(2+1-4-3-3)+2+0=2, (1-2+1)*(2+3-3-8-4-1)+0+2=2, (4+3-7)*(1-2-5+9+1+3-2)+2+0="
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
    
    get_act_block_every_pos(model, corrupted_tokens, clean_cache, ioi_metric)
    
@torch.no_grad()
def circuits():
    device = utils.get_device()
    model = AutoModelForCausalLM.from_pretrained("/home/qingyu_yin/model/Qwen1.5-1.8B", torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("/home/qingyu_yin/model/Qwen1.5-1.8B")
    model = HookedTransformer.from_pretrained_no_processing(model_name="Qwen1.5-1.8B", hf_model=model ,dtype=torch.float16, )
    patching(model)

if __name__ == "__main__":
    circuits()