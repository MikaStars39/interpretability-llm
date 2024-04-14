import torch
import plotly.express as px
from transformer_lens import utils, HookedTransformer, FactoredMatrix
from functools import partial

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def patching():
    clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
    corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer=" Mary"):
        # model.to_single_token maps a string value of a single token to the token index for that token
        # If the string is not a single token, it raises an error.
        correct_index = model.to_single_token(correct_answer)
        incorrect_index = model.to_single_token(incorrect_answer)
        return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

    # We run on the clean prompt with the cache so we store activations to patch in later.
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_logit_diff = logits_to_logit_diff(clean_logits)
    print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

    # We don't need to cache on the corrupted prompt.
    corrupted_logits = model(corrupted_tokens)
    corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
    print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")
    # We define a residual stream patching hook
    # We choose to act on the residual stream at the start of the layer, so we call it resid_pre
    # The type annotations are a guide to the reader and are not necessary
    def residual_stream_patching_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        position: int
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Each HookPoint has a name attribute giving the name of the hook.
        clean_resid_pre = clean_cache[hook.name]
        resid_pre[:, position, :] = clean_resid_pre[:, position, :]
        return resid_pre

    # We make a tensor to store the results for each patching run. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    num_positions = len(clean_tokens[0])
    ioi_patching_result = torch.zeros((model.cfg.n_layers, num_positions), device=model.cfg.device)

    for layer in tqdm.tqdm(range(model.cfg.n_layers)):
        for position in range(num_positions):
            # Use functools.partial to create a temporary hook function with the position fixed
            temp_hook_fn = partial(residual_stream_patching_hook, position=position)
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
                (utils.get_act_name("resid_pre", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
            # Store the result, normalizing by the clean and corrupted logit difference so it's between 0 and 1 (ish)
            ioi_patching_result[layer, position] = (patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff)

@torch.no_grad()
def circuits():
    device = utils.get_device()
    model = HookedTransformer.from_pretrained("qwen-7b-chat", device=device)

    model_description_text = """## Loading Models HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.  For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""
    loss = model(model_description_text, return_type="loss")
    print("Model loss:", loss)

if __name__ == "__main__":
    circuits()