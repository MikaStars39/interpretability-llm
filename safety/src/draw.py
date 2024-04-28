import tqdm
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

