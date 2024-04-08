import torch
import matplotlib.pyplot as plt
from einops import rearrange

@torch.no_grad()
def modified_rv_coefficient(X_i, X_j):

    S_i = torch.matmul(X_i, X_i.transpose(1, 0))
    S_j = torch.matmul(X_j, X_j.transpose(1, 0))

    # Subtract the diagonal elements from each covariance matrix
    S_i_mod = S_i - torch.diag(torch.diag(S_i))
    S_j_mod = S_j - torch.diag(torch.diag(S_j))

    # Compute the trace of the product of the modified covariance matrices
    numerator = torch.trace(torch.mm(S_i_mod, S_j_mod))

    # Compute the denominator
    denominator = torch.sqrt(torch.trace(torch.mm(S_i_mod, S_i_mod)) * torch.trace(torch.mm(S_j_mod, S_j_mod)))

    # Compute the modified RV coefficient
    rv_mod = numerator / denominator

    return rv_mod


@torch.no_grad()
def draw_rv_figure(
    hidden_state: torch.Tensor,
    heads_num: int,
    heads_dim: int,
    layers: int,
    fig_path: str,
    type_a: int,
    type_b: int,
    ):
    a = torch.tensor([])
    for i in range(0, layers):
        a_layer = hidden_state[i][type_a][0]
        b_layer = hidden_state[i][type_b][0] + hidden_state[i][type_a][0]
        b = []
        for j in range(1, heads_num+1):
            result = - torch.log(1 - modified_rv_coefficient(a_layer[:, (j-1) * heads_dim:j * heads_dim], b_layer[:, (j-1) * heads_dim:j * heads_dim]))
            b.append(result.item())
        b = torch.tensor(b)
        b = b.unsqueeze(0)
        a = torch.concat([a, b], dim=0)
    
    plt.imshow(a.detach().numpy(), cmap='Blues')
    plt.title(f'Attention')
    plt.savefig(fig_path)
    print(f"Successfully draw RV figure, saved in {fig_path}")


def draw_attention_maps(
    attentions: tuple,
    fig_path: str,
):
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = attentions[0].shape[-1]
    
    num_subplots_per_layer = num_heads

    # 创建大图
    fig, axs = plt.subplots(num_layers, num_subplots_per_layer, figsize=(150, 100))

    for layer_idx, layer_attention in enumerate(attentions):
        for head_idx in range(num_heads):
            # 选择子图并绘制注意力权重
            ax = axs[layer_idx, head_idx]
            im = ax.imshow(layer_attention[0, head_idx].to("cpu"), cmap='Blues')

            # 隐藏坐标轴
            ax.axis('off')

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.02, hspace=0.04)
    plt.savefig(fig_path)
    print(f"Successfully draw RV figure, saved in {fig_path}")


def patching_alternation(
    hidden_state_clean: torch.Tensor,
    hidden_state_corrupt: torch.Tensor,
    patch_heads: list,
    head_dim: int,
    position = None,
):
    for head in patch_heads:
        if position is None:
            hidden_state_clean[:, :, head * head_dim:(head + 1) * head_dim] = hidden_state_corrupt[:, :, head * head_dim:(head + 1) * head_dim]
        elif type(position) == list:
            for each in position:
                hidden_state_clean[:, each[0]:each[1], head * head_dim:(head + 1) * head_dim] = hidden_state_corrupt[:, each[0]:each[1], head * head_dim:(head + 1) * head_dim]
        else:
            hidden_state_clean[:, position, head * head_dim:(head + 1) * head_dim] = hidden_state_corrupt[:, position, head * head_dim:(head + 1) * head_dim]
    
    return hidden_state_clean


def param_free_attention(inputs, num_heads):
    batch_size, sequence_length, hidden_size = inputs.size()
    
    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
    head_dim = hidden_size // num_heads
    
    inputs = rearrange(inputs, 'b s (h d) -> b h s d', h=num_heads)
    
    Q = inputs
    K = inputs
    V = inputs
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    attention = torch.softmax(scores, dim=-1)
    
    output = torch.matmul(attention, V)
    
    output = rearrange(output, 'b h s d -> b s (h d)')
    
    return output