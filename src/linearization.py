import torch
import math

def generate_mask(
    mask: torch.Tensor,
    seq_length: int,
):
    attention_mask = torch.zeros(seq_length, seq_length).to(mask.device)
    assert len(attention_mask.shape) == 2
    ones = torch.ones_like(attention_mask).to(mask.device)
    window_length = math.ceil(math.sqrt(seq_length))
    new_attention_mask = 1 - torch.triu(ones, -window_length+1)
    local_mask = torch.where(new_attention_mask == 0, attention_mask, -1e5)
    new_attention_mask = new_attention_mask +  torch.triu(ones, 1)
    local_mask = torch.where(new_attention_mask == 0, attention_mask, -1e5) 
    local_mask = torch.cat([torch.zeros(attention_mask.size(-1), 1).to(mask.device), local_mask[:, 1:]], dim=-1)
    
    new_attention_mask = torch.zeros_like(attention_mask).to(mask.device)
    for idx in range(0, seq_length // window_length):
        new_attention_mask += torch.triu(ones, -window_length*idx) - torch.triu(ones, -window_length*idx+1)
    cross_mask = torch.where(new_attention_mask == 1, attention_mask, -1e5)
    cross_mask = torch.cat([torch.zeros(attention_mask.size(-1), 1).to(mask.device), cross_mask[:, 1:]], dim=-1)
    # print(local_mask, cross_mask)
    return local_mask.unsqueeze(0)[:, -mask.size(-2):, -mask.size(-1):], cross_mask.unsqueeze(0)[:, -mask.size(-2):, -mask.size(-1):]

def StageAttention(
    attn_weights: torch.Tensor = None,
    local_mask: torch.Tensor = None,
    cross_mask: torch.Tensor = None,
    attention_dropout = None,
    training = None,

):
    local_attention = attn_weights + local_mask
    
    