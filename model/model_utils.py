import torch
import torch.nn as nn

def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'non':
        act_layer = nn.Identity()
    else:
        raise KeyError(f"Unknown activation function {act_type}.")
    return act_layer
