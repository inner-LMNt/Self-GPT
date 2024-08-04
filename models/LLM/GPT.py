import torch
from torch import nn

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_weights()
        pass
    
    def init_weights(self):
        pass

    def forward(self, x):
        pass

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass
