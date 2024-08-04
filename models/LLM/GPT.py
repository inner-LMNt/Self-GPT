import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.LLM.config import Config

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.init_weights()
    
    def init_weights(self):
        pass

    def forward(self, x):
        pass

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()

    def forward(self, x):
        pass

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()

    def forward(self, x):
        pass

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()

    def forward(self, x):
        pass
