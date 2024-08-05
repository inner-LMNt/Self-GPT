import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.LLM.config import Config

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.query = nn.Linear(self.config.embed_size, self.config.head_size, bias=False)
        self.key = nn.Linear(self.config.embed_size, self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.embed_size, self.config.head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(self.config.context_len, self.config.context_len)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) / (C**0.5)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)

        return weights @ v