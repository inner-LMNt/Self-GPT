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
        self.projection = nn.Linear(self.config.head_size, self.config.embed_size)
        self.dropout = nn.Dropout(self.config.pdrop)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        weights = Q @ K.transpose(-2, -1) / (C**0.5) # B x T x T
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        return weights @ V # B x T x C
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.attention_heads = nn.ModuleList([AttentionHead() for _ in range(self.config.num_heads)])
        self.projection = nn.Linear(self.config.num_heads * self.config.head_size, self.config.embed_size)
        self.dropout = nn.Dropout(self.config.pdrop)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.attention_heads], dim=-1)
        x = self.dropout(self.projection(x))
        return x
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.proj1 = nn.Linear(self.config.embed_size, self.config.intermediate_size)
        self.proj2 = nn.Linear(self.config.intermediate_size, self.config.intermediate_size)
        self.proj3 = nn.Linear(self.config.intermediate_size, self.config.embed_size)

        self.seq = nn.Sequential(
            self.proj1,
            nn.GELU(),
            self.proj2, # Extra layer compared to regular GPT
            nn.GELU(),
            self.proj3,
            nn.Dropout(self.config.pdrop)
        )

    def forward(self, x):
        return self.seq(x)
    
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.attention_heads = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.norm1 = nn.LayerNorm(self.config.embed_size)
        self.norm2 = nn.LayerNorm(self.config.embed_size)

    def forward(self, x):
        # Residual skip connection
        x = x + self.attention_heads(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x