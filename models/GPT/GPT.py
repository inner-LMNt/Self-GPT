import torch
from torch import nn
from config import Config

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.context_length, self.config.hidden_size))
        self.drop = nn.Dropout(self.config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.num_layers)])
        self.ln_final = nn.LayerNorm(self.config.hidden_size)
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()
    
    def init_weights(self):
        pass

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.attn = Attention()
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
