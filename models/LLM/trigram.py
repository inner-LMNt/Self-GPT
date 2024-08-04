import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.LLM.config import Config
from models.LLM.attention import AttentionHead

### Trigram: P(w_i | w_{i-2}, w_{i-1})
class TrigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.tok_embedding = nn.Embedding(self.config.vocab_size**2, self.config.embed_size)
        self.pos_embedding = nn.Embedding(self.config.context_length, self.config.embed_size)
        self.attention_head = AttentionHead()
        self.linear_layer = nn.Linear(self.config.embed_size, self.config.vocab_size)
        self.vocab_size = self.config.vocab_size

    def forward(self, x, targets=None):
        x = x[:, :-1] * self.vocab_size + x[:, 1:]
        B, T = x.shape
        token_embed = self.tok_embedding(x)
        position_embed = self.pos_embedding(torch.arange(T, device=x.device))
        x = token_embed + position_embed
        x = self.attention_head(x)
        logits = self.linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, x, n):
        for _ in range(n):
            x_trunc = x[:, -self.config.context_length:]
            logits, loss = self.forward(x_trunc)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
        return x