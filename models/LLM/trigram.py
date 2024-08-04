import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.LLM.config import Config

### Trigram: P(w_i | w_{i-2}, w_{i-1})
class TrigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        # Embedding basically maps the previous two tokens to the probabilities of the next token
        self.embedding = nn.Embedding(self.config.vocab_size**2, self.config.hidden_size)
        self.linear_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.vocab_size = self.config.vocab_size

    def forward(self, x, targets=None):
        x = x[:, :-1] * self.vocab_size + x[:, 1:]
        x = self.embedding(x)
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
            logits, loss = self.forward(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
        return x