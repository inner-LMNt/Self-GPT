import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.GPT.config import Config

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
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


### N-gram language models, one of the simplest language models
### Bigram: P(w_i | w_{i-1})
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.embedding(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
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

### Trigram: P(w_i | w_{i-2}, w_{i-1})
class TrigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size * vocab_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x, targets=None):
        x = x[:, :-1] * self.vocab_size + x[:, 1:]
        logits = self.embedding(x)

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
