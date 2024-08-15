import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.tokenizer import Tokenizer

class Config:
    def __init__(self):
        # Model (params taken from GPT-2)
        self.context_len = 256      # Number of tokens to self-attend to
        self.embed_size = 768       # Vector size of embeddings
        self.head_size = 64         # Attention head size
        self.inference_len = 1024   # Number of tokens to generate
        self.intermediate_size = 4 * self.embed_size    # Size of feed-forward layer
        self.num_heads = 12         # Number of attention heads in multi-head attention
        self.num_layers = 12        # Transformer layers
        self.pdrop = 0.1            # Dropout probability (for attention head, multi-head, feed-forward)
        self.vocab_size = Tokenizer().vocab_size

        # Training
        self.batch_size = 64
        self.learning_rate = 5e-5
        self.num_epochs = 10
        self.train_split = 0.85

        # Data
        self.data_dir = "./data"
        self.data_file = "processed/tinyshakespeare.json"

        # Checkpoint
        self.checkpoint_dir = "./models/checkpoints"

        # Save
        self.save_dir = "./data/samples"

        # Miscalaneous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 1234
