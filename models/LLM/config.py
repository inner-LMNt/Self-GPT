import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.tokenizer import Tokenizer

class Config:
    def __init__(self):
        # Model
        self.vocab_size = Tokenizer().vocab_size
        self.context_len = 32 #128, 512
        self.inference_len = 1024
        self.embed_size = 64 # GPT-2 uses 768
        self.pdrop = 0.1
        self.num_layers = 12 # Transformer layers
        self.head_size = 64 # Attention head size, same as embed_size for now
        self.head_num = 12 # Number of attention heads in multi-head attention
        self.intermediate_size = 3072 # Size of feed-forward layer
        self.act = "gelu"

        # Training
        self.learning_rate = 1e-3 #5e-5
        self.batch_size = 32
        self.num_epochs = 3

        # Data
        self.data_dir = "./data"
        self.data_file = "processed/tinyshakespeare.json"

        # Checkpoint
        self.checkpoint_dir = "./models/checkpoints"
        self.save_steps = 500
        self.save_total_limit = 5

        # Save
        self.save_dir = "./data/samples"

        # Misc
        self.seed = 1234
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
