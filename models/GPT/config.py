import torch

class Config:
    def __init__(self):
        # Model
        self.vocab_size = 93
        self.context_length = 16 #128, 512
        self.max_length = 1024
        self.hidden_size = 768
        self.pdrop = 0.1
        self.num_layers = 12 # Transformer layers
        self.num_attention_heads = 12
        self.intermediate_size = 3072 # Size of feed-forward layer
        self.act = "gelu"

        # Training
        self.learning_rate = 1e-2 #5e-5
        self.batch_size = 32 #16, 32
        self.num_epochs = 3

        # Data
        self.data_dir = "./data"
        self.data_file = "processed/interview.json"

        # Checkpoint
        self.checkpoint_dir = "./models/checkpoints"
        self.save_steps = 500
        self.save_total_limit = 5

        # Save
        self.save_dir = "./data/samples"
        

        # Misc
        self.seed = 1234
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
