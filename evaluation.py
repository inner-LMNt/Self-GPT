import json
import torch

from models.LLM.config import Config
from models.LLM.GPT import GPT
from models.LLM.bigram import BigramLanguageModel
from models.LLM.trigram import TrigramLanguageModel

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def evaluate():
    config = Config()

    # Change based on model type
    model_type = "bigram"
    model_name = "new_model"
    path = config.checkpoint_dir + f"/{model_type}/{model_name}.pth"
    LLM = BigramLanguageModel().to(config.device)

    try:
        LLM.load_state_dict(torch.load(path, weights_only=True))
        print("Model loaded successfully:", path)
        print("Evaluating...")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    data = load_json(config.data_dir + '/' + config.data_file)
    data = torch.tensor(data, dtype=torch.long)
    
    def generate_batch():
        offsets = torch.randint(len(data) - config.context_len, (config.batch_size,))
        inputs = torch.stack([data[i:i+config.context_len] for i in offsets])
        targets = torch.stack([data[i+1:i+config.context_len+1] for i in offsets])
        return inputs.to(config.device), targets.to(config.device)
    
    loss = LLM.forward(*generate_batch())[1].item()
    print(f"Cross-entropy loss: {loss}")
    

def main():
    evaluate()
    print("\nFinished.")

if __name__ == "__main__":
    main()