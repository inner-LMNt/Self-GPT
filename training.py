import torch
import json
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.LLM.config import Config
from models.LLM.GPT import GPT
from models.LLM.bigram import BigramLanguageModel
from models.LLM.trigram import TrigramLanguageModel

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def n_gram(number=2):
    config = Config()
    if number == 2:
        LLM = BigramLanguageModel().to(config.device)
    else:
        LLM = TrigramLanguageModel().to(config.device)
    # torch.manual_seed(config.seed)

    data = load_json(config.data_dir + '/' + config.data_file)
    data = torch.tensor(data, dtype=torch.long)

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    
    def generate_batch(train):
        data = train_data if train else val_data
        offsets = torch.randint(len(data) - config.context_length, (config.batch_size,))
        inputs = torch.stack([data[i:i+config.context_length] for i in offsets])
        targets = torch.stack([data[i+1:i+config.context_length+1] for i in offsets])
        return inputs.to(config.device), targets.to(config.device)

    print("Training...")
    start = time.time()
    optimizer = torch.optim.Adam(LLM.parameters(), lr=config.learning_rate)
    for _ in range(100000): # 1000, 10000, 100000
        x, y = generate_batch(True)
        logits, loss = LLM.forward(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # Gradients
        optimizer.step() # Update weights
    
    print(f"Loss: {loss.item()}")
    print(f"Training time: {time.time() - start} seconds")
    if number == 2:
        path = config.checkpoint_dir + '/bigram/model.pth'
    else:
        path = config.checkpoint_dir + '/trigram/model.pth'
    torch.save(LLM.state_dict(), path)


def main():
    n_gram(2)
    print("Finished")

if __name__ == '__main__':
    main()