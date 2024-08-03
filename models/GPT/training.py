import torch
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.GPT.LLM import GPT, BigramLanguageModel, TrigramLanguageModel

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def n_gram():
    LLM = GPT()
    config = LLM.config
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
        return inputs, targets

    m = TrigramLanguageModel(config.vocab_size)

    optimizer = torch.optim.Adam(m.parameters(), lr=config.learning_rate)
    for _ in range(100000):
        x, y = generate_batch(True)
        logits, loss = m.forward(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")
    torch.save(m.state_dict(), 'models/checkpoints/trigram/model.pth')


def main():
    n_gram() # N-gram demonstration

if __name__ == '__main__':
    main()