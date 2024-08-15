import torch
import json
import time

from models.LLM.config import Config
from models.LLM.GPT import GPT
from models.LLM.bigram import BigramLanguageModel
from models.LLM.trigram import TrigramLanguageModel

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def train_n_gram(n=2):
    config = Config()
    torch.manual_seed(config.seed)

    model_name = "new_model"

    if n == 2:
        path = config.checkpoint_dir + f"/bigram/{model_name}.pth"
        LLM = BigramLanguageModel().to(config.device)
    else:
        path = config.checkpoint_dir + f"/trigram/{model_name}.pth"
        LLM = TrigramLanguageModel().to(config.device)

    try:
        LLM.load_state_dict(torch.load(path, weights_only=True))
        print("Model loaded successfully:", path)
        print("Training...")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training model from scratch...")

    data = load_json(config.data_dir + '/' + config.data_file)
    data = torch.tensor(data, dtype=torch.long)

    split = int(config.train_split * len(data))
    train_data, val_data = data[:split], data[split:]
    
    def generate_batch(train):
        data = train_data if train else val_data
        offsets = torch.randint(len(data) - config.context_len, (config.batch_size,))
        inputs = torch.stack([data[i:i+config.context_len] for i in offsets])
        targets = torch.stack([data[i+1:i+config.context_len+1] for i in offsets])
        return inputs.to(config.device), targets.to(config.device)

    start = time.time()
    optimizer = torch.optim.Adam(LLM.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs):
        x, y = generate_batch(True)
        logits, loss = LLM.forward(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # Gradients
        optimizer.step() # Update weights
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    print("\n______________________________________________________")
    print(f"Training loss: {loss.item()}, Validation loss: {LLM.forward(*generate_batch(False))[1].item()}")
    print(f"Training time: {time.time() - start} seconds")
    print("______________________________________________________\n")

    print(f"Saving model to {path}")
    try:
        torch.save(LLM.state_dict(), path)
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")

def train_gpt():
    config = Config()
    torch.manual_seed(config.seed)

    model_name = "GPT-mini"
    path = config.checkpoint_dir + f"/gpt/{model_name}.pth"
    LLM = GPT().to(config.device)

    try:
        LLM.load_state_dict(torch.load(path, weights_only=True))
        print("Model loaded successfully:", path)
        print("Training...")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training model from scratch...")

    data = load_json(config.data_dir + '/' + config.data_file)
    data = torch.tensor(data, dtype=torch.long)

    split = int(config.train_split * len(data))
    train_data, val_data = data[:split], data[split:]
    
    def generate_batch(train):
        data = train_data if train else val_data
        offsets = torch.randint(len(data) - config.context_len, (config.batch_size,))
        inputs = torch.stack([data[i:i+config.context_len] for i in offsets])
        targets = torch.stack([data[i+1:i+config.context_len+1] for i in offsets])
        return inputs.to(config.device), targets.to(config.device)

    start = time.time()
    optimizer = torch.optim.Adam(LLM.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs):
        x, y = generate_batch(True)
        logits, loss = LLM.forward(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # Gradients
        optimizer.step() # Update weights
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    print("\n______________________________________________________")
    # print(f"Training loss: {loss.item()}, Validation loss: {LLM.forward(*generate_batch(False))[1].item()}") # Too slow
    print(f"Training time: {time.time() - start} seconds")
    print("______________________________________________________\n")

    print(f"Saving model to {path}")
    try:
        torch.save(LLM.state_dict(), path)
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    # train_n_gram(2)
    train_gpt()
    print("\nFinished.")

if __name__ == "__main__":
    main()