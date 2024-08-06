import torch
import json
import time

from models.LLM.config import Config
from models.LLM.GPT import GPT
from models.LLM.bigram import BigramLanguageModel, BigramNoAttention
from models.LLM.trigram import TrigramLanguageModel, TrigramNoAttention

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def n_gram(number=2):
    config = Config()
    # torch.manual_seed(config.seed)

    model_name = "shakespeare_model"
    # model_name = "basic_model"
    
    if number == 2:
        path = config.checkpoint_dir + f"/bigram/{model_name}.pth"
        LLM = BigramLanguageModel().to(config.device)
        # LLM = BigramNoAttention().to(config.device)
    else:
        path = config.checkpoint_dir + f"/trigram/{model_name}.pth"
        LLM = TrigramLanguageModel().to(config.device)
        # LLM = TrigramNoAttention().to(config.device)

    try:
        LLM.load_state_dict(torch.load(path, weights_only=True))
        print("Model loaded successfully:", path)
        print("Training...")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training model from scratch...")

    data = load_json(config.data_dir + '/' + config.data_file)
    data = torch.tensor(data, dtype=torch.long)

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    
    def generate_batch(train):
        data = train_data if train else val_data
        offsets = torch.randint(len(data) - config.context_len, (config.batch_size,))
        inputs = torch.stack([data[i:i+config.context_len] for i in offsets])
        targets = torch.stack([data[i+1:i+config.context_len+1] for i in offsets])
        return inputs.to(config.device), targets.to(config.device)

    start = time.time()
    optimizer = torch.optim.Adam(LLM.parameters(), lr=config.learning_rate)
    for _ in range(10000):
        x, y = generate_batch(True)
        logits, loss = LLM.forward(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # Gradients
        optimizer.step() # Update weights
    
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


def main():
    n_gram(2)
    print("\nFinished.")

if __name__ == "__main__":
    main()