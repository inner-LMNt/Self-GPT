import torch

from models.LLM.config import Config
from models.LLM.GPT import GPT
from models.LLM.bigram import BigramLanguageModel
from models.LLM.trigram import TrigramLanguageModel
from data.tokenizer import Tokenizer

def bigram_inference():
    config = Config()
    LLM = BigramLanguageModel()
    tokenizer = Tokenizer()

    LLM.load_state_dict(torch.load('models/checkpoints/bigram/shakespeare_model.pth', weights_only=True))
    context = "M"
    context = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=config.device)
    out = LLM.generate(context, 5000)

    path = config.save_dir + '/bigram_out.txt'
    with open(path, 'w') as f:
        f.write(tokenizer.decode(out.squeeze().tolist()))

def trigram_inference():
    config = Config()
    LLM = TrigramLanguageModel()
    tokenizer = Tokenizer()

    LLM.load_state_dict(torch.load('models/checkpoints/trigram/shakespeare_model.pth', weights_only=True))
    context = "Ma"
    context = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=config.device)
    out = LLM.generate(context, 5000)
    
    path = config.save_dir + '/trigram_out.txt'
    with open(path, 'w') as f:
        f.write(tokenizer.decode(out.squeeze().tolist()))


def main():
    # bigram_inference()
    trigram_inference()
    print("Finished")

if __name__ == '__main__':
    main()