import torch

from models.GPT.LLM import GPT, BigramLanguageModel, TrigramLanguageModel
from data.tokenizer import Tokenizer

def trigram_inference():
    LLM = GPT()
    tokenizer = Tokenizer()
    config = LLM.config

    m = TrigramLanguageModel(config.vocab_size)
    m.load_state_dict(torch.load('models/checkpoints/trigram/model.pth', weights_only=True))
    out = m.generate(torch.zeros((1, 2), dtype=torch.long), 5000)
    out = out[:, 2:]

    path = config.save_dir + '/trigram.txt'
    with open(path, 'w') as f:
        f.write(tokenizer.decode(out.squeeze().tolist()))


def main():
    trigram_inference()

if __name__ == '__main__':
    main()