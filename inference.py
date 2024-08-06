import torch

from models.LLM.config import Config
from models.LLM.GPT import GPT
from models.LLM.bigram import BigramLanguageModel, BigramNoAttention
from models.LLM.trigram import TrigramLanguageModel, TrigramNoAttention
from data.tokenizer import Tokenizer

def bigram_inference(save=False):
    config = Config()
    tokenizer = Tokenizer()

    LLM = BigramLanguageModel()
    model_name = "shakespeare_model"

    # LLM = BigramNoAttention()
    # model_name = "basic_model"

    LLM.load_state_dict(torch.load(f'models/checkpoints/bigram/{model_name}.pth', weights_only=True))
    context = "M"
    context = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=config.device)
    out = LLM.generate(context, config.inference_len)

    if save:
        out_name = "bigram_out"
        path = config.save_dir + f"/{out_name}.txt"
        with open(path, 'w') as f:
            f.write(tokenizer.decode(out.squeeze().tolist()))
    else:
        print(tokenizer.decode(out.squeeze().tolist()))

def trigram_inference(save=False):
    config = Config()
    tokenizer = Tokenizer()

    LLM = TrigramLanguageModel()
    model_name = "shakespeare_model"
    
    # LLM = TrigramNoAttention()
    # model_name = "basic_model"

    LLM.load_state_dict(torch.load(f"models/checkpoints/trigram/{model_name}.pth", weights_only=True))
    context = "QU"
    context = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=config.device)
    out = LLM.generate(context, config.inference_len)
    
    if save:
        out_name = "trigram"
        path = config.save_dir + f"/{out_name}.txt"
        with open(path, 'w') as f:
            f.write(tokenizer.decode(out.squeeze().tolist()))
    else:
        print(tokenizer.decode(out.squeeze().tolist()))


def main():
    # bigram_inference()
    trigram_inference()
    print("\nFinished.")

if __name__ == "__main__":
    main()