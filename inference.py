import torch

from models.LLM.config import Config
from models.LLM.GPT import GPT
from models.LLM.bigram import BigramLanguageModel
from models.LLM.trigram import TrigramLanguageModel
from data.tokenizer import Tokenizer

def n_gram_inference(n=2, save=False):
    config = Config()
    tokenizer = Tokenizer()

    if n == 2:
        LLM = BigramLanguageModel()
        model_name = "new_model"
        LLM.load_state_dict(torch.load(f'models/checkpoints/bigram/{model_name}.pth', weights_only=True))
        context = "M"
    else:
        LLM = TrigramLanguageModel()
        model_name = "shakespeare_model"
        LLM.load_state_dict(torch.load(f"models/checkpoints/trigram/{model_name}.pth", weights_only=True))
        context = "QU"

    context = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=config.device)
    out = LLM.generate(context, config.inference_len)
    
    if save:
        out_name = "bigram_out" if n == 2 else "trigram_out"
        path = config.save_dir + f"/{out_name}.txt"
        with open(path, 'w') as f:
            f.write(tokenizer.decode(out.squeeze().tolist()))
    else:
        print(tokenizer.decode(out.squeeze().tolist()))

def gpt_inference(save=False):
    config = Config()
    tokenizer = Tokenizer()

    GPT_model = GPT()
    model_name = "GPT-mini"
    GPT_model.load_state_dict(torch.load(f"models/checkpoints/gpt/{model_name}.pth", weights_only=True))
    context = "Who"

    context = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=config.device)
    out = GPT_model.generate(context, config.inference_len)
    
    if save:
        path = config.save_dir + "/GPT_out.txt"
        with open(path, 'w') as f:
            f.write(tokenizer.decode(out.squeeze().tolist()))
    else:
        print(tokenizer.decode(out.squeeze().tolist()))

def main():
    # n_gram_inference(n=2, save=True)
    gpt_inference(save=True)
    print("\nFinished.")

if __name__ == "__main__":
    main()