class Tokenizer:
    def __init__(self):
        self.chars = list(" \n0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;?@[\]^_`{|}~")
        self.vocab_size = len(self.chars)

    def encode(self, text): # character level tokenization
        tokens = [self.chars.index(c) if c in self.chars else 0 for c in text]
        return tokens
    
    def decode(self, tokens):
        text = [self.chars[t] if t < len(self.chars) else '' for t in tokens]
        return ''.join(text)
