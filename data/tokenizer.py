class Tokenizer:
    def __init__(self):
        pass

    def encode(self, text): # character level tokenization
        chars = list(" \n0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;?@[\]^_`{|}~")
        tokens = [chars.index(c) for c in text]
        return tokens
    
    def decode(self, tokens):
        chars = list(" \n0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;?@[\]^_`{|}~")
        text = [chars[t] for t in tokens]
        return ''.join(text)
