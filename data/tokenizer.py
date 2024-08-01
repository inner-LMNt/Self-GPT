class Tokenizer:
    def __init__(self):
        pass

    def encode(self, text): # character level tokenization
        chars = list(" !\"#$%&'()*+,-./0123456789:;?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
        tokens = [chars.index(c) for c in text]
        return tokens
    