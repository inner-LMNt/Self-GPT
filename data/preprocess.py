import argparse
import json
from tokenizer import Tokenizer

def load_data(file_path):
    file_path = './raw/' + file_path + '.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def save_data(data, file_path):
    file_path = './processed/' + file_path + '.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def preprocess(raw_data, tokenizer):
    tokens = tokenizer.encode(raw_data)
    return tokens

def main(args):
    tokenizer = Tokenizer()
    raw_data = load_data(args.input)
    processed_data = preprocess(raw_data, tokenizer)
    save_data(processed_data, args.output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text data for training.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Name of the data file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Name of the file to save data to.")

    args = parser.parse_args()
    main(args)
