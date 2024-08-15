# Self GPT

## File Structure

```
Self-GPT/
│
├── data/
|   ├── processed/
│   ├── raw/
│   ├── samples/
|   ├── __init__.py
│   ├── preprocess.py
|   └── tokenizer.py
|
├── models/
│   ├── checkpoints/
|   |   ├── bigram/
|   |   └── trigram/
|   |
│   ├── LLM/
│   │   ├── __init__.py
|   |   ├── attention.py
|   |   ├── bigram.py
│   │   ├── config.py
│   │   ├── GPT.py
│   │   └── trigram.py
│   └── utils.py
|
├── notebooks/
|   └── training.ipynb
|
├── .gitignore
├── evaluation.py
├── inference.py
├── README.md
├── requirements.txt
└── training.py
```
