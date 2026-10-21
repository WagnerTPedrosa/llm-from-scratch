import re

"""
Simple Tokenizer just to understand how a tokenizer works. In the end GPT-2+ uses byte pair encoding (BPE) 
"""


class Tokenizer:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, raw_text) -> list:
        # Split the text into words
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Replaces unknown words by <|unk|> tokens
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]

        # Create the token IDs from the text
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids) -> str:
        # Converts token IDs back into text
        text = " ".join([self.int_to_str[i] for i in ids])

        # Removes spaces before the specified punctuation
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)

        return text
