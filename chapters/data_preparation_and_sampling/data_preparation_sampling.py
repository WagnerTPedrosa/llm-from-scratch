"""Data pipeline:

1. Split the input text into tokens (e.g. ["This", "is", "an", "example", "."])
2. Map each token to a token ID
3. Turn the token IDs into embeddings (token + positional)
"""

import torch

from chapters.data_preparation_and_sampling.embeddings import InputEmbeddings
from src.data import create_dataloader_v1


def main():
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    vocab_size = 50257
    output_dim = 256

    torch.manual_seed(123)
    embedding_layer = InputEmbeddings(vocab_size, output_dim, context_length=max_length)
    input_embeddings = embedding_layer(inputs)
    print("Input embeddings shape:\n", input_embeddings.shape)


if __name__ == "__main__":
    main()
