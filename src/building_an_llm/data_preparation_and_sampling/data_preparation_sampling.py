"""
O flow de um tokenizer é:
1. A partir do input text gerar o tokenized text
2. pegar nesses tokenized text (ex. [[This], [is], [an], [example], [.]]) e passar para token IDs
3. Criar os token embeddings
"""

import tiktoken
from src.building_an_llm.data_preparation_and_sampling.gptDataset import GPTDataset
from src.building_an_llm.model.embeddings import InputEmbeddings
from torch.utils.data import DataLoader
import torch


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # inicialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the dataset
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def main():
    with open("data/the-veredict.txt", "r", encoding="utf-8") as f:
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
