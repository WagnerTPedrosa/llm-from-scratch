import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Uses a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            # The target is the input shifted by one token: at each position
            # the model learns to predict the next token
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Return the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    # Returns a single row from the dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def create_train_val_loaders(
    text, context_length, batch_size=2, train_ratio=0.9, for_training=True
):
    """Split text into train/val and build a loader for each.

    for_training=True shuffles the train loader and drops its last incomplete batch.
    """
    split_idx = int(train_ratio * len(text))
    train_loader = create_dataloader_v1(
        text[:split_idx],
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=for_training,
        drop_last=for_training,
    )
    val_loader = create_dataloader_v1(
        text[split_idx:],
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader
