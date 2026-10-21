import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Uses a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Return the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    # Returns a single row from the dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
