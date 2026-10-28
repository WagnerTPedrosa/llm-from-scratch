"""
Class that formats and pretokenizes all inputs in the dataset

1. Format data using promt template
2. Tokenize formatted data
3. Adjust to the same length with padding tokens
4. Create target token IDs for training
5 Replace padding tokens with placeholders (replace by -100 to exclude them from the training loss since PyTorch cross_entropy loss function as ignore_index=-100)
"""

from typing import Any

import torch
from torch.utils.data import Dataset


# Function that converts the entries in the data into the Alpaca-style format (we can also use the Phi-3 prompt style format)
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",
):
    # finds the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    # Pads and prepares inputs
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # Pads sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # removes extra padded token added earlier
        inputs = torch.tensor(padded[:-1])

        # Shifts +1 to the right for targets
        targets = torch.tensor(padded[1:])

        # Replaces all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncates to the maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # converts the list of inputs to a tensor and transfers it to the target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index) -> Any:
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def main():
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)
    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)


if __name__ == "__main__":
    main()
