"""Fine-tune GPT-2 to classify SMS messages as spam or not spam (chapter 6).

    uv run python -m src.finetuning.classification
"""

import os
import time

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader

from src.finetuning.spam_dataset import SpamDataset
from src.device import get_device
from src.pretrained import load_openai_gpt2

DATA_DIR = "data/sms_spam_collection"
MODEL_PATH = "checkpoints/review_classifier.pth"


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def create_balanced_dataset(df):
    """Undersample "ham" so both classes have as many examples as "spam"."""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])


def random_split(df, train_frac, validation_frac):
    """Shuffle and split into train / validation / test (the rest)."""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


def create_dataloaders(tokenizer, batch_size=8, num_workers=0):
    """Balance the dataset, split it, save each split as CSV and build a DataLoader for each."""
    df = pd.read_csv(
        f"{DATA_DIR}/SMSSpamCollection.tsv", sep="\t", header=None, names=["Label", "Text"]
    )
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    splits = random_split(balanced_df, train_frac=0.7, validation_frac=0.1)
    for name, split_df in zip(["train", "validation", "test"], splits):
        split_df.to_csv(f"{DATA_DIR}/{name}.csv", index=None)

    # Validation and test are padded/truncated to the longest training message
    train_dataset = SpamDataset(f"{DATA_DIR}/train.csv", tokenizer, max_length=None)
    val_dataset = SpamDataset(f"{DATA_DIR}/validation.csv", tokenizer, train_dataset.max_length)
    test_dataset = SpamDataset(f"{DATA_DIR}/test.csv", tokenizer, train_dataset.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def prepare_for_classification(model, num_classes=2):
    """Replace the output head with a classification head and pick which layers to train."""
    # Freeze the whole model, making all layers non-trainable
    for param in model.parameters():
        param.requires_grad = False

    # The new head is randomly initialized: seed it so results are reproducible
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(model.out_head.in_features, num_classes)

    # Train the new head (trainable by default), the last transformer block and the final norm
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Loss and accuracy (only the last token is used: it has seen the whole message)
# ---------------------------------------------------------------------------


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    correct_predictions, num_examples = 0, 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()
    return correct_predictions / num_examples


def print_accuracies(model, train_loader, val_loader, test_loader, device, num_batches=None):
    loaders = {"Training": train_loader, "Validation": val_loader, "Test": test_loader}
    for name, loader in loaders.items():
        accuracy = calc_accuracy_loader(loader, model, device, num_batches)
        print(f"{name} accuracy: {accuracy * 100:.2f}%")


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Resets loss gradients from the previous batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculates loss gradients
            optimizer.step()  # Updates model weights using loss gradients
            examples_seen += input_batch.shape[0]  # Tracks examples instead of tokens
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        # Accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)

    # Truncate to what the model supports, then pad to the length used in training
    supported_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_ids[: min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last token
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()

    # [Step 1] Prepare the dataset: balance, split and create the data loaders
    train_loader, val_loader, test_loader = create_dataloaders(tokenizer, batch_size=8)
    loaders = (train_loader, val_loader, test_loader)

    # [Step 2] Load GPT-2 with OpenAI's pretrained weights (no dropout for fine-tuning)
    model, _ = load_openai_gpt2("124M", drop_rate=0.0)

    # [Step 3] Turn it into a classifier: 2-class output head, train only the last layers
    prepare_for_classification(model, num_classes=2)
    model.to(device)

    # [Step 4] Accuracy before fine-tuning (~50%: the new head is random)
    torch.manual_seed(123)
    print_accuracies(model, *loaders, device, num_batches=10)

    # [Step 5] Fine-tune
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    start_time = time.time()
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=5, eval_freq=50, eval_iter=5,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    # [Step 6] Accuracy after fine-tuning, on the full datasets
    print_accuracies(model, *loaders, device)

    # [Step 7] Classify new messages
    max_length = train_loader.dataset.max_length
    for text in [
        "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
        "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
    ]:
        print(f"{classify_review(text, model, tokenizer, device, max_length)}: {text}")

    # [Step 8] Save the fine-tuned model. Load it back with:
    #   model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
