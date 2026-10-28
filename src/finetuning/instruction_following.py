"""Fine-tune GPT-2 (355M) to follow instructions (chapter 7).

    uv run python -m src.finetuning.instruction_following

The last step scores the responses with Llama 3, which needs Ollama running
(`ollama serve`, after `ollama pull llama3`).
"""

import json
import os
import time
import urllib.request
from functools import partial

import psutil
import tiktoken
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation import calc_loss_loader
from src.finetuning.instruction_dataset import (
    InstructionDataset,
    custom_collate_fn,
    format_input,
)
from src.generation import generate, text_to_token_ids, token_ids_to_text
from src.device import get_device
from src.pretrained import load_openai_gpt2
from src.training import train_model_simple

DATA_PATH = "data/instruction-data.json"
RESPONSES_PATH = "data/instruction-data-with-response.json"
FINETUNED_MODEL_PATH = "checkpoints/gpt2-medium355M-sft.pth"


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def split_dataset(data, train_frac=0.85, test_frac=0.1):
    """Split into train / validation (the rest) / test."""
    train_end = int(len(data) * train_frac)
    test_end = train_end + int(len(data) * test_frac)
    return data[:train_end], data[test_end:], data[train_end:test_end]


def create_dataloaders(train_data, val_data, tokenizer, device, batch_size=8):
    # The collate function pads each batch and moves it to the device
    collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    def make_loader(data, shuffle):
        return DataLoader(
            InstructionDataset(data, tokenizer),
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=shuffle,  # Only drop the incomplete last batch when training
            num_workers=0,
        )

    return make_loader(train_data, shuffle=True), make_loader(val_data, shuffle=False)


# ---------------------------------------------------------------------------
# Generating responses
# ---------------------------------------------------------------------------


def generate_response(entry, model, tokenizer, device, max_new_tokens=256):
    """Generate the model's answer to one entry, without the prompt."""
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=model.pos_emb.weight.shape[0],
        eos_id=50256,  # Stop at <|endoftext|>
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # The output repeats the prompt: keep only what comes after it
    return generated_text[len(input_text) :].replace("### Response:", "").strip()


def print_sample_responses(test_data, model, tokenizer, device, num_samples=3):
    for entry in test_data[:num_samples]:
        print(format_input(entry))
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {generate_response(entry, model, tokenizer, device)}")
        print("-------------------------------------")


def generate_test_set_responses(test_data, model, tokenizer, device):
    """Add a "model_response" to each test entry and save them all as JSON."""
    for entry in tqdm(test_data, desc="Generating responses"):
        entry["model_response"] = generate_response(entry, model, tokenizer, device)
    with open(RESPONSES_PATH, "w") as file:
        json.dump(test_data, file, indent=4)


# ---------------------------------------------------------------------------
# Scoring with a bigger LLM (Llama 3 via Ollama)
# ---------------------------------------------------------------------------


def check_ollama_running():
    running = any(
        "ollama" in (proc.info["name"] or "").lower()
        for proc in psutil.process_iter(["name"])
    )
    if not running:
        raise RuntimeError("Ollama is not running. Start it with `ollama serve`.")


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # Fixed seed and temperature 0 so the scores are reproducible
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048},
    }
    request = urllib.request.Request(
        url=url, data=json.dumps(data).encode("utf-8"), method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Ollama streams the answer as one JSON object per line
    response_data = ""
    with urllib.request.urlopen(request) as response:
        for line in response:
            response_data += json.loads(line)["message"]["content"]
    return response_data


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
    return scores


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()

    # [Step 1] Prepare the dataset: split it and create the data loaders
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    train_data, val_data, test_data = split_dataset(data)
    train_loader, val_loader = create_dataloaders(train_data, val_data, tokenizer, device)

    # [Step 2] Load GPT-2 medium (355M) with OpenAI's pretrained weights (no dropout)
    model, _ = load_openai_gpt2("355M", drop_rate=0.0)
    model.to(device)

    # [Step 3] Before fine-tuning: the model can't follow instructions yet
    print("Before fine-tuning:")
    print_sample_responses(val_data, model, tokenizer, device, num_samples=1)
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")

    # [Step 4] Fine-tune, or load the weights saved by a previous run
    # (delete the .pth file to train again)
    if os.path.exists(FINETUNED_MODEL_PATH):
        print(f"Loading fine-tuned model from {FINETUNED_MODEL_PATH}")
        model.load_state_dict(
            torch.load(FINETUNED_MODEL_PATH, map_location=device, weights_only=True)
        )
    else:
        torch.manual_seed(123)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        start_time = time.time()
        train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=2, eval_freq=5, eval_iter=5,
            start_context=format_input(val_data[0]), tokenizer=tokenizer,
        )
        print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
        os.makedirs(os.path.dirname(FINETUNED_MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
        print(f"Model saved to {FINETUNED_MODEL_PATH}")
    model.eval()

    # [Step 5] Compare a few responses with the expected ones
    print_sample_responses(test_data, model, tokenizer, device)

    # [Step 6] Generate a response for every test entry and save them as JSON
    # (slow, so it's skipped if the file exists; delete it after training again)
    if os.path.exists(RESPONSES_PATH):
        print(f"Loading test set responses from {RESPONSES_PATH}")
        with open(RESPONSES_PATH, "r") as file:
            test_data = json.load(file)
    else:
        generate_test_set_responses(test_data, model, tokenizer, device)

    # [Step 7] Score the responses from 0 to 100 with Llama 3
    check_ollama_running()
    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()
