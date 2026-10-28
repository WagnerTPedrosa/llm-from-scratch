"""
Instruction Following fine tunning

[Stage 1] Preparing the dataset:
- Dataset download and formatting
- Batching the dataset
- creating data loaders

[Stage 2] Fine tunning the LLM:
- Loading a pretrained LLM
- Instruction fine-tuning the LLM
- Inspecting the modeling loss

[Stage 3] Evaluating the LLM
- Extracting responses
- Qualitative evaluation
- Scoring the responses
"""

import json
import os
from functools import partial
import torch
from src.finetuning.instruction_dataset import custom_collate_fn, InstructionDataset
from torch.utils.data import DataLoader
import tiktoken

from src.configs import GPT2_MODEL_CONFIGS
from src.generation import generate, token_ids_to_text, text_to_token_ids

from src.finetuning.instruction_dataset import format_input
from src.model.gpt import GPTModel
from src.pretrained import load_weights_into_gpt
from src.gpt_download import download_and_load_gpt2
from src.evaluation import calc_loss_loader
from src.training import train_model_simple
import time
from tqdm import tqdm
import urllib.request
import psutil


def test_ollama_serve():

    def check_if_running(process_name):
        running = False
        for proc in psutil.process_iter(["name"]):
            if process_name in proc.info["name"]:
                running = True
                break
        return running

    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running")
    print("Ollama Running:", check_if_running("ollama"))


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048},
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url=url, data=payload, method="POST")

    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data


def divide_dataset(data):
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    return train_data, test_data, val_data


def batch_the_dataset(): ...


def train(
    model,
    train_loader,
    val_loader,
    device,
    tokenizer,
    val_data,
    num_epochs=2,
):
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


def extract_and_evaluate_responses(test_data, model, tokenizer, device, context_length):
    torch.manual_seed(123)

    for entry in test_data[:3]:
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer=tokenizer).to(device),
            max_new_tokens=256,
            context_size=context_length,
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )
        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")


def generate_test_set_response(test_data, model, tokenizer, device, BASE_CONFIG):
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer=tokenizer).to(device=device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )

        genereated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (
            genereated_text[len(input_text) :].replace("### Response:", "").strip()
        )
        test_data[i]["model_response"] = response_text
        with open("data/instruction-data-with-response.json", "w") as file:
            json.dump(test_data, file, indent=4)


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
            continue
    return scores


def main():
    file_path = "data/instruction-data.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    train_data, test_data, val_data = divide_dataset(data)

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=1024
    )

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_dataset = InstructionDataset(val_data, tokenizer=tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    test_dataset = InstructionDataset(test_data, tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }

    BASE_CONFIG.update(GPT2_MODEL_CONFIGS["355M"])

    settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = generated_text[len(input_text) :].strip()
    print(response_text)

    model.to(device=device)
    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    # Train once and save the weights. Later runs load them instead of training again
    # (delete the .pth file to train from scratch)
    if os.path.exists("gpt2-medium355M-sft.pth"):
        print(f"Loading fine-tuned model from gpt2-medium355M-sft.pt")
        model.load_state_dict(
            torch.load(
                "gpt2-medium355M-sft.pth", map_location=device, weights_only=True
            )
        )
    else:
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            tokenizer=tokenizer,
            val_data=val_data,
        )
        torch.save(model.state_dict(), "gpt2-medium355M-sft.pth")
        print(f"Model saved to gpt2-medium355M-sft.pth")
    model.eval()

    extract_and_evaluate_responses(
        test_data=test_data,
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=BASE_CONFIG["context_length"],
    )

    generate_test_set_response(
        test_data=test_data,
        model=model,
        tokenizer=tokenizer,
        device=device,
        BASE_CONFIG=BASE_CONFIG,
    )

    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")


if __name__ == "__main__":
    main()
