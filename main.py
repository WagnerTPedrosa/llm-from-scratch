"""Entry point.

    python main.py train                 # train a GPT from scratch on the-verdict.txt
    python main.py eval                  # loss of OpenAI's GPT-2 weights on the-verdict.txt
    python main.py generate "Some text"  # generate text with OpenAI's GPT-2 weights
"""

import argparse

import tiktoken
import torch

from src.configs import GPT_CONFIG_124M
from src.data import create_train_val_loaders
from src.device import get_device
from src.evaluation import calc_loss_loader
from src.generation import generate, text_to_token_ids, token_ids_to_text
from src.model.gpt import GPTModel
from src.pretrained import load_openai_gpt2
from src.training import train_model_simple

DATA_PATH = "data/the-verdict.txt"


def load_text():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return file.read()


def train(args):
    torch.manual_seed(123)
    device = get_device()
    model = GPTModel(GPT_CONFIG_124M).to(device)
    train_loader, val_loader = create_train_val_loaders(
        load_text(), context_length=GPT_CONFIG_124M["context_length"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tiktoken.get_encoding("gpt2"),
    )


def evaluate(args):
    torch.manual_seed(123)
    device = get_device()
    model, _ = load_openai_gpt2(args.model_size)
    model.to(device)
    train_loader, val_loader = create_train_val_loaders(
        load_text(), context_length=256, for_training=False
    )
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Train loss: {train_loss:.3f}")
    print(f"Val loss: {val_loss:.3f}")


def generate_text(args):
    device = get_device()
    model, cfg = load_openai_gpt2(args.model_size)
    model.to(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(args.prompt, tokenizer).to(device),
        max_new_tokens=args.max_new_tokens,
        context_size=cfg["context_length"],
        top_k=50,
        temperature=1.5,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("train")
    p.add_argument("--epochs", type=int, default=10)
    p.set_defaults(func=train)

    p = commands.add_parser("eval")
    p.add_argument("--model-size", default="124M")
    p.set_defaults(func=evaluate)

    p = commands.add_parser("generate")
    p.add_argument("prompt", nargs="?", default="Every effort moves you")
    p.add_argument("--max-new-tokens", type=int, default=25)
    p.add_argument("--model-size", default="124M")
    p.set_defaults(func=generate_text)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
