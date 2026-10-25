# LLM from scratch

A GPT-2 (124M) implemented from scratch in PyTorch, following [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka ([official repo](https://github.com/rasbt/LLMs-from-scratch)).


## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
uv run python main.py train --epochs 10                   # train from scratch on data/the-verdict.txt
uv run python main.py eval                                # loss of OpenAI's GPT-2 on data/the-verdict.txt
uv run python main.py generate "Every effort moves you"   # generate text with OpenAI's GPT-2
```

`eval` and `generate` accept `--model-size` (`124M`, `355M`, `774M`, `1558M`). The device is picked automatically: CUDA, then Apple MPS, then CPU.

## Project structure

```
main.py                  # entry point: train | eval | generate
src/
  configs.py             # model configs (GPT_CONFIG_124M, GPT-2 sizes)
  model/                 # the architecture, nothing else
    attention.py         # MultiHeadAttention (causal)
    gelu.py              # GELU activation
    feed_forward.py      # FeedForward (Linear → GELU → Linear)
    layer_norm.py        # LayerNorm
    transformer.py       # TransformerBlock
    gpt.py               # GPTModel
  data.py                # GPTDataset and dataloaders (sliding window over tokens)
  generation.py          # text ↔ token ids, generate (temperature, top-k)
  evaluation.py          # cross-entropy loss per batch / per loader
  training.py            # training loop
  pretrained.py          # load OpenAI's GPT-2 weights into GPTModel
  gpt_download.py        # download GPT-2 checkpoint (from the book's repo)
chapters/                # step-by-step study code from the book
  data_preparation_and_sampling/
  self_attention/
data/the-verdict.txt     # small training text used in the book
```

Code in `src/` is what `main.py` uses. Code in `chapters/` is the step-by-step version from the book, kept for learning. Run any of it as a module, for example:

```bash
uv run python -m chapters.self_attention.multi_head_self_attention
```

## Architecture

```
token ids
   │
   ├─ token embedding + positional embedding
   ├─ dropout
   │
   ├─ TransformerBlock × n_layers
   │     ├─ LayerNorm → MultiHeadAttention → dropout → + shortcut
   │     └─ LayerNorm → FeedForward        → dropout → + shortcut
   │
   ├─ final LayerNorm
   └─ linear output head → logits (vocab_size)
```

GPT-2 124M: 12 layers, 12 heads, 768-dim embeddings, 50,257-token vocabulary.

## Book chapters → code

| Chapter | Topic | Code | Status |
|---|---|---|---|
| 2 | Working with text data | `src/data.py`, `chapters/data_preparation_and_sampling/` |  Done |
| 3 | Attention mechanisms | `src/model/attention.py`, `chapters/self_attention/` | Done |
| 4 | Implementing a GPT model | `src/model/` |  Done |
| 5 | Pretraining on unlabeled data | `src/training.py`, `src/evaluation.py`, `src/generation.py`, `src/pretrained.py` | Done |
| 6 | Fine-tuning for classification | |  In progress |
| 7 | Fine-tuning to follow instructions | | In progress |

## Results

OpenAI's GPT-2 124M weights on *The Verdict* (`main.py eval`):

| Split | Loss |
|---|---|
| Train | 3.755 |
| Val | 3.560 |

Sample from `main.py generate` (top-k 50, temperature 1.5):

```
Every effort moves you like...but in doing that, you fail.
```

## Credits

Based on *Build a Large Language Model (From Scratch)* by Sebastian Raschka. `src/gpt_download.py` is taken from the [book's repository](https://github.com/rasbt/LLMs-from-scratch).
