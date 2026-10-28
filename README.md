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
  finetuning/
    classification.py    # fine-tune GPT-2 as a spam classifier (chapter 6)
    data/spamdataset.py  # SpamDataset: tokenize + pad SMS messages
    instruction_dataset.py   # Alpaca prompt format, InstructionDataset, collate function
    instruction_following.py # fine-tune GPT-2 medium to follow instructions (chapter 7)
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
| 6 | Fine-tuning for classification | `src/finetuning/` |  Done |
| 7 | Fine-tuning to follow instructions | `src/finetuning/` | Done |

## Working with text data (chapter 2)

How raw text becomes the numbers a model can train on: **text → tokens → token IDs → embeddings**.

```bash
uv run python -m chapters.data_preparation_and_sampling.data_preparation_sampling
```

**Step 1: Tokenize.**
- **Simple tokenizer** (`chapters/data_preparation_and_sampling/tokenizer.py`). It splits text on whitespace and punctuation and maps each word to an ID from a fixed vocabulary. Any word not in the vocabulary becomes `<|unk|>`, so information is lost.
- **Byte pair encoding (BPE)**, used by GPT-2 through `tiktoken`. Common words are single tokens. Rare words are split into smaller known pieces, down to single bytes if needed. Any text can be encoded, so there is no `<|unk|>`. The vocabulary has 50,257 tokens.
- **`<|endoftext|>`** marks the boundary between documents. It is reused later as the padding token.

**Step 2: Create input/target pairs with a sliding window** (`GPTDataset` in `src/data.py`).
- **The target is the input shifted by one token.** For the input `[In, the, heart, of]` the target is `[the, heart, of, the]`. Every position is a training example: predict the next token from the ones before it. This is why pretraining needs no labels, because the text is its own label.
- **`max_length`** is how many tokens the model sees at once (the context length).
- **`stride`** is how far the window moves each time. With `stride = max_length` the chunks do not overlap, so the model does not see the same text twice per epoch, which helps against overfitting. A smaller stride gives more (overlapping) examples.

**Step 3: Turn token IDs into embeddings.**
- **Token embeddings.** A token ID is just an index, with no meaning. `nn.Embedding(vocab_size, emb_dim)` is a trainable lookup table: row *i* is the vector for token *i*. It is equivalent to a one-hot vector times a weight matrix, but faster.
- **Positional embeddings.** Attention on its own has no idea of word order, so "dog bites man" and "man bites dog" would look the same. GPT-2 adds a learned vector for each position (`nn.Embedding(context_length, emb_dim)`).
- **Input to the model** = token embedding + positional embedding.

## Attention mechanisms (chapter 3)

Attention lets each token's representation depend on the other tokens in the context. For example, "bank" means something different in "river bank" and "bank account". The output for each token is a **context vector**: a weighted mix of the information from all the tokens it attends to.

The code builds this up in 4 versions, from simplest to the real one:

```bash
uv run python -m chapters.self_attention.simplified_self-attention
uv run python -m chapters.self_attention.causal_self_attention
uv run python -m chapters.self_attention.multi_head_self_attention
uv run python -m src.model.attention
```

**Step 1: Simplified attention, without trainable weights** (`simplified_self-attention.py`).
- **Attention scores.** The dot product of every pair of input vectors (`inputs @ inputs.T`). A higher dot product means the vectors are more similar.
- **Attention weights.** Softmax turns each row of scores into weights that are positive and sum to 1.
- **Context vectors.** `weights @ inputs`: each token becomes a weighted average of all tokens.

**Step 2: Self-attention with trainable Q, K, V** (`SelfAttention` in `self_attention.py`).
- **Query, Key and Value.** Three `Linear` layers project each token into a *query* (what am I looking for?), a *key* (what do I contain?) and a *value* (what do I pass on if selected?). Scores are `Q @ K.T`, and the context is `weights @ V`. Because Q, K and V are learned, the model learns *what* to pay attention to.
- **Scaling by √d_k.** With large dimensions the dot products get large, the softmax becomes almost one-hot, and the gradients become tiny. Dividing by the square root of the key dimension keeps the scores in a good range.

**Step 3: Causal attention** (`CausalAttention`).
- **Causal mask.** GPT predicts the next token, so a token must not see the tokens after it, because that would be cheating. Scores above the diagonal are set to `-inf` before the softmax, so their weight becomes exactly 0.
- **Why `-inf` and not 0 after the softmax.** Masking after the softmax would leave rows that no longer sum to 1. Masking before means the softmax normalizes over the allowed tokens only.
- **Dropout on the attention weights.** Randomly zeroes some weights during training, to reduce overfitting. It is off in `model.eval()`.

**Step 4: Multi-head attention** (`src/model/attention.py`).
- **Several heads in parallel.** Instead of one attention with `d_out` dimensions, use `num_heads` attentions with `head_dim = d_out / num_heads` each. Every head has its own projections, so different heads can focus on different relations (for example syntax in one head, nearby words in another).
- **Done as one big matrix multiplication.** Q, K and V are reshaped to `(batch, tokens, heads, head_dim)` and transposed to `(batch, heads, tokens, head_dim)`. PyTorch then computes all heads at once with batched matmul. The step-by-step comments in `attention.py` show the shapes at each point.
- **`out_proj`.** After the heads are concatenated back to `d_out`, a final linear layer mixes the information from the different heads.

## Implementing a GPT model (chapter 4)

Putting the pieces together into GPT-2 (see the [architecture diagram](#architecture) above). All the code is in `src/model/`.

**Step 1: LayerNorm** (`layer_norm.py`).
- **What it does.** Normalizes each token's vector to mean 0 and variance 1. This keeps the values in a stable range through many layers, so training is more stable and faster.
- **Trainable `scale` and `shift`.** They let the model undo the normalization where that helps.
- **Pre-LayerNorm.** GPT-2 applies LayerNorm *before* attention and before the feed forward. This trains more stably than putting it after, as the original Transformer did.

**Step 2: GELU and the feed forward network** (`gelu.py`, `feed_forward.py`).
- **GELU.** A smooth version of ReLU. Small negative inputs give small negative outputs instead of exactly 0, which can make optimization easier.
- **FeedForward.** Expands each vector from 768 to 3,072 dimensions (4×), applies GELU, and projects back to 768. The larger hidden layer gives the model room to compute richer features.
- **Attention vs feed forward.** Attention mixes information *between* tokens. The feed forward processes *each token on its own*.

**Step 3: Shortcut (residual) connections** (`transformer.py`).
- **What they are.** The output of a layer is added to its input: `x = x + f(x)`.
- **Why.** In deep networks the gradients can shrink to almost nothing on the way back to the first layers (vanishing gradients). The addition gives the gradient a direct path backwards, so all 12 blocks can learn.

**Step 4: The transformer block.**
- **Structure.** `LayerNorm → attention → dropout → + shortcut`, then `LayerNorm → feed forward → dropout → + shortcut`.
- **Same shape in and out** (`batch, tokens, 768`). This is what lets us stack 12 blocks with `nn.Sequential`.

**Step 5: The GPT model** (`gpt.py`).
- **Pipeline.** Embeddings → dropout → 12 transformer blocks → final LayerNorm → output head.
- **Output head.** A `Linear(768, 50257)` that gives one logit per vocabulary token, at every position. The logits at position *i* are the prediction for token *i + 1*.
- **Why "124M".** Our model has ~163M parameters, because the output head (768 × 50,257 ≈ 38.6M) is separate from the token embedding. GPT-2 uses **weight tying**: the output head reuses the token embedding matrix. Without counting it you get the ~124M in the name.

**Step 6: Generating text (greedy).** Feed the tokens in, take the logits of the *last* position, pick the most likely token (`argmax`), append it, and repeat. The input is cropped to the last `context_length` tokens, because the positional embeddings only go that far. Before training, the output is random tokens.

## Pretraining on unlabeled data (chapter 5)

Training the model to predict the next token, and loading OpenAI's pretrained weights.

```bash
uv run python main.py train --epochs 10
uv run python main.py eval
uv run python main.py generate "Every effort moves you"
```

**Step 1: Measure how wrong the model is** (`src/evaluation.py`).
- **Cross-entropy loss.** For every position, take the probability the model gave to the *correct* next token. The loss is the average of `-log(probability)`. If the model is sure and right, the loss is close to 0. If it gives the right token a tiny probability, the loss is large.
- **Shapes.** The logits `(batch, tokens, vocab)` and targets `(batch, tokens)` are flattened, so every token position counts as one classification example.
- **Perplexity** = `exp(loss)`. It can be read as "how many tokens the model is choosing between". An untrained model has a loss of about 10.8, which is `ln(50257)`: a uniform guess over the whole vocabulary.

**Step 2: Split train/validation.** 90% of *The Verdict* for training, 10% for validation. The validation loss shows whether the model learns general patterns or only memorizes the training text.

**Step 3: The training loop** (`src/training.py`).
- **Each batch.** `optimizer.zero_grad()` (clear the old gradients) → forward → loss → `loss.backward()` (compute the gradients) → `optimizer.step()` (update the weights).
- **AdamW** (lr 4e-4, weight decay 0.1). Adam adapts the step size for each parameter. Weight decay pulls the weights towards 0 to reduce overfitting.
- **Monitoring.** Every `eval_freq` steps the train and validation loss are estimated on `eval_iter` batches, and after each epoch a sample text is printed to see the progress.
- **What to expect.** *The Verdict* is tiny (~5k tokens), so after a few epochs the training loss keeps falling while the validation loss stops improving: the model is memorizing. This is expected with so little data.

**Step 4: `model.train()` vs `model.eval()` and `torch.no_grad()`.**
- **`train()`/`eval()`.** These turn dropout on and off. Always use `eval()` when evaluating or generating.
- **`torch.no_grad()`.** Tells PyTorch not to store what it needs for gradients. Evaluation becomes faster and uses less memory.

**Step 5: Better text generation** (`src/generation.py`).
- **Greedy decoding** (`argmax`) always picks the most likely token, so the text is deterministic and tends to repeat itself.
- **Temperature.** Divide the logits by T before the softmax. T > 1 flattens the distribution (more random and creative). T < 1 sharpens it (more conservative). We then *sample* with `torch.multinomial` instead of taking the argmax.
- **Top-k.** Keep only the k most likely tokens and set the rest to `-inf` before sampling. This stops very unlikely (nonsense) tokens from being picked when the temperature is high.

**Step 6: Load OpenAI's GPT-2 weights** (`src/pretrained.py`).
- **Why.** Pretraining GPT-2 on a large corpus costs far more compute than a laptop has. OpenAI released the weights, so we load them into *our* architecture.
- **Config changes to match OpenAI.** `context_length = 1024` and `qkv_bias = True`.
- **Converting the weights.** OpenAI stores Q, K and V in one matrix, which we split in 3. TensorFlow stores linear weights as `(in, out)` and PyTorch as `(out, in)`, hence the `.T`. The output head reuses the token embeddings (weight tying).
- **Check.** `main.py eval` gives a train loss of 3.755 and a validation loss of 3.560 on *The Verdict*. This matches the book, which confirms the weights were loaded correctly.

## Fine-tuning for classification (chapter 6)

Turns the pretrained GPT-2 into a spam classifier for SMS messages.

```bash
uv run python -m src.finetuning.classification
```

**Step 1: Prepare the dataset.**
- **Balance the classes.** The dataset has 747 spam and 4,825 ham messages. A model that always answers "ham" would already be about 87% accurate without learning anything. To avoid this, we keep only 747 random ham messages (undersampling), so both classes are 50/50 and the accuracy means something. This is the simplest option, but it throws away most of the ham data. Alternatives that keep all the data are oversampling spam or giving spam a higher weight in the loss.
- **Split** into 70% train, 10% validation and 20% test. Validation is used to watch for overfitting during training. Test is only looked at in the end.
- **Pad the messages.** Tensors in a batch must all have the same length, so every message is padded with `<|endoftext|>` (token 50256) up to the longest training message (120 tokens). Validation and test use the same length.

**Step 2: Load GPT-2 with OpenAI's weights.** The pretrained model already understands English, so it only has to learn the task. Training from scratch on ~1,000 messages would not work. Dropout is set to 0, as in the book.

**Step 3: Turn it into a classifier.**
- **Replace the output head.** The original head maps each 768-dim vector to 50,257 logits, one per token in the vocabulary. The new one maps it to 2 logits: not spam (0) or spam (1).
- **Freeze most of the model.** Only the last transformer block, the final LayerNorm and the new head are trained. This is faster, needs less data, and keeps the general language knowledge in the lower layers.
- **Use only the last token.** Because of the causal mask, each token only attends to the tokens before it. The last token is the only one that has seen the whole message, so its output is used for the prediction (`model(x)[:, -1, :]`).

**Step 4: Check accuracy before fine-tuning.** It should be about 50%, which is what a random head gets on balanced data. This sanity check confirms that the data and the evaluation work before spending minutes on training.

**Step 5: Fine-tune.**
- **Loss.** Cross-entropy between the last token's 2 logits and the label.
- **Optimizer.** AdamW with a small learning rate (5e-5), so the pretrained weights are adjusted gently instead of overwritten. Weight decay (0.1) keeps the weights small to reduce overfitting.
- **Monitoring.** 5 epochs. Every 50 steps the train and validation loss are estimated on 5 batches. This is cheaper than the full dataset and enough to see the trend.

**Step 6: Measure accuracy on the full datasets.** The book reports about 97% on train and validation and about 96% on test. A large gap between train and test would mean overfitting.

**Step 7: Classify new messages.** `classify_review` applies the same preprocessing as training: tokenize, truncate, pad to 120 tokens, then take the argmax of the last token's logits.

**Step 8: Save the model.** `torch.save(model.state_dict())` saves only the weights. To load them, build the same architecture first (GPT-2 plus the 2-class head) and call `load_state_dict(torch.load(path, weights_only=True))`. `weights_only=True` stops `torch.load` from running arbitrary code from the file.

## Fine-tuning to follow instructions (chapter 7)

Teaches GPT-2 medium (355M) to answer instructions like "Rewrite the sentence using a simile" instead of just continuing the text.

```bash
uv run python -m src.finetuning.instruction_following
```

**Step 1: Prepare the dataset** (`src/finetuning/instruction_dataset.py`).
- **The data.** 1,100 entries, each with an `instruction`, an optional `input` and the expected `output`. Split into 85% train, 10% test and 5% validation.
- **Prompt format (Alpaca style).** `format_input` turns every entry into the same template: a fixed intro, `### Instruction:`, `### Input:` (only if there is one) and `### Response:`. The model learns that after `### Response:` comes the answer. The same template must be used at training and at generation time.
- **Pad per batch, not per dataset.** Unlike chapter 6, `custom_collate_fn` pads each batch only to its own longest entry. This wastes fewer tokens, since entry lengths vary a lot.
- **Targets = inputs shifted by one**, the same next-token task as pretraining. The model is trained on the whole text, prompt included.
- **Ignore the padding in the loss.** All padding tokens in the targets except the first are replaced by `-100`. PyTorch's `cross_entropy` skips targets equal to `-100` (`ignore_index`), so the model is not trained to predict padding. The first one is kept as `<|endoftext|>`, so the model learns where to stop.

**Step 2: Load GPT-2 medium with OpenAI's weights.** The 124M model is too small to follow instructions well, so we use the 355M (24 layers, 16 heads, 1,024-dim embeddings). Dropout is set to 0.

**Step 3: Check the model before fine-tuning.** Given a prompt, the pretrained model just repeats or continues it, because it was only trained to predict the next token. The initial loss is about 3.83 (train) and 3.76 (validation), matching the book.

**Step 4: Fine-tune.**
- **All the weights are trained** (no freezing as in chapter 6), because the model must change how it writes, not just add a small head.
- **The training loop is the same as in pretraining** (`train_model_simple`): only the data changed. AdamW, lr 5e-5, weight decay 0.1, 2 epochs.
- **Save the weights** to `gpt2-medium355M-sft.pth` ("sft" = supervised fine-tuning). Later runs load the file instead of training again. Delete it to retrain.

**Step 5: Look at a few responses.** `generate_response` formats the entry, generates with greedy decoding until `<|endoftext|>`, and cuts off the prompt from the output. Comparing a few answers with the expected ones by hand is the quickest check.

**Step 6: Generate responses for the whole test set** and save them to `data/instruction-data-with-response.json`, with a new `model_response` field. This is slow, so if the file exists it is loaded instead. Delete it after training again.

**Step 7: Score the responses with a bigger LLM.**
- **Why.** Answers to instructions can be correct in many different words, so there is no simple accuracy. Instead, a larger model judges them ("LLM as a judge").
- **How.** [Ollama](https://ollama.com) runs Llama 3 (8B) locally. For each entry it gets the prompt, the expected output and the model's response, and is asked for a score from 0 to 100. `seed` and `temperature 0` make the scores reproducible.
- **Before running.** Install Ollama, then `ollama pull llama3` and `ollama serve`. `check_ollama_running` stops early with a clear error if it is not running.
- **Result.** The book gets an average score of about 50. It is a rough measure, useful for comparing models or settings with each other.

## Credits

Based on *Build a Large Language Model (From Scratch)* by Sebastian Raschka. `src/gpt_download.py` is taken from the [book's repository](https://github.com/rasbt/LLMs-from-scratch).
