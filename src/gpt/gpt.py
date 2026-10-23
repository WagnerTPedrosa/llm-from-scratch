import tiktoken

import torch
import torch.nn as nn
from src.configs import GPT_CONFIG_124M
from src.gpt.transformer_blocks.layer_normalization import LayerNorm
from src.gpt.transformer_blocks.feed_forward import FeedForward
from src.gpt.transformer import TransformerBlock
from src.gpt.transformer_blocks.layer_normalization import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Initializes the token and positional embedding layers using the config (cfg).
        # These embedding layers are responsible for converting input token indices into dence vectors and adding positional information
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["embedding_layer_drop_rate"])

        # Creates a sequential stack of TransformerBlock modules equal to the number of layers specified in cfg
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Layer normalization allows for standardizing the outputs from the transformer blocks to stabilize the learning process
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # A linear output head without bias is defined, which projects the transformer’s output into the vocabulary space of the tokenizer to generate logits for each token in the vocabulary.
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # [Step 1] compute token and posicional embbeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds

        # [Step 2] Dropout
        x = self.drop_emb(x)

        # [Step 3] Process the data through the transformer blocks
        x = self.trf_blocks(x)

        # [Step 4] Applies normalization that allows for standardizing the outputs from the transformer blocks to stabilize the learning process
        x = self.final_norm(x)

        # [Step 5] A linear output head without bias is defined, which projects the transformer’s output into the vocabulary space of the tokenizer to generate logits for each token in the vocabulary.
        logits = self.out_head(x)

        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(GPT_CONFIG_124M)

    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


if __name__ == "__main__":
    main()
