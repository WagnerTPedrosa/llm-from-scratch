import torch
import torch.nn as nn

from src.model.layer_norm import LayerNorm
from src.model.transformer import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Token embeddings (what each token is) and positional embeddings (where it is)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Creates a sequential stack of TransformerBlock modules equal to the number of layers specified in cfg
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Normalizes the output of the last transformer block to stabilize training
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Projects each embedding to vocab_size logits (one score per token in the vocabulary)
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # [Step 1] Token + positional embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds

        # [Step 2] Dropout on the embeddings
        x = self.drop_emb(x)

        # [Step 3] Transformer blocks
        x = self.trf_blocks(x)

        # [Step 4] Final normalization
        x = self.final_norm(x)

        # [Step 5] Logits over the vocabulary
        logits = self.out_head(x)

        return logits
