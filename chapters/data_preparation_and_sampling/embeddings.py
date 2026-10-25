import torch
from torch import nn


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, output_dim: int, context_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, output_dim)
        self.position_embedding = nn.Embedding(context_length, output_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_embeddings = self.position_embedding(positions)
        return token_embeddings + position_embeddings