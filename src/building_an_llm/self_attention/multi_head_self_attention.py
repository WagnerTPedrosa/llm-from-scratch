import torch
import torch.nn as nn

from src.building_an_llm.self_attention.self_attention import CausalAttention

""" 
This multi head attention class uses the CausalAttention class and instanciates the number of heads.
This is not computational efficient.
"""


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in=d_in,
                    d_out=d_out,
                    context_length=context_length,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Maskes future tokens
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


def forward(self, x):
    # [Step 1] Compute the Queries (Q), Keys (K), and Values (V).
    # Each linear layer projects the input embeddings from d_in to d_out.
    # Input shape:  (batch, num_tokens, d_in)
    # Output shape: (batch, num_tokens, d_out)
    b, num_tokens, d_in = x.shape

    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    # [Step 2] Reshape Q, K, and V to separate the attention heads.
    # We split the last dimension (d_out) into num_heads and head_dim,
    # where head_dim = d_out // num_heads.
    # This allows each head to work with its own portion of Q, K, and V.
    # The reshape only reorganizes the tensor; it does not compute attention.
    # Shape: (batch, num_tokens, d_out)
    #     -> (batch, num_tokens, num_heads, head_dim)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    # [Step 3] Transpose Q, K, and V to place the heads before the tokens.
    # PyTorch treats batch and num_heads as batch dimensions,
    # allowing attention to be computed independently for each head.
    # All heads can be processed in parallel using batched matrix multiplication.
    # Shape: (batch, num_tokens, num_heads, head_dim)
    #     -> (batch, num_heads, num_tokens, head_dim)
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    # [Step 4] Compute attention scores for each head.
    # Each query is compared with all keys using the dot product.
    # Transposing the last two dimensions of K allows matrix multiplication.
    # Q:   (batch, num_heads, num_tokens, head_dim)
    # K^T: (batch, num_heads, head_dim, num_tokens)
    # Result: (batch, num_heads, num_tokens, num_tokens)
    # Each head produces an attention matrix showing token-to-token scores.
    attention_scores = queries @ keys.transpose(2, 3)

    # [Step 5] Apply the causal mask to prevent attention to future tokens.
    # The mask is sliced to match the actual sequence length.
    # Mask shape: (num_tokens, num_tokens)
    # Broadcasting applies the same mask to every batch and attention head.
    # Masked positions are set to -inf, resulting in zero probability
    # after the softmax operation.
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attention_scores.masked_fill_(mask_bool, -torch.inf)

    # [Step 6] Scale, normalize, and apply dropout to attention scores.
    # Dividing by sqrt(head_dim) prevents large dot products from
    # making the softmax distribution excessively sharp.
    # Softmax (dim=-1) converts scores into attention weights across
    # the key/token dimension, so each query attends to previous tokens
    # and itself (but never to future tokens).
    # Dropout randomly zeros some attention weights during training
    # to reduce overfitting. After dropout, weights may no longer sum to 1.
    # Shape: (batch, num_heads, num_tokens, num_tokens)
    attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
    attention_weights = self.dropout(attention_weights)

    # [Step 7] Compute the context vectors for each attention head.
    # Multiply the attention weights by V to obtain a weighted sum
    # of the value vectors for each token.
    # Attention weights: (batch, num_heads, num_tokens, num_tokens)
    # Values:            (batch, num_heads, num_tokens, head_dim)
    # Result:            (batch, num_heads, num_tokens, head_dim)
    # Transpose to move the token dimension before the heads dimension.
    # Final shape: (batch, num_tokens, num_heads, head_dim)
    context_vec = (attention_weights @ values).transpose(1, 2)

    # [Step 8] Concatenate the outputs of all attention heads.
    # Since d_out = num_heads * head_dim, we merge the last two dimensions.
    # contiguous() ensures the tensor has a compatible memory layout
    # for view(), since transpose() may produce a non-contiguous tensor.
    # Shape: (batch, num_tokens, num_heads, head_dim)
    #     -> (batch, num_tokens, d_out)
    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

    # [Step 9] Apply the final learned linear projection.
    # This mixes information from the different attention heads.
    # The output shape remains (batch, num_tokens, d_out).
    # Although optional in custom implementations, this projection
    # is a standard component of Multi-Head Attention in Transformers.
    context_vec = self.out_proj(context_vec)

    return context_vec


def main():
    # [Step 0] Here we start with an embedding vector of the frase "Your journey starts with one step"
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your (x^1)
            [0.55, 0.87, 0.66],  # journey (x^2)
            [0.57, 0.85, 0.64],  # starts (x^3)
            [0.22, 0.58, 0.33],  # with (x^4)
            [0.77, 0.25, 0.10],  # one (x^5)
            [0.05, 0.80, 0.55],  # step (x^6)
        ]
    )

    batch = torch.stack((inputs, inputs), dim=0)

    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == "__main__":
    main()
