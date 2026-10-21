"""
Simplefied variant of self-attention, free from any trainable weigths
"""

import torch


def compute_attention_scores(inputs):

    attention_scores = torch.empty(6, 6)
    # Here we use matrix multiplication cause is faster but we can use this as well:
    """for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attention_scores[i,j] = torch.dot(x_i, x_j)"""

    attention_scores = inputs @ inputs.T
    print(f"Attention Scores:\n {attention_scores}")

    return attention_scores


def normalize_attention_scores(attention_scores, dim=-1):
    """
    Here we use the softmax for normalization where the formula is torch.exp(x) / torch.exp(x).sum(dim=0) but this formula may encounter overflows if one value is way higher than the other
    So we can use the torch softmax function that as already been optimized
    """
    attention_weights = torch.softmax(attention_scores, dim=dim)
    print(f"Attention weights:\n {attention_weights}")
    return attention_weights


def compute_context_vectors(attention_weights, inputs):
    all_context_vecs = attention_weights @ inputs
    print(f"All context vectors:\n {all_context_vecs}")

    return all_context_vecs


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

    # [Step 1] Compute the attention scores as the dot product to measure the similarity of how closely two vectores are aligned
    # the higher the dot product, the higher the similarity and attention score between two elements.

    attention_scores = compute_attention_scores(inputs=inputs)

    # [Step 2] Normalize the attention scores we comptued in order to obtain attention weights that sum up to 1 (useful for interpretation and maintaining training stability un an LLM).

    attention_weights = normalize_attention_scores(attention_scores, dim=-1)

    # [Step 3] Calculate the context vector for each token
    all_context_vecs = compute_context_vectors(attention_weights, inputs)
    print(all_context_vecs)


if __name__ == "__main__":
    main()
