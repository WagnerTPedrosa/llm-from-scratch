from src.building_an_llm.self_attention.self_attention import SelfAttention, CausalAttention
import torch


def compute_attention_matrix_and_attention_weights(queries, keys):
    attention_scores = queries @ keys.T
    attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
    return attention_scores, attention_weights


def create_and_apply_mask(attention_scores):
    context_length = attention_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attention_scores.masked_fill(mask.bool(), -torch.inf)

    return masked


def normalize_attention_weights(masked, keys):
    attention_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)

    return attention_weights


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
    context_length = batch.shape[1]
    ca = CausalAttention(3, 2, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)
    


if __name__ == "__main__":
    main()
