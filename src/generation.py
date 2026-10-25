import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    for _ in range(max_new_tokens):
        # Crop to the last context_size tokens, the most the model supports
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Only the last position matters: it holds the prediction for the next token
        logits = logits[:, -1, :]

        # Top-k: keep only the k most likely tokens, set the rest to -inf
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )
        # Temperature > 1 flattens the distribution (more random), < 1 sharpens it.
        # With temperature 0 we use greedy decoding (always the most likely token)
        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(input=logits, dim=-1)
            idx_next = torch.multinomial(input=probas, num_samples=1)
        else:
            idx_next = torch.argmax(
                input=logits,
                dim=-1,
                keepdim=True,
            )
        if idx_next == eos_id:  # Stop early at the end-of-sequence token
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
