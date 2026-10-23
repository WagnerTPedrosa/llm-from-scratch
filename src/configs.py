GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers (number of transformer blocks in the model)
    "embedding_layer_drop_rate": 0.1,       # Embeding Layer Dropout rate
    "shorcut_drop_rate": 0.1,       # Shorcut Dropout rate
    "mha_drop_rate": 0.1,       # Multi head attention Dropout rate
    "qkv_bias": False,      # Query-Key-Value bias
}
