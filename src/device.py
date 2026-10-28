import torch


def get_device():
    """Use the best available device: an NVIDIA/AMD GPU (cuda), Apple GPU (mps), or the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
