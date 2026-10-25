import torch
import torch.nn as nn


class GELU(nn.Module):
    """GELU activation, tanh approximation (the one GPT-2 uses).

    Unlike ReLU it is smooth and lets small negative values through, which can
    make optimization easier.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
