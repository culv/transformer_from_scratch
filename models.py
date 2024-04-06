from typing import Optional

import torch
from torch import nn

from layers import EncoderLayer, PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        num_encoders: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        d_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        """todo: docstring"""
        super().__init__()
        self.pos_enc = PositionalEncoding(
            d_model=d_model, max_len=max_len, dropout=dropout
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads=num_heads,
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_encoders)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """todo: docstring"""
        out = self.pos_enc(x)
        for encoder in self.encoder_layers:
            out = encoder(out, mask)
        return out


if __name__ == "__main__":
    # Smoke test to make sure encoder can do forward pass
    enc = Encoder()
    x = torch.randn(4, 10, 512)
    out = enc(x)
