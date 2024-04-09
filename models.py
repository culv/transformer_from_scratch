from typing import Optional

import matplotlib.pyplot as plt
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
        seq_len: int = 1024,
    ):
        """todo: docstring"""
        super().__init__()
        self.pos_enc = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout=dropout
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


class EncoderOnlyClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_encoders: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        d_feedforward: int = 2048,
        dropout: float = 0.1,
        num_tokens: int = 50257,  # vocab size of GPT-2 BPE tokenizer
        seq_len: int = 256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_tokens, d_model)

        self.encoder = Encoder(
            num_encoders=num_encoders,
            num_heads=num_heads,
            d_model=d_model,
            d_feedforward=d_feedforward,
            dropout=dropout,
            seq_len=seq_len,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * d_model, num_classes),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.embedding(x)
        out = self.encoder(out, mask)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    # Smoke test to make sure encoder can do forward pass
    enc = Encoder()
    x = torch.randn(4, 10, 512)
    out = enc(x)

    # Smoke test for encoder-only classifier
    x = torch.randint(50256, (4, 160))
    mask = torch.tril(torch.ones(160, 160))
    model = EncoderOnlyClassifier(
        num_classes=2, num_encoders=3, num_heads=4, d_model=256, seq_len=160
    )
    out = model(x, mask)
    plt.matshow(
        model.encoder.encoder_layers[0].multi_head_attention.attentions[0][0].detach()
    )
    plt.show()
