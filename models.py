from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn

from layers import EncoderLayer, DecoderLayer, PositionalEncoding


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


class Decoder(nn.Module):
    def __init__(
        self,
        num_decoders: int = 6,
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

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads,
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoders)
            ]
        )

        # Mask that prevents future tokens from attending to previous tokens
        self.subsequent_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """todo: docstring"""
        out = self.pos_enc(target)

        # todo: maybe this should be moved to DecoderLayer?
        # Combine target and subsequent masks
        if target_mask is not None:
            target_mask = target_mask * self.subsequent_mask

        for decoder in self.decoder_layers:
            out = decoder(source, out, source_mask, target_mask)
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


class Seq2Seq(nn.Module):
    def __init__(
        self,
        num_encoders: int = 6,
        num_decoders: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        d_feedforward: int = 2048,
        dropout: float = 0.1,
        num_tokens: int = 50257,  # vocab size of GPT-2 BPE tokenizer
        seq_len: int = 256,
    ):
        super().__init__()

        # Share embeddings for encoder and decoder
        self.embedding = nn.Embedding(num_tokens, d_model)

        self.encoder = Encoder(
            num_encoders=num_encoders,
            num_heads=num_heads,
            d_model=d_model,
            d_feedforward=d_feedforward,
            dropout=dropout,
            seq_len=seq_len,
        )

        self.decoder = Decoder(
            num_decoders=num_decoders,
            num_heads=num_heads,
            d_model=d_model,
            d_feedforward=d_feedforward,
            dropout=dropout,
            seq_len=seq_len,
        )

        # Project output of final decoder layer to logits over token space
        self.projection = nn.Sequential(
            nn.Linear(d_model, num_tokens),
            nn.ReLU(),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ):
        source_emb = self.embedding(source)
        target_emb = self.embedding(target)

        enc_out = self.encoder(source_emb, mask=source_mask)
        dec_out = self.decoder(
            enc_out, target_emb, source_mask=source_mask, target_mask=target_mask
        )

        logits = self.projection(dec_out)
        return logits


if __name__ == "__main__":
    bs = 4
    L = 100
    d_model = 64

    x = torch.randn(bs, L, d_model)
    src = torch.randint(50257, (bs, L))
    trg = torch.randint(50257, (bs, L))
    src_mask = torch.concat([torch.ones(bs, L, L - 20), torch.zeros(bs, L, 20)], dim=-1)
    trg_mask = torch.flip(src_mask, dims=(-1,))

    # Smoke test to make sure encoder can do forward pass
    enc = Encoder(d_model=d_model, seq_len=L)
    out = enc(x, mask=src_mask)

    # Smoke test for decoder
    dec = Decoder(d_model=d_model, seq_len=L)
    out = dec(x, out, source_mask=src_mask, target_mask=trg_mask)

    # Smoke test for encoder-only classifier
    model = EncoderOnlyClassifier(
        num_encoders=3, num_heads=4, d_model=d_model, seq_len=L
    )
    out = model(src, mask=src_mask)
    plt.matshow(
        model.encoder.encoder_layers[0]
        .multihead_self_attention.attentions[0][0]
        .detach()
    )

    # Smoke test for encoder-decoder model
    seq2seq = Seq2Seq(
        num_encoders=3, num_decoders=3, num_heads=4, d_model=d_model, seq_len=L
    )
    out = seq2seq(src, trg, source_mask=src_mask, target_mask=trg_mask)
    plt.matshow(
        seq2seq.encoder.encoder_layers[0]
        .multihead_self_attention.attentions[0][0]
        .detach()
    )
    plt.matshow(
        seq2seq.decoder.decoder_layers[0]
        .multihead_self_attention.attentions[0][0]
        .detach()
    )

    plt.show()
