import math
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn

from ops import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        d_model: int = 512,
    ):
        """Multihead attention block

        Args:
            num_heads: The number of attention heads
            d_model: The hidden dimension of the model

        Returns:
            None
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads since d_model/num_heads is used as"
                f"the dimension of the query, key, and value tensors. Received {num_heads=}, {d_model=}"
            )
        d_query = d_model // num_heads

        # Use same weight initilization as nn.Linear
        self.Q = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(num_heads, d_model, d_query), a=math.sqrt(5)
            ),
            requires_grad=True,
        )
        self.K = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(num_heads, d_model, d_query), a=math.sqrt(5)
            ),
            requires_grad=True,
        )
        self.V = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(num_heads, d_model, d_query), a=math.sqrt(5)
            ),
            requires_grad=True,
        )

        self.Wout = nn.Linear(d_model, d_model, bias=False)

        # We can just store the self attention values in the instance i guess if we want to look at them later
        self.attentions = None

    def forward(
        self,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project inputs down to queries, keys, and values and then perform scaled dot product attention."""
        # Project inputs to queries/keys/values
        # Note:
        #   * Q has shape [num_heads, d_model, d_query]
        #   * x_query has shape [bs, L, d_model]
        # We need to add singleton dimensions to make:
        #   * Q have shape [1, num_heads, d_model, d_query] and
        #   * x_query have shape [bs, 1, L, d_model]
        # before doing our matrix multiply so that everything is broadcasted correctly
        # The result of the matrix multiply has shape [bs, num_heads, L, d_query]
        q = torch.matmul(x_query.unsqueeze(1), self.Q.unsqueeze(0))
        k = torch.matmul(x_key.unsqueeze(1), self.K.unsqueeze(0))
        v = torch.matmul(x_value.unsqueeze(1), self.V.unsqueeze(0))

        # Perform scaled dot product attention
        heads, self.attentions = scaled_dot_product_attention(q, k, v, mask)

        # Concat heads together and project
        return self.Wout(heads.view(q.shape[0], q.shape[2], -1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, seq_len: int = 1024, dropout: float = 0.1):
        """todo: docstring"""
        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        self.pos_enc = torch.zeros(seq_len, self.d_model)
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)

        self.pos_enc = nn.Parameter(self.pos_enc, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = x + self.pos_enc[: x.shape[-2], :]
        return self.dropout(enc)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        d_model: int = 512,
        d_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """todo: docstring"""
        super().__init__()

        self.multihead_self_attention = MultiHeadAttention(num_heads, d_model)
        self.layernorm_self_attention = nn.LayerNorm(d_model)
        self.dropout_self_attention = nn.Dropout(p=dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model),
        )
        self.layernorm_feedforward = nn.LayerNorm(d_model)
        self.dropout_feedforward = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """todo: docstring"""
        out = self.multihead_self_attention(x, x, x, mask)
        out = self.dropout_self_attention(out)
        out = out + x
        res = self.layernorm_self_attention(out)

        out = self.feedforward(res)
        out = self.dropout_feedforward(out)
        out = out + res
        out = self.layernorm_feedforward(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        d_model: int = 512,
        d_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """todo: docstring"""
        super().__init__()

        self.multihead_self_attention = MultiHeadAttention(num_heads, d_model)
        self.layernorm_self_attention = nn.LayerNorm(d_model)
        self.dropout_self_attention = nn.Dropout(p=dropout)

        self.multihead_source_attention = MultiHeadAttention(num_heads, d_model)
        self.layernorm_source_attention = nn.LayerNorm(d_model)
        self.dropout_source_attention = nn.Dropout(p=dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model),
        )
        self.layernorm_feedforward = nn.LayerNorm(d_model)
        self.dropout_feedforward = nn.Dropout(p=dropout)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """todo: docstring"""
        # self-attention
        out = self.multihead_self_attention(target, target, target, target_mask)
        out = self.dropout_self_attention(out)
        out = out + target
        res = self.layernorm_self_attention(out)

        # source-attention
        out = self.multihead_source_attention(res, source, source, source_mask)
        out = self.dropout_source_attention(out)
        out = out + res
        res = self.layernorm_source_attention(out)

        # feedforward
        out = self.feedforward(res)
        out = self.dropout_feedforward(out)
        out = out + res
        out = self.layernorm_feedforward(out)

        return out


if __name__ == "__main__":
    num_heads = 8
    d_model = 64
    context_length = 10
    bs = 4

    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
    x = torch.randn(bs, context_length, d_model)
    mask = torch.tril(torch.ones(context_length, context_length))
    out = mha(x, x, x, mask)

    # Visualize the attention for each head for one element in the batch
    rows = 2
    cols = num_heads // rows
    batch_element = 2
    fig, axs = plt.subplots(rows, cols)
    for i in range(num_heads):
        ax = axs[i // cols, i % cols]
        ax.matshow(mha.attentions[i][batch_element].detach())
        ax.set_title(f"self-attention head {i}")

    # Visualize positional encoding
    pe = PositionalEncoding()
    fig, ax = plt.subplots()
    ax.matshow(pe.pos_enc)

    # Smoke test to make sure EncoderLayer can do forward pass
    d_ff = 2048
    enc_layer = EncoderLayer(num_heads=num_heads, d_model=d_model, d_feedforward=d_ff)
    out = enc_layer(x, mask)

    # Smoke test to make sure DecoderLayer can do forward pass
    dec_layer = DecoderLayer(num_heads=num_heads, d_model=d_model, d_feedforward=d_ff)
    out = dec_layer(x, x, mask, mask)

    plt.show()
