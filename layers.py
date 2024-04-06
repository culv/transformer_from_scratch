from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn

from ops import scaled_dot_product_attention


# i guess this is technically a sublayer
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        d_model: int = 512,
    ):
        """Multihead attention block

        Args:
            num_heads: The number of attention heads
            d_model: The dimension of the model (input and output of this layer will have shape [bs, L, d_model])

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

        self.Qs = nn.ModuleList(
            [nn.Linear(d_model, d_query, bias=False) for _ in range(num_heads)]
        )
        self.Ks = nn.ModuleList(
            [nn.Linear(d_model, d_query, bias=False) for _ in range(num_heads)]
        )
        self.Vs = nn.ModuleList(
            [nn.Linear(d_model, d_query, bias=False) for _ in range(num_heads)]
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
        """todo: docstring"""
        # Project inputs down to queries, keys, and values
        # todo: do this as matrix multiplication instead of list of nn.Linear modules
        queries = [Q(x_query) for Q in self.Qs]
        keys = [K(x_key) for K in self.Ks]
        values = [V(x_value) for V in self.Vs]

        # Apply attention
        attention_output = [
            scaled_dot_product_attention(q, k, v, mask)
            for q, k, v in zip(queries, keys, values)
        ]
        heads = [a[0] for a in attention_output]
        self.attentions = [a[1] for a in attention_output]

        # Concat heads together
        heads = torch.concat(heads, dim=-1)

        return self.Wout(heads)


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

        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)

        self.multi_head_attention_layernorm = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model),
        )

        self.feed_forward_layernorm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """todo: docstring"""
        res = x
        out = self.multi_head_attention(x, x, x, mask)
        out = self.dropout(out)
        out = out + res

        res = self.multi_head_attention_layernorm(out)
        out = self.feed_forward(res)
        out = self.dropout(out)
        out = out + res
        out = self.feed_forward_layernorm(out)

        return out


if __name__ == "__main__":
    num_heads = 8
    d_model = 256
    context_length = 10
    bs = 4

    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
    x = torch.randn(bs, context_length, d_model)
    mask = torch.tril(torch.ones(context_length, context_length))
    out = mha(x, x, x, mask)

    # Visualize the attention for each head for one element in the batch
    rows = 2
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(10, 20))
    for i in range(num_heads):
        ax = axs[i // cols, i % cols]
        ax.matshow(mha.attentions[i][2].detach())
        ax.set_title(f"self-attention head {i}")

    plt.show()

    d_ff = 2048
    enc_block = EncoderLayer(num_heads=num_heads, d_model=d_model, d_feedforward=d_ff)
    out = enc_block(x, mask)
