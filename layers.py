from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn

from ops import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        d_query_key: int = 64,
        d_value: int = 64,
        d_model: int = 512,
    ):
        """Multihead attention block. NOTE we require that num_heads * d_value == d_model

        Args:
            num_heads: The number of self attention heads
            d_query_key: The hidden dimension of the query and key matrices (both shape [L, d_value])
            d_value: The hidden dimension of the value matrix (shape [L, d_value])
            d_model: The hidden dimension of the model (output of each layer will be [bs, L, d_model])

        """
        # todo: just calculate d_query_key and d_value from num_heads and d_model
        super().__init__()

        if num_heads * d_value != d_model:
            raise ValueError(
                f"Received {num_heads=}, {d_value=}, {d_model=} that dont satisfy num_heads * d_value == d_model"
            )

        self.Qs = nn.ModuleList(
            [nn.Linear(d_model, d_query_key, bias=False) for _ in range(num_heads)]
        )
        self.Ks = nn.ModuleList(
            [nn.Linear(d_model, d_query_key, bias=False) for _ in range(num_heads)]
        )
        self.Vs = nn.ModuleList(
            [nn.Linear(d_model, d_value, bias=False) for _ in range(num_heads)]
        )

        self.Wout = nn.Linear(num_heads * d_value, d_model, bias=False)

        # We can just store the self attention values in the instance i guess if we want to look at them later
        self.attentions = None

    def forward(
        self,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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


if __name__ == "__main__":
    num_heads = 8
    mha = MultiHeadAttention(num_heads=num_heads)
    x = torch.randn(1, 10, 512)
    mask = torch.tril(torch.ones(10, 10))
    out = mha(x, x, x, mask)

    # Visualize the attention for each head
    rows = 2
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(10, 20))
    for i in range(num_heads):
        ax = axs[i // cols, i % cols]
        ax.matshow(mha.attentions[i].squeeze().detach())
        ax.set_title(f"self-attention head {i}")

    plt.show()
