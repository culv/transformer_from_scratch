import math
from typing import Optional

import torch
from torch.nn import functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute scaled dot product attention for the given key/query/value.

    Args:
        query (torch.Tensor): Tensor with shape [..., L, d_query] where L is the length of the sequence
        key (torch.Tensor): Tensor with shape [..., L, d_key]
        value (torch.Tensor): Tensor with shape [..., L, d_value]
        mask (torch.Tensor): Optional mask to apply to attention scores (with shape [..., L, L])

    Returns:
        output: The final result of scaled dot product attention (with shape [..., L, d_value])
        attention: A [..., L, L] tensor representing the attention
    """
    d_k = key.shape[-1]
    attention = torch.matmul(query, key.transpose(-1, -2))
    attention /= math.sqrt(d_k)

    if mask is not None:
        # The mask must be broadcastable with attention, so we may have to add a singleton dimension
        # Usually this is the case for multihead attention where attention is [bs, num_heads, L, L]
        # but the mask is [bs, L, L]
        if len(mask.shape) != len(attention.shape):
            mask = mask.unsqueeze(1)
        attention = attention.masked_fill(mask == 0, -1e9)

    attention = F.softmax(attention, dim=-1)
    output = torch.matmul(attention, value)

    return output, attention


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    q = torch.randn(10, 64)
    k = torch.randn(10, 64)
    v = torch.randn(10, 64)
    mask = torch.tril(torch.ones(10, 10))

    out, att = scaled_dot_product_attention(q, k, v, mask=mask)

    print(out.shape)
    plt.matshow(att)
    plt.show()
