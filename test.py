from transformer.ops import scaled_dot_product_attention
from transformer.layers import MultiHeadAttention, EncoderLayer, DecoderLayer, PositionalEncoding
from transformer.models import Encoder, Decoder, EncoderOnlyClassifier
import matplotlib.pyplot as plt
import torch


if __name__ == "__main__":
    q = torch.randn(10, 64)
    k = torch.randn(10, 64)
    v = torch.randn(10, 64)
    mask = torch.tril(torch.ones(10, 10))

    out, att = scaled_dot_product_attention(q, k, v, mask=mask)

    print(out.shape)
    plt.matshow(att)

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

    # Smoke test to make sure encoder can do forward pass
    enc = Encoder(d_model=d_model, seq_len=context_length)
    x = torch.randn(bs, context_length, d_model)
    out = enc(x)

    # Smoke test for decoder
    dec = Decoder(d_model=d_model, seq_len=context_length)
    out = dec(x, out, target_mask=dec.subsequent_mask)

    # Smoke test for encoder-only classifier
    x = torch.randint(50256, (4, 160))
    mask = torch.tril(torch.ones(160, 160))
    model = EncoderOnlyClassifier(
        num_classes=2, num_encoders=3, num_heads=4, d_model=256, seq_len=160
    )
    out = model(x, mask)
    plt.matshow(
        model.encoder.encoder_layers[0]
        .multihead_self_attention.attentions[0][0]
        .detach()
    )
    plt.show()
