import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv1D, Deconv1D, ResNetBlock


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = Deconv1D(in_ch, in_ch // 2, kernel_size=4, stride=2, padding=1)
        self.conv = Conv1D(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # Pad x1 to the size of x2
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff// 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransUNetNILM(nn.Module):
    def __init__(
            self,
            in_size,
            seq_len,
            out_size,
            pred_len,
            dropout=0.1,
            n_layers_unet=3,
            n_layers_transformer=4,
            n_heads=4,
            features_start=64,
            d_ff=128
    ):
        super().__init__()

        self.conv = Conv1D(n_channels=in_size, n_kernels=features_start, kernel_size=3, stride=1, padding=1, padding_mode='zeros', last=True)
        layers = []
        feats = features_start
        for _ in range(n_layers_unet):
            layers.append(ResNetBlock(feats, feats*2, kernel_size=5, stride=1, padding=2, padding_mode='replicate'))
            feats *= 2

        self.upsample = Deconv1D(n_channels=feats, n_kernels=feats, kernel_size=4, stride=2, padding=1, last=True)

        for _ in range(n_layers_unet - 1):
            layers.append(Up(feats, feats//2))
            feats //= 2

        self.in_size = in_size
        self.seq_len = seq_len
        self.out_size = out_size
        self.pred_len = pred_len
        self.dropout_rate = dropout
        self.n_layers_unet = n_layers_unet
        self.n_layers_transformer = n_layers_transformer
        self.n_heads = n_heads
        self.hidden_size = features_start*(2**n_layers_unet)
        self.d_ff = d_ff

        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        self.position = PositionalEmbedding(max_len=seq_len//2, d_model=self.hidden_size)
        self.layer_norm = LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        transformer_blocks = [TransformerBlock(self.hidden_size, self.n_heads, self.hidden_size * 4, self.dropout_rate) for _ in range(self.n_layers_transformer)]
        self.transformer = nn.Sequential(*transformer_blocks)

        conv = Conv1D(n_channels=feats, n_kernels=feats, kernel_size=1, stride=1, padding=0, padding_mode='zeros', last=True)
        layers.append(conv)
        self.layers = nn.ModuleList(layers)

        self.deconv = Deconv1D(n_channels=features_start*2, n_kernels=self.hidden_size, kernel_size=4, stride=2, padding=1, last=True)
        self.linear1 = nn.Linear(self.hidden_size, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.out_size)

    def forward(self, sequence):
        # Conv 3x3
        x = self.conv(sequence.permute(0, 2, 1))

        # ResUNet encoder
        xi = [self.layers[0](x)]
        for layer in self.layers[1:self.n_layers_unet]:
            xi.append(layer(xi[-1]))

        # Positional embedding
        x_token = self.pool(xi[-1]).permute(0, 2, 1)
        embedding = x_token + self.position(x)
        x = self.dropout(self.layer_norm(embedding))

        # Transformer
        x = self.transformer(x)

        # UNet decoder
        x = x.permute(0, 2, 1)
        xi[-1] = self.upsample(x)
        for i, layer in enumerate(self.layers[self.n_layers_unet:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        x = self.layers[-1](xi[-1])

        # Prediction
        x = self.deconv(x).permute(0, 2, 1)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        x = x[:, -self.pred_len:, :]

        return x
