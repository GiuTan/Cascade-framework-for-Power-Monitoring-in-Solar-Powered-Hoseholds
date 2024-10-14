import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 padding_mode='zeros',
                 last=False,
                 activation=nn.ReLU()):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(
            n_channels, n_kernels,
            kernel_size, stride, padding, padding_mode=padding_mode
        )
        if not last:
            self.net = nn.Sequential(
                self.conv,
                nn.BatchNorm1d(n_kernels),
                activation)
        else:
            self.net = self.conv
        nn.utils.weight_norm(self.conv)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.net(x)


class Deconv1D(nn.Module):
    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 last=False,
                 activation=nn.ReLU()):
        super(Deconv1D, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.deconv,
                nn.BatchNorm1d(n_kernels),
                activation
            )
        else:
            self.net = self.deconv
        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, x):
        return self.net(x)


class ResNetBlock(nn.Module):
    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 padding_mode='replicate'):
        super(ResNetBlock, self).__init__()
        self.net = nn.Sequential(
            Conv1D(n_channels, n_kernels, kernel_size, stride, padding, padding_mode, last=True),
            nn.ReLU(),
            Conv1D(n_kernels, n_kernels, kernel_size, stride, padding, padding_mode, last=True)
        )
        self.residual = Conv1D(n_channels, n_kernels, kernel_size=1, padding=0, padding_mode=padding_mode, last=True)


    def forward(self, x):
        return F.relu(self.net(x) + self.residual(x))


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


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TransUNetNILM(nn.Module):
    def __init__(
            self,
            in_size=1,
            seq_len=480,
            out_size=7,
            pred_len=480,
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
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads, dim_feedforward=self.hidden_size*4, dropout=self.dropout_rate, activation='gelu', batch_first=True, norm_first=True), num_layers=self.n_layers_transformer)

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

        # Transformer
        x = self.transformer(embedding)

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
