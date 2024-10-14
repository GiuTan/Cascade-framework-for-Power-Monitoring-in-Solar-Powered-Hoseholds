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
