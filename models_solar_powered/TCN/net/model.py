import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            buffer=None,
            lookahead=0,
            **kwargs,
    ):

        super(CausalConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

        self.pad_len = (kernel_size - 1) * dilation
        if lookahead > self.pad_len // 2:
            lookahead = self.pad_len // 2
        self.lookahead = lookahead

        self.buffer_len = self.pad_len - self.lookahead
        # print( 'pad len:', self.pad_len )
        # print( 'lookahead:', self.lookahead )
        # print( 'buffer len:', self.buffer_len )

        if buffer is None:
            buffer = torch.zeros(
                1,
                in_channels,
                self.pad_len,
            )

        self.register_buffer(
            'buffer',
            buffer,
        )

        return

    def _forward(self, x):
        p = nn.ConstantPad1d(
            (self.buffer_len, self.lookahead),
            0.0,
        )
        x = p(x)
        x = super().forward(x)
        return x

    def forward(
            self,
            x,
            inference=False,
    ):
        if inference:
            x = self.inference(x)
        else:
            x = self._forward(x)
        return x

    def inference(self, x):
        if x.shape[0] != 1:
            raise ValueError(
                f"""
                Streaming inference of CausalConv1D layer only supports
                a batch size of 1, but batch size is {x.shape[0]}.
                """
            )
        if x.shape[2] < self.lookahead + 1:
            raise ValueError(
                f"""
                Input time dimension {x.shape[2]} is too short for causal
                inference with lookahead {self.lookahead}. You must pass at
                least lookhead + 1 time steps ({self.lookahead + 1}).
                """
            )
        x = torch.cat(
            (self.buffer, x),
            -1,
        )
        if self.lookahead > 0:
            self.buffer = x[:, :, -(self.pad_len + self.lookahead): -self.lookahead]
        else:
            self.buffer = x[:, :, -self.buffer_len:]
        x = super().forward(x)
        return x

    def reset_buffer(self):
        self.buffer.zero_()
        if self.buffer.shape[2] != self.pad_len:
            raise ValueError(
                f"""
                Buffer shape {self.buffer.shape} does not match the expected
                shape (1, {self.in_channels}, {self.pad_len}).
                """
            )
        return
class TCNBlock(nn.Module):
    def __init__(self, filters, kernel_size, dilation_rate, dropout):
        super(TCNBlock, self).__init__()
        #self.conv1 = weight_norm(nn.Conv1d(in_channels=filters, out_channels=filters,kernel_size= kernel_size, dilation=dilation_rate,stride=1,padding='same'))
        #self.conv2 = weight_norm(nn.Conv1d(in_channels=filters, out_channels=filters,kernel_size= kernel_size,  dilation=dilation_rate,stride=1,padding='same'))
        self.conv1 = weight_norm(CausalConv1d(in_channels=filters, out_channels=filters,kernel_size= kernel_size, dilation=dilation_rate))
        self.conv2 = weight_norm(CausalConv1d(in_channels=filters, out_channels=filters,kernel_size= kernel_size,  dilation=dilation_rate))


        self.dropout = nn.Dropout(dropout)
        self.residual_conv = nn.Conv1d(in_channels=filters,out_channels=1,kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        return x + residual


class TCN(nn.Module):
    def __init__(self, window_size, appliances, n_blocks=4, kernel_sizes=[3], dilation_rate=[2, 4, 8, 16], filters=[32],
                 dropout=0.1):
        super(TCN, self).__init__()
        #self.residual_conv2 = nn.Conv1d(1, 1, 1)
        self.initial_conv = nn.Conv1d(1, filters[0], kernel_sizes[0], padding='same')
        self.blocks = nn.ModuleList()
        for filter in filters:
            for kernel_size in kernel_sizes:
                for i in range(n_blocks):
                    self.blocks.append(TCNBlock(filter, kernel_size, dilation_rate[i], dropout))
        self.label_layer = nn.Linear(filters[0], 1024)
        self.output_layer = nn.Linear(1024, appliances)

    def forward(self, x):

        x = self.initial_conv(x.permute(0,2,1))
        for block in self.blocks:
            x = block(x)


        x = x.permute(0,2,1)
        x = self.label_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)

        return x




# # # Example usage
# window_size = 100
# appliances = 1
# model = TCN(window_size, appliances)
# print(model)

