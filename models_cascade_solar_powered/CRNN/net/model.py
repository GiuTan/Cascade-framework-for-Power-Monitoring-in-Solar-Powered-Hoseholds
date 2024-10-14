import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CRNN_block(nn.Module):
    def __init__(self, in_channels, kernel, drop_out, filters):
        super(CRNN_block, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, filters, kernel, stride=1, padding='same')
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        self.bn1 = nn.BatchNorm1d(filters)
        self.pool1 = nn.MaxPool1d(1)
        self.drop1 = nn.Dropout(drop_out)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        return x

class CRNN(nn.Module):
    def __init__(self, classes=6, drop_out=0.1, kernel=5, n_conv_blocks=3, gru_units=64):
        super(CRNN, self).__init__()
        self.layers_NILM = nn.ModuleList()
        self.layers_SOLAR = nn.ModuleList()

        in_channels = 1
        n_conv_blocks_S = 3
        for i in range(n_conv_blocks_S):
            filters = 2 ** (i + 5)
            self.layers_SOLAR.append(CRNN_block(in_channels, kernel=kernel, drop_out=drop_out, filters=filters))
            in_channels = filters

        self.bi_direct_S = nn.GRU(filters, gru_units, bidirectional=True, batch_first=True)
        self.dense1_S = nn.Linear(gru_units * 2, 1024)


        in_channels = 1
        for i in range(n_conv_blocks):
            filters = 2 ** (i + 5)
            self.layers_NILM.append(CRNN_block(in_channels, kernel=kernel, drop_out=drop_out, filters=filters))
            in_channels = filters
        self.bi_direct_N = nn.GRU(filters, gru_units, bidirectional=True, batch_first=True)


        self.dense1_N = nn.Linear(gru_units * 2, 1024)

        self.relu = nn.ReLU()
        self.solar_output = nn.Linear(1024, 1)
        self.nilm_output = nn.Linear(1024, classes)
        #self.add1 = torch.add()

    def forward(self, x_in):
        x_s = x_in.permute(0, 2, 1)
        for layer in self.layers_SOLAR:
            x_s = layer(x_s)

        x_s = torch.permute(x_s, (0, 2, 1))
        x_s, _ = self.bi_direct_S(x_s)
        x_s = self.dense1_S(x_s)
        x_s = self.relu(x_s)
        solar_output = self.solar_output(x_s)

        x_sum = x_in.permute(0, 2, 1).add(solar_output.permute(0, 2, 1))
        x = x_sum
        for layer in self.layers_NILM:
            x = layer(x)

        x = torch.permute(x, (0, 2, 1))
        x, _ = self.bi_direct_N(x)
        x = self.dense1_N(x)
        x = self.relu(x)
        nilm_output = self.nilm_output(x)
        
        return solar_output, nilm_output,  x_sum.permute(0, 2, 1)

