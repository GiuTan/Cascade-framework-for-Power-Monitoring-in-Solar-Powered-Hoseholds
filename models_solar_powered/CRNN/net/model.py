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
    def __init__(self, classes=7, drop_out=0.1, kernel=5, n_conv_blocks=3, gru_units=64):
        super(CRNN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 1
        for i in range(n_conv_blocks):
            filters = 2 ** (i + 5)
            self.layers.append(CRNN_block(in_channels, kernel=kernel, drop_out=drop_out, filters=filters))
            in_channels = filters


        self.bi_direct = nn.GRU(filters, gru_units, bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(gru_units * 2, 1024)
        self.relu = nn.ReLU()
        self.frame_level = nn.Linear(1024, classes)



    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)

        x = torch.permute(x, (0, 2, 1))
        x, _ = self.bi_direct(x)
        x = self.dense1(x)
        x = self.relu(x)
        frame_level_output = self.frame_level(x)
        #print(frame_level_output.shape)

        return frame_level_output

