import torch
import torch.nn as nn

class Seq2PointModel(nn.Module):
    def __init__(self, input_window_length):
        super(Seq2PointModel, self).__init__()
        #self.reshape = nn.Unflatten(1, (1, input_window_length, 1))
        self.conv1 = nn.Conv1d(1, 30, kernel_size=10, stride=1, padding='same')
        self.conv2 = nn.Conv1d(30, 30, kernel_size=8, stride=1, padding='same')
        self.conv3 = nn.Conv1d(30, 40, kernel_size=6, stride=1, padding='same')
        self.conv4 = nn.Conv1d(40, 50, kernel_size=5, stride=1, padding='same')
        self.conv5 = nn.Conv1d(50, 50, kernel_size=5, stride=1, padding='same')
        #self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(50, 1024)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1024, 7)

    def forward(self, x):
        #x = self.reshape(x)
        #print(x.shape)
        x = self.relu(self.conv1(x.permute(0,2,1)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        #x = self.flatten(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.fc1(x))
        x = self.linear(x)
        return x

model = Seq2PointModel(input_window_length=480)
print(model)