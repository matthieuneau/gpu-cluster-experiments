import torch
from torch import nn


class HeavyModel(nn.Module):
    def __init__(self):
        super(HeavyModel, self).__init__()
        self.flatten = nn.Flatten()

        # Large fully connected layers to bloat parameter count
        self.fc1 = nn.Linear(32 * 32 * 3, 8192)
        self.fc2 = nn.Linear(8192, 8192)
        self.fc3 = nn.Linear(8192, 4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
