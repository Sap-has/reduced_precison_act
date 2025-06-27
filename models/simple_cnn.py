import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, activation_fn, precision):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.act_fn = lambda x: activation_fn(x, precision)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(self.act_fn(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x