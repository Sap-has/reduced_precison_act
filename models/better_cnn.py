import torch.nn as nn

class BetterCNN(nn.Module):
    def __init__(self, activation_fn, precision):
        super().__init__()
        self.precision = precision
        self.activation_fn = activation_fn
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # Always use ReLU in early layers to stabilize precision issues
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        
        x = self.apply_custom_activation(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def apply_custom_activation(self, x):
        """Applies the custom activation with precision to the entire feature map"""
        if self.activation_fn is not None:
            x = self.activation_fn(x, self.precision)
        return x
