import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, no_of_classes=1000, in_channels=3):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, no_of_classes),
        )
        self.init_parameter()

    def init_parameter(self):
        for layer in list(self.convs) + list(self.classifier):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        for idx in (4, 10, 12):
            nn.init.constant_(self.convs[idx].bias, 1)
        for idx in (2, 5):
            nn.init.constant_(self.classifier[idx].bias, 1)

    def forward(self, x):
        x = self.convs(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
