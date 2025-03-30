import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Optional downsampling for skip connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # squeeze
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class ChessMoveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []

        # Input: (12, 8, 8) -> initial conv to expand channels
        in_channels = 12
        channels = [('bb', 64), ('bb', 64), ('bb', 128), ('se', 128), ('bb', 128), ('bb', 128), ('bb', 128), ('se', 128)]
        for type, out_channels in channels:
            if (type == 'bb'):
                layers.append(BasicBlock(in_channels, out_channels))
            elif (type == 'se'):
                layers.append(SEBlock(in_channels))
            in_channels = out_channels

        self.res_blocks = nn.Sequential(*layers)

        # Final prediction: 128 channels -> 2 heatmaps (start + end square)
        self.out_conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x):
        x = self.res_blocks(x)      # (N, 128, 8, 8)
        x = self.out_conv(x)        # (N, 2, 8, 8)
        return x