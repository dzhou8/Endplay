import torch.nn as nn

class BasicBlock(nn.Module):
    """
    A small residual block:
    1) Conv -> BN -> ReLU
    2) Conv -> BN
    Then add skip connection and final ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in_channels != out_channels, use a 1x1 conv to match dimensions for the skip.
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Possibly transform the skip
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)

        # Add skip connection
        out += identity
        out = self.relu(out)
        return out

class ChessMoveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial block (maps 12 channels -> 64)
        self.block1 = BasicBlock(12, 64)
        # Second block (64 -> 64)
        self.block2 = BasicBlock(64, 64)
        # Third block (64 -> 128)
        self.block3 = BasicBlock(64, 128)
        # Fourth block (128 -> 128)
        self.block4 = BasicBlock(128, 128)

        # Final 1×1 conv to reduce 128 channels to 2 (start-square map and end-square map)
        self.out_conv = nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)     # (N, 64, 8, 8)
        x = self.block2(x)     # (N, 64, 8, 8)
        x = self.block3(x)     # (N, 128, 8, 8)
        x = self.block4(x)     # (N, 128, 8, 8)
        x = self.out_conv(x)   # (N, 2, 8, 8)
        return x