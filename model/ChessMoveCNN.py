import torch.nn as nn
import torch

import sys
sys.path.insert(1, '../preprocessing/')
import utils

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

        # Input: (14, 8, 8) -> initial conv to expand channels
        in_channels = 14
        channels = [('bb', 64), ('bb', 64), ('bb', 128), ('se', 128), ('bb', 128), ('bb', 128), ('bb', 256), ('se', 256), ('bb', 256), ('bb', 256)]
        for type, out_channels in channels:
            if (type == 'bb'):
                layers.append(BasicBlock(in_channels, out_channels))
            elif (type == 'se'):
                layers.append(SEBlock(out_channels))
            in_channels = out_channels

        self.res_blocks = nn.Sequential(*layers)

        self.dropout = nn.Dropout(p=0.2)

        self.out_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

    def forward(self, x):
        x = self.res_blocks(x)      # (N, 128, 8, 8)
        x = self.dropout(x)
        x = self.out_head(x)        # (N, 2, 8, 8)
        return x
        
    def get_top_moves(self, board, n=5):
        x = utils.board_to_tensor(board)
        y = self(x)[0]
        pred = torch.sigmoid(y).detach().cpu().numpy()
        start_map, end_map = pred

        move_scores = []
        for move in board.legal_moves():
            score = start_map[move.from_square] * end_map[move.to_square]
            move_scores.append((move, score))
        
        move_scores.sort(lambda x: x[1], reverse=True)
        print(move_scores)
        return move_scores[:n]
