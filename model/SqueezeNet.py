import torch.nn as nn
import torch
import torch.nn.functional as F

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
import utils

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([
            self.expand1x1(x), self.expand3x3(x)
        ], dim=1))

class ChessMoveCNN(nn.Module):
    def __init__(self):
        super(ChessMoveCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(14, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Fire(32, 16, 32),   # Output: 64 channels
            Fire(64, 16, 32),   # Output: 64 channels
            Fire(64, 32, 48),   # Output: 96 channels
            nn.Dropout(p=0.2)
        )

        self.out_head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)  # Predicts (2, 8, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.out_head(x)
        return x
        
    def get_top_moves(self, board, n=5):
        x = utils.board_to_tensor(board)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = self(x)[0]
            pred = torch.sigmoid(y).cpu().numpy()
        start_map, end_map = pred

        move_scores = []
        for move in board.legal_moves:
            score = start_map[move.from_square // 8][move.from_square % 8] * end_map[move.to_square // 8][move.to_square % 8]
            move_scores.append((move, score))
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[:n]
    
    def get_move_score(self, board, move):
        x = utils.board_to_tensor(board)
        x = torch.from_numpy(x).float().unsqueeze(0)

        with torch.no_grad():
            y = self(x)[0]
            pred = torch.sigmoid(y)
        
        start_map, end_map = pred[0], pred[1]

        fr, fc = divmod(move.from_square, 8)
        tr, tc = divmod(move.to_square, 8)
        score = start_map[fr, fc] * end_map[tr, tc]
        return score.item()
