"""
model.py – 1-D CNN for FTIR spectral classification.

Architecture
------------
Three convolutional blocks (Conv1d → BatchNorm → ReLU → MaxPool → Dropout)
followed by a small fully-connected head.  Designed for input length ~3601.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, pool_size=4, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SpectraCNN(nn.Module):
    """
    Parameters
    ----------
    n_classes : int
        Number of plastic types to classify.
    seq_len : int
        Length of the input spectrum (default 3601 for the common grid).
    """

    def __init__(self, n_classes: int = 6, seq_len: int = 3601):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,  32, kernel_size=15, pool_size=4, dropout=0.15),
            ConvBlock(32, 64, kernel_size=7,  pool_size=4, dropout=0.20),
            ConvBlock(64, 128, kernel_size=5, pool_size=4, dropout=0.25),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SpectraResNet(nn.Module):
    """
    Slightly deeper residual variant for harder problems.
    Uses residual connections within each block.
    """

    class ResBlock(nn.Module):
        def __init__(self, channels, kernel_size=5, dropout=0.2):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=pad),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size, padding=pad),
                nn.BatchNorm1d(channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x + self.net(x))

    def __init__(self, n_classes: int = 6, seq_len: int = 3601):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

        self.body = nn.Sequential(
            self.ResBlock(64, dropout=0.15),
            nn.MaxPool1d(4),
            self.ResBlock(64, dropout=0.20),
            nn.MaxPool1d(4),
            self.ResBlock(64, dropout=0.25),
            nn.AdaptiveAvgPool1d(8),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.body(self.stem(x)))
