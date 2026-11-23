"""PyTorch architecture matching the bundled model state dictionary."""

import torch
from torch import nn


class SQIGatedMultimodalNet(nn.Module):
    def __init__(self, seq_channels=6, feat_dim=18, num_classes=3):
        super().__init__()
        self.branch_fast = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.branch_slow = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.sqi_gate = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 128), nn.Sigmoid()
        )
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 + 64, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, seq_x, feat_x):
        temp_emb = torch.cat([self.branch_fast(seq_x), self.branch_slow(seq_x)], dim=1)
        motion_ctx = feat_x[:, 12:16]
        gate = self.sqi_gate(motion_ctx)
        gated_temp = temp_emb * gate
        dom_emb = self.feat_net(feat_x)
        return self.classifier(torch.cat([gated_temp, dom_emb], dim=1))
