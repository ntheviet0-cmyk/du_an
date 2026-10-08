"""AttritionGAT (Giai doan 3, co dinh — khong tuning):
Encoder MLP (Linear -> LayerNorm -> ReLU -> Dropout) -> 2 x GATv2Conv
(heads=4, attn dropout, edge_dim=1) voi residual -> concat [emb goc, emb graph]
-> head 2 lop -> logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class AttritionGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, heads: int = 4,
                 dropout: float = 0.3, attn_dropout: float = 0.3):
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.g1 = GATv2Conv(hidden, hidden, heads=heads, concat=False,
                            dropout=attn_dropout, edge_dim=1, add_self_loops=True)
        self.g2 = GATv2Conv(hidden, hidden, heads=heads, concat=False,
                            dropout=attn_dropout, edge_dim=1, add_self_loops=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        h0 = self.enc(x)
        h0 = self.ln(h0)
        h0 = F.relu(h0)
        h0 = F.dropout(h0, p=self.dropout, training=self.training)
        h = self.g1(h0, edge_index, edge_attr)
        h = F.elu(h) + h0
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.g2(h, edge_index, edge_attr)
        h = F.elu(h) + h
        out = torch.cat([h0, h], dim=-1)
        return self.head(out).squeeze(-1)
