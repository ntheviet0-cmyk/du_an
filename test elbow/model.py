"""Independent GraphSAGE candidate; no production artifacts are changed."""
import torch
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.c1 = SAGEConv(in_dim, h1, aggr="mean")
        self.c2 = SAGEConv(h1, h2, aggr="mean")
        self.lin = torch.nn.Linear(h2, 1)
        self.dropout = dropout

    def forward(self, x, edges):
        x = F.relu(self.c1(x, edges))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(F.relu(self.c2(x, edges))).squeeze(-1)
