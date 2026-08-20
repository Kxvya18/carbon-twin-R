"""
Optional PyTorch Geometric GNN.
"""

def build_gnn(in_channels, hidden=32, out_channels=1):
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
    except ImportError as e:
        raise RuntimeError("Install requirements-advanced.txt for the GNN.") from e

    class FacilityGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GCNConv(in_channels, hidden)
            self.c2 = GCNConv(hidden, hidden)
            self.out = nn.Linear(hidden, out_channels)

        def forward(self, x, edge_index):
            x = F.relu(self.c1(x, edge_index))
            x = F.relu(self.c2(x, edge_index))
            return self.out(x)

    return FacilityGCN()
