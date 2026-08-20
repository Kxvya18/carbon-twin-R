"""
Optional spatial-then-temporal model skeleton.
"""

def build_temporal_graph_model(node_feature_dim, hidden=32):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
    except ImportError as e:
        raise RuntimeError("Install requirements-advanced.txt for Temporal-GNN.") from e

    class TemporalGraphNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn = GCNConv(node_feature_dim, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def spatial(self, x, edge_index):
            return F.relu(self.gcn(x, edge_index))

        def forward(self, sequence, edge_index):
            # sequence: [T, N, F] for a single example
            spatial = []
            for t in range(sequence.shape[0]):
                spatial.append(self.spatial(sequence[t], edge_index))
            h = torch.stack(spatial, dim=0).transpose(0,1)  # [N,T,H]
            z, _ = self.gru(h)
            return self.head(z[:,-1,:]).squeeze(-1)

    return TemporalGraphNet()
