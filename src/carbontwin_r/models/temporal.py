"""
Optional PyTorch temporal model.
Install requirements-advanced.txt before using this file.
"""

def build_tcn(input_channels=1, hidden=32, kernel_size=3):
    try:
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError("Install requirements-advanced.txt for the TCN.") from e

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_channels, hidden, kernel_size, padding=kernel_size-1, dilation=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size, padding=2*(kernel_size-1), dilation=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.net(x)[..., 0]
            return self.head(h).squeeze(-1)

    return TCN()
