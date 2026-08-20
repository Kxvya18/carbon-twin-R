from __future__ import annotations

"""Optional temporal pretraining + fine-tuning.

This module is intentionally not used unless PyTorch is installed and an adaptation experiment is requested.
Tree ensembles are hyperparameter-tuned, not misleadingly called 'fine-tuned'. Fine-tuning is reserved for the
neural temporal model and is evaluated against the frozen pretrained model.
"""

import numpy as np


def _torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise RuntimeError("Install `pip install -e '.[advanced]'` to run temporal fine-tuning.") from e


def make_sequences(X, y, length=36):
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    xs, ys = [], []
    for i in range(length, len(X)):
        xs.append(X[i-length:i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


def build_gru(input_dim, hidden=48):
    torch, nn = _torch()
    class GRURegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        def forward(self, x):
            z, _ = self.gru(x)
            return self.head(z[:, -1]).squeeze(-1)
    return GRURegressor()


def train_gru(model, X, y, *, epochs=8, lr=1e-3, batch_size=128, seed=42):
    torch, nn = _torch()
    torch.manual_seed(seed)
    ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    model.train()
    losses = []
    for _ in range(epochs):
        running = 0.0
        for xb, yb in dl:
            opt.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb); loss.backward(); opt.step()
            running += float(loss.detach())*len(xb)
        losses.append(running/len(ds))
    return losses


def predict_gru(model, X):
    torch, _ = _torch()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()


def pretrain_then_finetune(X_pre, y_pre, X_adapt, y_adapt, X_eval, *, sequence_length=36,
                           pretrain_epochs=8, finetune_epochs=4, learning_rate=1e-3,
                           finetune_learning_rate=2e-4, seed=42):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    scaler = StandardScaler().fit(X_pre)
    Xp = scaler.transform(X_pre); Xa = scaler.transform(X_adapt); Xe = scaler.transform(X_eval)
    Xp_s, yp_s = make_sequences(Xp, y_pre, sequence_length)
    Xa_s, ya_s = make_sequences(Xa, y_adapt, sequence_length)
    Xe_s, ye_dummy = make_sequences(Xe, np.zeros(len(Xe)), sequence_length)
    model = build_gru(Xp.shape[1])
    pre_losses = train_gru(model, Xp_s, yp_s, epochs=pretrain_epochs, lr=learning_rate, seed=seed)
    before = predict_gru(model, Xe_s)
    # Fine-tune with lower learning rate on the most recent clean adaptation period.
    fine_losses = train_gru(model, Xa_s, ya_s, epochs=finetune_epochs, lr=finetune_learning_rate, seed=seed)
    after = predict_gru(model, Xe_s)
    return {
        "model": model,
        "scaler": scaler,
        "pretrain_losses": pre_losses,
        "finetune_losses": fine_losses,
        "pred_before": before,
        "pred_after": after,
        "sequence_offset": sequence_length,
    }
