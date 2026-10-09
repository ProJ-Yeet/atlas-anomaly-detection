"""Shared PyTorch helpers for the deep models."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def loader_from(windows, batch_size=128, shuffle=True):
    ds = TensorDataset(torch.tensor(windows, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def fit(model, loader, loss_fn, epochs=15, lr=1e-3, tag=""):
    """Generic training loop; loss_fn(model, batch) -> scalar loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs + 1):
        total, count = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            loss = loss_fn(model, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            count += len(xb)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  {tag} epoch {ep:3d}/{epochs}  loss {total / count:.5f}")
    model.eval()
    return model


@torch.no_grad()
def batched_apply(fn, windows, batch_size=512):
    """Apply fn to windows in batches, concatenate 1-D numpy outputs."""
    out = []
    for i in range(0, len(windows), batch_size):
        xb = torch.tensor(windows[i:i + batch_size], dtype=torch.float32).to(DEVICE)
        out.append(fn(xb).cpu().numpy())
    return np.concatenate(out)
