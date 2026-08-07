"""Track B: 1D-CNN + Gaussian NLL head, trained as a Deep Ensemble for UQ.

Each ensemble member predicts its own (mu, sigma) instead of a point value.
Combining members via mixture-of-Gaussians moment matching (Lakshminarayanan
et al., 2017, "Simple and Scalable Predictive Uncertainty Estimation using
Deep Ensembles") gives an overall (mean, scale) pair with the same shape
NGBoost produces for Track A — so `turbofan_rul.calibration`'s conformal
calibration applies unchanged to either track.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def pick_device() -> str:
    """Return "cuda" only if the assigned GPU's compute capability is one the
    installed torch build actually shipped kernels for.

    Some cloud notebook environments (observed on Kaggle) hand out an older
    GPU (e.g. Pascal-era P100, sm_60) while the preinstalled torch wheel only
    ships kernels for Volta and newer (sm_70+) — `torch.cuda.is_available()`
    returns True in that case, but any real op fails with
    "CUDA error: no kernel image is available for execution on the device".
    Checking `get_arch_list()` against the actual device catches this before
    training even starts.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        cap = f"sm_{major}{minor}"
        supported = torch.cuda.get_arch_list()
        if cap not in supported:
            print(f"GPU capability {cap} not in torch's supported arch list {supported}; using CPU")
            return "cpu"
        return "cuda"
    except Exception as exc:  # pragma: no cover - defensive, hardware-dependent
        print("GPU capability check failed, falling back to CPU:", exc)
        return "cpu"


class RULCNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # (mu, log_var)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, window, features) -> (batch, features, window)
        x = self.conv(x)
        return self.head(x)


def train_member(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 32,
    device: str = "cpu",
    seed: int | None = None,
    verbose: bool = False,
    member_id: int = 0,
) -> RULCNN:
    if seed is not None:
        torch.manual_seed(seed)

    model = RULCNN(n_features=X.shape[-1], hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.GaussianNLLLoss(eps=1e-3)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        t0 = time.time()
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu_log_var = model(xb)
            mu, log_var = mu_log_var[:, 0], mu_log_var[:, 1]
            loss = loss_fn(mu, yb, torch.exp(log_var))
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss ({loss.item()}) at member {member_id}, epoch {epoch + 1} — "
                    "training diverged. Check that inputs are standardized (see "
                    "sequences.standardize_features) before blaming the model."
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * len(xb)
        if verbose:
            print(
                f"  member {member_id} epoch {epoch + 1}/{n_epochs} "
                f"loss={running_loss / len(dataset):.3f} ({time.time() - t0:.1f}s)"
            )

    model.eval()
    return model


@torch.no_grad()
def predict_member(
    model: RULCNN, X: np.ndarray, device: str = "cpu", batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, var) for one ensemble member."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    mus, variances = [], []
    for start in range(0, len(X_t), batch_size):
        batch = X_t[start : start + batch_size].to(device)
        out = model(batch)
        mus.append(out[:, 0].cpu().numpy())
        variances.append(torch.exp(out[:, 1]).cpu().numpy())
    return np.concatenate(mus), np.concatenate(variances)


def train_ensemble(
    X: np.ndarray, y: np.ndarray, n_members: int = 5, seed: int = 0, **train_kwargs
) -> list[RULCNN]:
    return [
        train_member(X, y, seed=seed + i, member_id=i, **train_kwargs) for i in range(n_members)
    ]


def predict_ensemble(
    models: list[RULCNN], X: np.ndarray, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Combine member (mu, var) predictions into overall (mean, scale) via
    mixture-of-Gaussians moment matching."""
    mus, variances = zip(*(predict_member(m, X, device=device) for m in models))
    mus = np.stack(mus)
    variances = np.stack(variances)

    mean = mus.mean(axis=0)
    var = (variances + mus**2).mean(axis=0) - mean**2
    scale = np.sqrt(np.maximum(var, 1e-12))
    return mean, scale
