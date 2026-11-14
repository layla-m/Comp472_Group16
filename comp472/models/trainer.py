"""
Training loop of the MLP.

With a validation set: early stopping on validation accuracy, the best epoch is restored.
Without: trains for exactly `epochs` epochs (final models on 100 % of the data).
"""
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import SEED
from .mlp import MLP, predict


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def update_bn(model, X, batch_size=512):
    """Recompute the BatchNorm statistics of an averaged (SWA) model on the training data."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average
    model.train()
    for i in range(0, len(X), batch_size):
        model(X[i:i + batch_size])


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    best_acc: float = None                         # None when trained without validation
    history: list = field(default_factory=list)   # (epoch, train_loss, train_acc, val_acc)


class MLPTrainer:
    """Trains one MLP for an `MLPConfig`.

    record_history: also compute the training accuracy at every epoch (for learning curves).
    It only adds evaluation passes and does not change the trained weights.
    """

    def __init__(self, cfg, seed=SEED, record_history=False, verbose=False):
        self.cfg, self.seed = cfg, seed
        self.record_history, self.verbose = record_history, verbose

    def fit(self, X_tr, y_tr, n_classes, X_va=None, y_va=None, epochs=None, counts=None, tprep=None):
        """counts / tprep: raw visual-word counts of X_tr and the torch transform, needed by the
        `thin` augmentation (each patch kept with probability 1 - thin, then re-normalised)."""
        cfg = self.cfg
        set_seed(self.seed)
        model = MLP.from_config(X_tr.shape[1], n_classes, cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.smoothing)
        thin, noise, swa_start = cfg.thin, cfg.noise, cfg.swa_start
        inputs = torch.from_numpy(counts) if thin > 0 else torch.from_numpy(X_tr)
        loader = DataLoader(TensorDataset(inputs, torch.from_numpy(y_tr)),
                            batch_size=cfg.batch_size, shuffle=True)
        X_tr_t = torch.from_numpy(X_tr)
        swa = torch.optim.swa_utils.AveragedModel(model) if swa_start else None
        n_epochs = epochs or cfg.max_epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        beta = torch.distributions.Beta(cfg.mixup, cfg.mixup) if cfg.mixup > 0 else None

        best_acc, best_epoch, best_state, wait, history = -1, 0, None, 0, []
        for ep in range(1, n_epochs + 1):
            model.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                if thin > 0:
                    xb = tprep(torch.binomial(xb, torch.full_like(xb, 1 - thin)))
                if noise > 0:
                    xb = xb + noise * torch.randn_like(xb)
                if beta is not None:
                    lam = beta.sample().item()
                    perm = torch.randperm(len(xb))
                    out = model(lam * xb + (1 - lam) * xb[perm])
                    loss = lam * loss_fn(out, yb) + (1 - lam) * loss_fn(out, yb[perm])
                else:
                    loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            sched.step()
            # SWA: average the weights of every epoch from swa_start on
            use_swa = swa is not None and ep >= swa_start
            if use_swa:
                swa.update_parameters(model)
                if X_va is not None or ep == n_epochs:
                    update_bn(swa, X_tr_t)
            if X_va is None:
                continue
            net = swa if use_swa else model
            va_acc = (predict(net, X_va) == y_va).mean()
            if self.record_history:
                tr_acc = (predict(net, X_tr) == y_tr).mean()
                history.append((ep, total / len(X_tr), tr_acc, va_acc))
            if self.verbose:
                print(f"  epoch {ep:3d} val_acc={va_acc:.4f}", flush=True)
            if va_acc > best_acc:
                best_acc, best_epoch, wait = va_acc, ep, 0
                best_state = (use_swa, {k: v.clone() for k, v in net.state_dict().items()})
            else:
                wait += 1
                if wait >= cfg.patience:
                    break

        if X_va is None:
            final = swa if swa is not None and n_epochs >= swa_start else model
            return TrainResult(final, n_epochs)
        if best_state is not None:
            was_swa, state = best_state
            net = swa if was_swa else model
            net.load_state_dict(state)
            return TrainResult(net, best_epoch, best_acc, history)
        return TrainResult(model, best_epoch, best_acc, history)
