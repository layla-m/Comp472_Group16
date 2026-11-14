"""Feed-forward network: [Dropout(in)] -> (Linear -> BatchNorm -> ReLU -> Dropout) x n -> Linear."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out, dropout, in_dropout=0.0):
        super().__init__()
        layers, d = ([nn.Dropout(in_dropout)] if in_dropout > 0 else []), n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))  # logits (the softmax is inside the loss)
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, n_in, n_out, cfg):
        return cls(n_in, cfg.hidden, n_out, cfg.dropout, cfg.in_dropout)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.softmax(model(torch.from_numpy(X)), 1).numpy()


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.from_numpy(X)).argmax(1).numpy()
