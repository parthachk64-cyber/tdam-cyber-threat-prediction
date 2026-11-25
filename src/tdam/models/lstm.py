"""Unidirectional LSTM reference model (companion default, not the paper's verified architecture).

Tensor shapes
    x        [B, T, F]   float32, oldest bin first, newest at index T-1 (right aligned)
    lengths  [B]         valid bins per row (< T only when the stream started inside the window)
    logit    [B]         one binary logit: P(an annotated onset lies in (t, t+H])
Padding rule: the leading T - length positions are padding. They are shifted behind the data and
the sequence is packed, so the final valid hidden state is the state after the newest bin and padding
cannot influence it.
Objective: binary cross-entropy with logits, positive-class weight n_neg / n_pos fitted on the
training windows only. Optimiser Adam; early stopping on validation average precision
(validation loss when the validation slice has a single class).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def seed_everything(seed: int, threads: int | None = None) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if threads:
        torch.set_num_threads(threads)


class SeqLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        shift = (t - lengths).unsqueeze(1)  # leading pad count
        idx = (torch.arange(t, device=x.device).unsqueeze(0) + shift).clamp(max=t - 1)
        left = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        packed = pack_padded_sequence(left, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)


class LSTMClassifier:
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1):
        self.n_features, self.hidden, self.layers = n_features, hidden, layers
        self.net = SeqLSTM(n_features, hidden, layers)

    # -- inference ---------------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, Z: np.ndarray, lengths: np.ndarray, batch: int = 2048) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(Z), batch):
            x = torch.from_numpy(Z[i : i + batch])
            ln = torch.from_numpy(lengths[i : i + batch])
            out.append(torch.sigmoid(self.net(x, ln)).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    # -- training ----------------------------------------------------------------------------
    def fit(
        self,
        Z: np.ndarray,
        lengths: np.ndarray,
        y: np.ndarray,
        Zv: np.ndarray | None = None,
        lv: np.ndarray | None = None,
        yv: np.ndarray | None = None,
        epochs: int = 12,
        patience: int = 3,
        lr: float = 1e-3,
        batch: int = 256,
        seed: int = 0,
        pos_weight: float | None = None,
    ) -> dict:
        gen = torch.Generator().manual_seed(seed)
        n_pos = max(int(y.sum()), 1)
        pw = float(pos_weight) if pos_weight is not None else (len(y) - n_pos) / n_pos
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(pw, 1.0)))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        xt, lt, yt = torch.from_numpy(Z), torch.from_numpy(lengths), torch.from_numpy(y.astype(np.float32))
        best, best_state, bad, history = -np.inf, None, 0, []
        for _epoch in range(epochs):
            self.net.train()
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(perm), batch):
                sel = perm[i : i + batch]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[sel], lt[sel]), yt[sel])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            if Zv is None:
                continue
            score = self._val_score(Zv, lv, yv, loss_fn)
            history.append(score)
            if score > best + 1e-6:
                best, best_state, bad = score, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {"val_score_history": history, "epochs_run": len(history) or epochs}

    def _val_score(self, Zv, lv, yv, loss_fn) -> float:
        p = self.predict_proba(Zv, lv)
        if yv.min() != yv.max():
            return float(average_precision_score(yv, p))
        eps = 1e-7
        return float(-np.mean(-(yv * np.log(p + eps) + (1 - yv) * np.log(1 - p + eps))))

    def state(self) -> dict:
        return {k: v.cpu() for k, v in self.net.state_dict().items()}

    def load_state(self, state: dict) -> None:
        self.net.load_state_dict(state)
