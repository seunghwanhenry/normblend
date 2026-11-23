"""
dtin_v2.py - standalone implementation of the proposed method (final configuration).

A masked Transformer for incomplete multivariate time series, used for
  (1) missing-value imputation and
  (2) self-supervised pre-training of an encoder for downstream classification.

Configuration selected on development data (phase 1-2; ECG200, Siemens, DodgerLoopGame):

  Encoder      input = [visible values, observation mask] -> linear projection -> sinusoidal positional
               encoding -> 3 Transformer encoder layers (d_model 64, 4 heads, FFN 128, dropout 0.1)
               -> linear output layer. Visible values are normalised per sequence and variable
               (RevIN-style) inside the model; outputs are returned in the input scale.
  Imputation   mixed artificial masking (per batch: 50 % -> 15 % random points; 50 % -> random spans of
               length 1-20 with masking ratio drawn from {0.1, ..., 0.5}); MSE on the artificially masked
               observed values only (pseudo_weight = 0); Adam 1e-3, batch 64, 500 warm-up steps + cosine
               decay, at most 20,000 steps; early stopping on validation imputation RMSE (every 250 steps,
               patience 12).
  Pre-training same encoder; the loss also includes local Gaussian pseudo-targets for inherently missing
  (downstream) values (window 20, pseudo_weight = 1); 2,000 steps; then fine-tuning of encoder + linear
               head on mean-pooled outputs (Adam 1e-3, batch 32, CE), number of epochs chosen by
               stratified cross-validation on the training set.

Data format: numpy arrays (N, T, D), NaN at missing positions. Standardise every variable with
training-set statistics before use (see Standardizer).

Requires: numpy, torch (scikit-learn only for select_epochs_cv).
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================================== config
@dataclass
class Config:
    # architecture
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.1
    mask_input: bool = True                # observation mask as extra input channels (ablation switch)
    revin: bool = True                     # per-sequence normalisation of visible values (ablation switch)
    # artificial masking
    mask_mode: str = "mix"                 # "point" | "span" | "mix"
    point_ratio: float = 0.15
    span_max: int = 20
    span_ratios: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    # pseudo-targets for inherently missing values (0 for imputation, 1 for downstream pre-training)
    pseudo_weight: float = 0.0
    window: int = 20
    # optimisation
    lr: float = 1e-3
    batch_size: int = 64
    warmup: int = 500
    max_steps: int = 20_000
    eval_every: int = 250
    patience: int = 12
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def imputation_config(**kw) -> Config:
    """Final configuration for imputation."""
    return Config(**kw)


def pretraining_config(**kw) -> Config:
    """Final configuration for downstream pre-training (pseudo-targets on, 2,000 steps)."""
    return Config(**dict(dict(pseudo_weight=1.0, max_steps=2_000), **kw))


# =============================================================================================== utilities
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse_at(X_true: np.ndarray, X_imp: np.ndarray, M: np.ndarray) -> float:
    """RMSE at the positions where M is True."""
    return float(np.sqrt(np.mean((X_true[M] - X_imp[M]) ** 2)))


class Standardizer:
    """Per-variable standardisation with training-set statistics (NaN-aware)."""

    def fit(self, X: np.ndarray):
        self.mean_ = np.nanmean(X, axis=(0, 1))
        sd = np.nanstd(X, axis=(0, 1))
        self.std_ = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X): return (X - self.mean_) / self.std_
    def inverse_transform(self, X): return X * self.std_ + self.mean_


def _tensors(X_nan: np.ndarray, device):
    x = torch.tensor(np.nan_to_num(X_nan), dtype=torch.float32, device=device)
    o = torch.tensor(~np.isnan(X_nan), dtype=torch.float32, device=device)
    return x, o


# =============================================================================================== model
class DTINv2Net(nn.Module):
    def __init__(self, n_features: int, cfg: Config, max_len: int = 4096):
        super().__init__()
        dm = cfg.d_model
        self.mask_input, self.revin = cfg.mask_input, cfg.revin
        self.inp = nn.Linear((2 if cfg.mask_input else 1) * n_features, dm)   # [values, observation mask]
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dm, 2).float() * (-math.log(10000.0) / dm))
        pe = torch.zeros(max_len, dm); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(dm, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.pred = nn.Linear(dm, n_features)

    def _prepare(self, x, v):
        """Visible values (normalised if revin) concatenated with the mask (if mask_input)."""
        x = x * v
        if self.revin:
            x, mu, sd = self._normalise(x, v)
        else:
            mu, sd = None, None
        return (torch.cat([x, v], -1) if self.mask_input else x), mu, sd

    @staticmethod
    def _normalise(x, v):
        """Per-sequence, per-variable normalisation of the visible values (v = 1)."""
        cnt = v.sum(1, keepdim=True)
        mu = (x * v).sum(1, keepdim=True) / cnt.clamp(min=1)
        sd = torch.sqrt((((x - mu) * v) ** 2).sum(1, keepdim=True) / cnt.clamp(min=1) + 1e-5)
        mu = torch.where(cnt > 0, mu, torch.zeros_like(mu))
        sd = torch.where(cnt > 1, sd, torch.ones_like(sd))
        return (x - mu) / sd * v, mu, sd

    def encode(self, x, v):
        """Hidden states (B, T, d_model). x: values (anything at invisible positions), v: 1 = visible."""
        h, _, _ = self._prepare(x, v)
        return self.enc(self.inp(h) + self.pe[: x.shape[1]])

    def forward(self, x, v):
        """Reconstruction of every position, in the input scale."""
        h, mu, sd = self._prepare(x, v)
        y = self.pred(self.enc(self.inp(h) + self.pe[: x.shape[1]]))
        return y * sd + mu if self.revin else y


# =============================================================================================== masking / targets
def point_mask(o, ratio, gen):
    return (torch.rand(o.shape, generator=gen, device=o.device) < ratio).float() * o


def span_mask(o, ratio, gen, max_len=20):
    """Span starts ~ Bernoulli(ratio / mean span length), span length ~ Uniform{1..max_len}."""
    B, T, D = o.shape
    p = ratio / ((1 + max_len) / 2)
    starts = torch.rand(o.shape, generator=gen, device=o.device) < p
    lens = torch.randint(1, max_len + 1, o.shape, generator=gen, device=o.device)
    t = torch.arange(T, device=o.device).view(1, T, 1)
    reach = torch.where(starts, t + lens, torch.full_like(lens, -1))
    return (torch.cummax(reach, dim=1).values > t).float() * o


def artificial_mask(o, cfg: Config, gen):
    mode = cfg.mask_mode
    if mode == "mix":
        mode = "point" if torch.rand(1, generator=gen, device=o.device).item() < 0.5 else "span"
    if mode == "point":
        return point_mask(o, cfg.point_ratio, gen)
    r = cfg.span_ratios[int(torch.randint(len(cfg.span_ratios), (1,), generator=gen, device=o.device).item())]
    return span_mask(o, r, gen, cfg.span_max)


def local_gaussian_stats(x, valid, w, mu_tr, var_tr):
    """Local mean / unbiased variance of the valid values in a window of length w centred at t.
    Fallbacks: no valid value -> training mean and variance; one valid value -> training variance."""
    T = x.shape[1]
    xd, vd = x.double(), valid.double(); xv = xd * vd
    pad = lambda c: F.pad(c, (0, 0, 1, 0))
    c1, c2, cn = pad(xv.cumsum(1)), pad((xv * xv).cumsum(1)), pad(vd.cumsum(1))
    t = torch.arange(T, device=x.device)
    s = (t - w // 2).clamp(0, max(T - w, 0)); e = (s + w).clamp(max=T)
    S1, S2, N = c1[:, e] - c1[:, s], c2[:, e] - c2[:, s], cn[:, e] - cn[:, s]
    mean = S1 / N.clamp(min=1)
    var = ((S2 - N * mean ** 2) / (N - 1).clamp(min=1)).clamp(min=0)
    mean = torch.where(N == 0, mu_tr.double().expand_as(mean), mean)
    var = torch.where(N <= 1, var_tr.double().expand_as(var), var)
    return mean.float(), var.float()


# =============================================================================================== training
@torch.no_grad()
def _reconstruct(model, X_nan, device, batch_size=256):
    model.eval(); out = []
    for b in range(0, len(X_nan), batch_size):
        x, o = _tensors(X_nan[b:b + batch_size], device)
        out.append(model(x, o).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train_dtin(X_nan: np.ndarray, cfg: Config, seed: int = 0,
               X_val_true: Optional[np.ndarray] = None, M_val: Optional[np.ndarray] = None, verbose: bool = False):
    """Self-supervised training.
    X_nan       training data (N, T, D) with NaN at missing positions.
    X_val_true  optional validation data with known values and M_val (True = position hidden for validation):
                enables early stopping on the validation imputation RMSE; otherwise cfg.max_steps steps.
    Returns (model, history)."""
    device = torch.device(cfg.device)
    set_seed(seed); gen = torch.Generator(device=device); gen.manual_seed(seed)
    N, T, D = X_nan.shape
    X, O = _tensors(X_nan, device)
    mu = np.nanmean(X_nan, axis=(0, 1)); var = np.nanvar(X_nan, axis=(0, 1), ddof=1)
    mu_tr = torch.tensor(np.nan_to_num(mu), dtype=torch.float32, device=device)
    var_tr = torch.tensor(np.where(np.isfinite(var), var, 1.0), dtype=torch.float32, device=device)

    model = DTINv2Net(D, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, cfg.max_steps) / cfg.max_steps)))
    X_val_nan = None
    if X_val_true is not None:
        X_val_nan = X_val_true.copy(); X_val_nan[M_val] = np.nan

    best, best_state, best_step, bad, step, history = np.inf, None, 0, 0, 0, []
    for step in range(1, cfg.max_steps + 1):
        model.train()
        idx = torch.randint(N, (min(cfg.batch_size, N),), generator=gen, device=device)
        x, o = X[idx], O[idx]
        a = artificial_mask(o, cfg, gen); v = o * (1 - a)
        xhat = model(x, v)
        w, target = a, x
        if cfg.pseudo_weight > 0:
            mean, vr = local_gaussian_stats(x, v, cfg.window, mu_tr, var_tr)
            z = mean + vr.sqrt() * torch.randn(o.shape, generator=gen, device=device)
            target = torch.where(o.bool(), x, z); w = a + cfg.pseudo_weight * (1 - o)
        cnt = w.sum((1, 2))
        loss = ((((xhat - target) ** 2) * w).sum((1, 2)) / cnt.clamp(min=1e-8))[cnt > 0].mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()

        if X_val_nan is not None and step % cfg.eval_every == 0:
            rec = _reconstruct(model, X_val_nan, device)
            r = rmse_at(X_val_true, np.where(np.isnan(X_val_nan), rec, X_val_nan), M_val)
            history.append((step, r))
            if verbose: print(f"step {step:6d}  loss {loss.item():.4f}  val RMSE {r:.4f}")
            if r < best - 1e-6:
                best, best_step, bad = r, step, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model, dict(best_val_rmse=None if best == np.inf else float(best), best_step=best_step,
                       last_step=step, history=history)


# =============================================================================================== imputation API
class DTINv2Imputer:
    """Imputer with the final configuration.

        imp = DTINv2Imputer(seed=0).fit(X_train_nan, X_val_true, M_val)
        X_filled = imp.impute(X_test_nan)
    """

    def __init__(self, cfg: Optional[Config] = None, seed: int = 0):
        self.cfg = cfg or imputation_config(); self.seed = seed; self.model = None; self.info = None

    def fit(self, X_nan, X_val_true=None, M_val=None, verbose=False):
        self.n_features = X_nan.shape[2]
        self.model, self.info = train_dtin(X_nan, self.cfg, self.seed, X_val_true, M_val, verbose)
        return self

    def impute(self, X_nan, batch_size=256):
        rec = _reconstruct(self.model, X_nan, torch.device(self.cfg.device), batch_size)
        return np.where(np.isnan(X_nan), rec, X_nan)

    def save(self, path):
        torch.save(dict(cfg=asdict(self.cfg), seed=self.seed, n_features=self.n_features,
                        state=self.model.state_dict(), info=self.info), path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu")
        cfg = Config(**dict(ck["cfg"], span_ratios=tuple(ck["cfg"]["span_ratios"])))
        if device: cfg.device = device
        obj = cls(cfg, ck["seed"]); obj.n_features = ck["n_features"]; obj.info = ck["info"]
        obj.model = DTINv2Net(obj.n_features, cfg).to(cfg.device); obj.model.load_state_dict(ck["state"]); obj.model.eval()
        return obj


# =============================================================================================== downstream API
def pretrain_encoder(X_nan, cfg: Optional[Config] = None, seed: int = 0):
    """Self-supervised pre-training for downstream use (unlabelled training sequences only)."""
    model, _ = train_dtin(X_nan, cfg or pretraining_config(), seed)
    return model


@torch.no_grad()
def embed(model, X_nan, batch_size=256):
    """Mean-pooled representations (N, d_model)."""
    device = next(model.parameters()).device; model.eval(); out = []
    for b in range(0, len(X_nan), batch_size):
        x, o = _tensors(X_nan[b:b + batch_size], device)
        out.append(model.encode(x, o).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


class FineTuneClassifier(nn.Module):
    """Pre-trained encoder + linear head on mean-pooled outputs. Input: concat([values, mask]) on the last axis."""

    def __init__(self, encoder: DTINv2Net, n_features: int, n_classes: int = 2):
        super().__init__()
        self.encoder, self.D = encoder, n_features
        self.head = nn.Linear(encoder.pred.in_features, n_classes)

    def forward(self, xa):
        return self.head(self.encoder.encode(xa[..., : self.D], xa[..., self.D:]).mean(dim=1))


def _with_mask(X_nan):
    return np.concatenate([np.nan_to_num(X_nan), (~np.isnan(X_nan)).astype(float)], -1)


def _fit_classifier(encoder, X_nan, y, epochs, seed, lr=1e-3, batch_size=32, X_val_nan=None):
    device = next(encoder.parameters()).device
    set_seed(seed)
    clf = FineTuneClassifier(copy.deepcopy(encoder), X_nan.shape[2], int(y.max()) + 1).to(device)
    g = torch.Generator(); g.manual_seed(seed)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    Xt = torch.tensor(_with_mask(X_nan), dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    Xv = None if X_val_nan is None else torch.tensor(_with_mask(X_val_nan), dtype=torch.float32, device=device)
    val_probs = []
    for _ in range(epochs):
        clf.train(); perm = torch.randperm(len(Xt), generator=g).to(device)
        for b in range(0, len(Xt), batch_size):
            idx = perm[b:b + batch_size]
            loss = F.cross_entropy(clf(Xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if Xv is not None:
            clf.eval()
            with torch.no_grad(): val_probs.append(torch.softmax(clf(Xv), 1).cpu().numpy())
    clf.eval()
    return clf, (np.array(val_probs) if Xv is not None else None)


def _binary_f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d


def select_epochs_cv(encoder, X_nan, y, n_splits=5, max_epochs=100, seed=0):
    """Stratified CV on the training set: number of epochs at the earliest maximum of the fold-averaged
    validation binary F1 (positive class 1). Returns (epochs, mean_curve)."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    curves = []
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        _, probs = _fit_classifier(encoder, X_nan[tr], y[tr], max_epochs, seed + 100 * k, X_val_nan=X_nan[va])
        curves.append([_binary_f1(y[va], p.argmax(1)) for p in probs])
    curve = np.mean(curves, axis=0)
    return int(np.argmax(curve)) + 1, curve


def fit_classifier(encoder, X_nan, y, epochs, seed=0):
    """Fine-tune encoder + linear head on all training sequences for a fixed number of epochs."""
    return _fit_classifier(encoder, X_nan, y, epochs, seed)[0]


@torch.no_grad()
def predict_proba(clf, X_nan, batch_size=256):
    device = next(clf.parameters()).device; out = []
    for b in range(0, len(X_nan), batch_size):
        out.append(torch.softmax(clf(torch.tensor(_with_mask(X_nan[b:b + batch_size]), dtype=torch.float32,
                                                  device=device)), 1).cpu().numpy())
    return np.concatenate(out)


# =============================================================================================== demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T, D = 256, 96, 3
    t = np.linspace(0, 6 * np.pi, T)
    X = np.stack([np.sin(t[None] + rng.uniform(0, 2 * np.pi, (N, 1))) * rng.uniform(0.5, 2, (N, 1))
                  for _ in range(D)], -1) + 0.05 * rng.normal(size=(N, T, D))
    sc = Standardizer().fit(X[:200]); X = sc.transform(X)
    Xtr, Xva, Xte = X[:180], X[180:200], X[200:]
    M_tr, M_va, M_te = (rng.random(a.shape) < 0.3 for a in (Xtr, Xva, Xte))
    Xtr_nan = Xtr.copy(); Xtr_nan[M_tr] = np.nan
    Xte_nan = Xte.copy(); Xte_nan[M_te] = np.nan

    imp = DTINv2Imputer(imputation_config(max_steps=1_000, eval_every=100), seed=0).fit(Xtr_nan, Xva, M_va)
    print("imputation RMSE (MCAR 30 %):", round(rmse_at(Xte, imp.impute(Xte_nan), M_te), 4), "| best step", imp.info["best_step"])

    y = (X[:, :, 0].mean(1) > 0).astype(int)
    enc = pretrain_encoder(Xtr_nan, pretraining_config(max_steps=300), seed=0)
    ep, _ = select_epochs_cv(enc, Xtr_nan[:60], y[:60], max_epochs=30, seed=0)
    clf = fit_classifier(enc, Xtr_nan[:60], y[:60], ep, seed=0)
    acc = (predict_proba(clf, Xte_nan).argmax(1) == y[200:]).mean()
    print("downstream: epochs", ep, "| accuracy", round(float(acc), 3))
