"""
dtin_core.py - shared code for the DTIN verification experiments (APIN-D-26-02073).

Imported by DTIN_parallel.ipynb (control notebook) and dtin_worker.py (GPU worker processes).
Nothing runs at import time; call init(cfg) first.
"""
import os, json, time, math, random, hashlib, platform, zipfile, glob, socket, traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.linear_model import RidgeClassifierCV

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CFG = dict(
    out_dir="dtin_outputs",
    data_dir="data",
    siemens_csv="SmA-Four-Tank-Batch-Process_V2.csv",
    seeds=list(range(10)),
    rates=[0.1, 0.2, 0.3, 0.4, 0.5],
    gap_lengths=[10, 20, 30, 40, 50], gap_total=72,
    mar_alpha=1.0, mnar_quantile=0.70, mnar_p_high=0.6, mech_rate=0.30,
    window=20, mask_ratio=0.15,
    d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.1,
    lr=1e-3, batch_size=128, epochs=100,
    head_lr=1e-3, head_batch=32, head_epochs=100, n_val=2,
    dodger_positive_label="2",
    knn_k=5,
    csdi_samples=10,                       # PyPOTS training schedules: see PYPOTS_TRAIN
    cv_folds=10,
    siemens_len=180, siemens_segmentation="step", siemens_train_frac=700 / 1095, siemens_split_seed=0,
    siemens_drop_windows_with_time_gaps=True,
)
DEFAULT_RUN = dict(table5=True, table5_cv=True, table8=True, cv=False,
                   tables_2_3=False, table4=False, table6=False, table7=False)

CFG = dict(DEFAULT_CFG)
RUN = dict(DEFAULT_RUN)
_DATA = {}


def init(cfg=None, run=None, deterministic=True):
    """Set configuration (dicts or a config.json path) and GPU flags."""
    global CFG, RUN
    if isinstance(cfg, str):
        with open(cfg) as f:
            obj = json.load(f)
        cfg, run = obj["CFG"], obj["RUN"]
    CFG = dict(DEFAULT_CFG, **(cfg or {}))
    RUN = dict(DEFAULT_RUN, **(run or {}))
    os.makedirs(CFG["out_dir"], exist_ok=True)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def save_config(path=None):
    path = path or os.path.join(CFG["out_dir"], "config.json")
    with open(path, "w") as f:
        json.dump(dict(CFG=CFG, RUN=RUN), f, indent=2)
    return path


def load_ucr(name):
    from aeon.datasets import load_classification
    os.makedirs(CFG["data_dir"], exist_ok=True)
    Xtr, ytr = load_classification(name, split="train", extract_path=CFG["data_dir"])
    Xte, yte = load_classification(name, split="test", extract_path=CFG["data_dir"])
    return to_ntd(Xtr), np.asarray(ytr).astype(str), to_ntd(Xte), np.asarray(yte).astype(str)


def get_data(name):
    """ECG200 / DodgerLoopGame / Siemens, standardized with training statistics (cached per process)."""
    if name in _DATA:
        return _DATA[name]
    if name in ("ECG200", "DodgerLoopGame"):
        D = dict(zip(["Xtr", "ytr", "Xte", "yte"], load_ucr(name)))
    elif name == "Siemens":
        D = {}
        D["Xtr"], D["Xte"], D["info"] = load_siemens(CFG["siemens_csv"], CFG)
    else:
        raise ValueError(name)
    D["Xtr"], D["Xte"], D["norm"] = standardize(D["Xtr"], D["Xte"])
    _DATA[name] = D
    return D


# ============================== shared building blocks ==============================
def stable_seed(*keys):
    # deterministic 32-bit seed from any tuple of keys
    h = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return int(h[:8], 16)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def result_path(table, key):
    d = os.path.join(CFG["out_dir"], "results", table); os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ".json")


def cached(table, key):
    p = result_path(table, key)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_result(table, key, obj):
    p = result_path(table, key); tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, p)                                   # atomic: readers never see a half-written file


def env_info():
    info = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                numpy=np.__version__, pandas=pd.__version__, device=str(DEVICE))
    if DEVICE.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    for mod in ("pypots", "aeon", "sklearn", "scipy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    return info


def to_ntd(X):
    X = np.asarray(X, dtype=np.float64)          # aeon: (n, channels, T) -> (n, T, D)
    return np.transpose(X, (0, 2, 1))


def standardize(Xtr, Xte):
    mu = np.nanmean(Xtr, axis=(0, 1)); sd = np.nanstd(Xtr, axis=(0, 1))
    sd = np.where(sd > 0, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd, dict(mean=mu.tolist(), std=sd.tolist())


def class_table(name, ytr, yte):
    rows = []
    for split, y in (("train", ytr), ("test", yte)):
        for lab, cnt in zip(*np.unique(y, return_counts=True)):
            rows.append(dict(dataset=name, split=split, label=str(lab), count=int(cnt)))
    return pd.DataFrame(rows)


def load_siemens(path, cfg):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    meta = ["timestamp", "DeviationID ValueY", "CuStepNo ValueY"]
    var_cols = [c for c in df.columns if c not in meta]
    assert len(var_cols) == 39, len(var_cols)
    vals = df[var_cols].to_numpy(np.float64)
    step = df["CuStepNo ValueY"].to_numpy()
    ts = pd.to_datetime(df["timestamp"]).to_numpy()
    dt = np.r_[1.0, np.diff(ts).astype("timedelta64[ms]").astype(float) / 1000.0]
    batch_start = np.r_[0, np.flatnonzero((step[1:] == 1) & (step[:-1] != 1)) + 1]
    batch_id = np.zeros(len(df), int); batch_id[batch_start[1:]] = 1; batch_id = np.cumsum(batch_id)
    if cfg["siemens_segmentation"] == "step":
        bounds = np.r_[0, np.flatnonzero(np.diff(step) != 0) + 1, len(df)]
    elif cfg["siemens_segmentation"] == "batch":
        bounds = np.r_[batch_start, len(df)]
    else:
        raise ValueError(cfg["siemens_segmentation"])
    L = cfg["siemens_len"]; W, starts, wb, dropped = [], [], [], 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        for s in range(a, b - L + 1, L):
            if cfg["siemens_drop_windows_with_time_gaps"] and np.any(dt[s + 1:s + L] > 1.5):
                dropped += 1; continue
            W.append(vals[s:s + L]); starts.append(s); wb.append(batch_id[s])
    X = np.stack(W); wb = np.asarray(wb); starts = np.asarray(starts)
    rng = np.random.default_rng(cfg["siemens_split_seed"])
    target = round(cfg["siemens_train_frac"] * len(X)); train_b, cnt = [], 0
    for bid in rng.permutation(np.unique(wb)):
        if cnt >= target: break
        train_b.append(bid); cnt += int((wb == bid).sum())
    tr = np.isin(wb, train_b)
    info = dict(n_windows=len(X), n_train=int(tr.sum()), n_test=int((~tr).sum()), dropped_gap_windows=dropped,
                segmentation=cfg["siemens_segmentation"], var_cols=var_cols)
    p = os.path.join(cfg["out_dir"], "siemens_windows.csv"); tmp = f"{p}.{os.getpid()}.tmp"
    pd.DataFrame(dict(window_start_row=starts, batch=wb, split=np.where(tr, "train", "test"))).to_csv(tmp, index=False)
    os.replace(tmp, p)
    return X[tr], X[~tr], info


def rng_for(*keys):
    return np.random.default_rng(stable_seed(*keys))


def mask_mcar(shape, rate, rng):
    return rng.random(shape) < rate


def mask_mar(X, rate, rng, alpha):
    # P(M[n,t,d]=1) = sigmoid(alpha * x[n,t,d'] + beta), d' != d chosen at random, beta calibrated to the target rate
    N, T, D = X.shape
    src = np.array([rng.choice([j for j in range(D) if j != d]) for d in range(D)])
    Z = alpha * X[:, :, src]
    lo, hi = -50.0, 50.0
    for _ in range(100):
        beta = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(Z + beta)))
        lo, hi = (beta, hi) if p.mean() < rate else (lo, beta)
    return rng.random(X.shape) < p


def mask_mnar(X, rate, rng, q, p_high):
    # values above the q-quantile of their variable: p_high; others: calibrated so the overall rate is `rate`
    thr = np.quantile(X.reshape(-1, X.shape[2]), q, axis=0)
    high = X > thr
    frac = high.mean()
    p_low = (rate - p_high * frac) / (1 - frac)
    assert 0 <= p_low <= 1, p_low
    p = np.where(high, p_high, p_low)
    return rng.random(X.shape) < p


def gap_segments(L, total):
    k, r = divmod(total, L)
    return [L] * k + ([r] if r else [])


def mask_gaps(shape, L, total, rng):
    # per sequence and variable: non-overlapping intervals of nominal length L, the final one shortened,
    # exactly `total` points removed, at least one observed point between intervals
    N, T, D = shape
    M = np.zeros(shape, dtype=bool)
    for n in range(N):
        for d in range(D):
            segs = list(rng.permutation(gap_segments(L, total)))
            free = T - sum(segs) - (len(segs) - 1)
            cuts = np.sort(rng.integers(0, free + 1, size=len(segs)))
            pos, prev = 0, 0
            for c, l in zip(cuts, segs):
                pos += c - prev; M[n, pos:pos + l, d] = True; pos += l + 1; prev = c
    assert (M.sum(axis=1) == total).all()
    return M


def make_mask(dataset, X, mech, level, seed, split):
    rng = rng_for("mask", dataset, mech, level, seed, split)
    if mech == "MCAR":
        return mask_mcar(X.shape, level, rng)
    if mech == "MAR":
        return mask_mar(X, level, rng, CFG["mar_alpha"])
    if mech == "MNAR":
        return mask_mnar(X, level, rng, CFG["mnar_quantile"], CFG["mnar_p_high"])
    if mech == "GAP":
        return mask_gaps(X.shape, level, CFG["gap_total"], rng)
    raise ValueError(mech)


def apply_mask(X, M):
    Y = X.copy(); Y[M] = np.nan
    return Y


def rmse_at(Xtrue, Ximp, M):
    return float(np.sqrt(np.mean((Xtrue[M] - Ximp[M]) ** 2)))


class DTINNet(nn.Module):
    def __init__(self, D, cfg, max_len=2048):
        super().__init__()
        dm = cfg["d_model"]
        self.inp = nn.Linear(D, dm)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dm, 2).float() * (-math.log(10000.0) / dm))
        pe = torch.zeros(max_len, dm); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(dm, cfg["n_heads"], cfg["d_ff"], cfg["dropout"], batch_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg["n_layers"], enable_nested_tensor=False)
        self.pred = nn.Linear(dm, D)

    def encode(self, x):
        return self.enc(self.inp(x) + self.pe[: x.shape[1]])

    def forward(self, x):
        return self.pred(self.encode(x))


def window_bounds(T, w, device):
    t = torch.arange(T, device=device)
    start = (t - w // 2).clamp(0, max(T - w, 0))
    return start, (start + w).clamp(max=T)


def local_stats(x, valid, w, mu_tr, var_tr):
    # x, valid: (B, T, D). Returns local mean and variance with the fallbacks of Section 3.3.
    xd, vd = x.double(), valid.double()
    xv = xd * vd
    pad = lambda c: F.pad(c, (0, 0, 1, 0))
    c1, c2, cn = pad(xv.cumsum(1)), pad((xv * xv).cumsum(1)), pad(vd.cumsum(1))
    s, e = window_bounds(x.shape[1], w, x.device)
    S1, S2, N = c1[:, e] - c1[:, s], c2[:, e] - c2[:, s], cn[:, e] - cn[:, s]
    mean = S1 / N.clamp(min=1)
    var = ((S2 - N * mean ** 2) / (N - 1).clamp(min=1)).clamp(min=0)
    mu_tr, var_tr = mu_tr.double(), var_tr.double()
    mean = torch.where(N == 0, mu_tr.expand_as(mean), mean)                 # no valid value
    var = torch.where(N <= 1, var_tr.expand_as(var), var)                    # 0 or 1 valid value
    return mean.float(), var.float()


def train_stats(X):
    mu = np.nanmean(X, axis=(0, 1)); var = np.nanvar(X, axis=(0, 1), ddof=1)
    return torch.tensor(np.nan_to_num(mu), dtype=torch.float32, device=DEVICE), \
           torch.tensor(np.where(np.isfinite(var), var, 1.0), dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------------------------
# Original DTIN design (2022 slides "221125_dtin" + first blinded manuscript). See ORIG_SPEC.
# ---------------------------------------------------------------------------------------------
ORIG_MAX_SPAN = 20                       # mask span length ~ Uniform{1, ..., 20}
ORIG_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)  # masking ratio drawn per batch


def span_mask(o, ratio, gen, max_len=ORIG_MAX_SPAN):
    """Per sequence and variable: span starts ~ Bernoulli(ratio / mean span length), span length
    ~ Uniform{1..max_len}; covered positions are masked. Only originally observed positions are masked."""
    B, T, D = o.shape
    p = ratio / ((1 + max_len) / 2)
    starts = torch.rand(o.shape, generator=gen, device=o.device) < p
    lens = torch.randint(1, max_len + 1, o.shape, generator=gen, device=o.device)
    t = torch.arange(T, device=o.device).view(1, T, 1)
    reach = torch.where(starts, t + lens, torch.full_like(lens, -1))
    covered = torch.cummax(reach, dim=1).values > t
    return covered.float() * o


def sliding_gaussian_targets(x, valid, w, mu_tr, var_tr, gen):
    """For every sliding window of length w: Gaussian (mean, unbiased variance) of the valid values,
    one sample per window. Target at t = average of the samples of all windows covering t.
    Windows without valid values are ignored; one valid value -> training variance.
    No covering window with valid values -> sample from N(training mean, training variance)."""
    B, T, D = x.shape
    w = min(w, T); S = T - w + 1
    vd = valid.double(); xd = x.double() * vd
    pad = lambda c: F.pad(c, (0, 0, 1, 0))
    c1, c2, cn = pad(xd.cumsum(1)), pad((xd * xd).cumsum(1)), pad(vd.cumsum(1))
    st = torch.arange(S, device=x.device)
    S1, S2, N = c1[:, st + w] - c1[:, st], c2[:, st + w] - c2[:, st], cn[:, st + w] - cn[:, st]
    mean = S1 / N.clamp(min=1)
    var = ((S2 - N * mean ** 2) / (N - 1).clamp(min=1)).clamp(min=0)
    var = torch.where(N <= 1, var_tr.double().expand_as(var), var)
    z = mean + var.sqrt() * torch.randn(mean.shape, generator=gen, device=x.device, dtype=torch.float64)
    v = (N > 0).double()
    Cz, Cv = pad((z * v).cumsum(1)), pad(v.cumsum(1))
    t = torch.arange(T, device=x.device)
    a = (t - w + 1).clamp(min=0); e = t.clamp(max=S - 1) + 1
    num, den = Cz[:, e] - Cz[:, a], Cv[:, e] - Cv[:, a]
    fb = mu_tr.double() + var_tr.double().sqrt() * torch.randn((B, T, D), generator=gen, device=x.device, dtype=torch.float64)
    return torch.where(den > 0, num / den.clamp(min=1), fb).float()


def train_ssl(Xtr_nan, mode, seed, window=None, epochs=None, log_every=0):
    # Xtr_nan: (N, T, D) with NaN at inherently missing positions
    cfg = CFG; window = window or cfg["window"]; epochs = epochs or cfg["epochs"]
    set_seed(seed)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
    gcpu = torch.Generator(); gcpu.manual_seed(seed)
    N, T, D = Xtr_nan.shape
    X = torch.tensor(np.nan_to_num(Xtr_nan), dtype=torch.float32, device=DEVICE)
    O = torch.tensor(~np.isnan(Xtr_nan), dtype=torch.float32, device=DEVICE)
    mu_tr, var_tr = train_stats(Xtr_nan)
    model = DTINNet(D, cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, generator=gcpu).to(DEVICE)
        for b in range(0, N, cfg["batch_size"]):
            idx = perm[b:b + cfg["batch_size"]]
            x, o = X[idx], O[idx]
            if mode in ("orig", "orig_mask_only"):
                r = ORIG_RATIOS[int(torch.randint(len(ORIG_RATIOS), (1,), generator=gcpu))]
                a = span_mask(o, r, gen)
                xhat = model(x * o * (1 - a))
                if mode == "orig":     # loss over all positions; inherently missing -> sliding-window Gaussian target
                    target = torch.where(o.bool(), x, sliding_gaussian_targets(x, o * (1 - a), window, mu_tr, var_tr, gen))
                    loss = ((xhat - target) ** 2).mean()
                else:                  # TST-style: loss only on artificially masked observed positions
                    loss = (((xhat - x) ** 2) * a).sum() / a.sum().clamp(min=1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                continue
            a = (torch.rand(o.shape, generator=gen, device=DEVICE) < cfg["mask_ratio"]).float() * o
            xhat = model(x * o * (1 - a))
            if mode == "mask_only":
                tset, target = a, x
            else:
                mean, var = local_stats(x, o * (1 - a), window, mu_tr, var_tr)
                z = mean if mode == "mean" else mean + var.sqrt() * torch.randn(o.shape, generator=gen, device=DEVICE)
                target = torch.where(o.bool(), x, z)
                tset = a + (1 - o)
            cnt = tset.sum((1, 2))
            per_seq = (((xhat - target) ** 2) * tset).sum((1, 2)) / cnt.clamp(min=1)
            loss = per_seq[cnt > 0].mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if log_every and (ep + 1) % log_every == 0:
            print(f"  epoch {ep + 1}/{epochs} loss {loss.item():.4f}")
    model.eval()
    return model


@torch.no_grad()
def ssl_impute(model, X_nan, bs=256):
    out = []
    for b in range(0, len(X_nan), bs):
        xb = X_nan[b:b + bs]
        x = torch.tensor(np.nan_to_num(xb), dtype=torch.float32, device=DEVICE)
        o = torch.tensor(~np.isnan(xb), dtype=torch.float32, device=DEVICE)
        out.append(model(x * o).cpu().numpy())
    pred = np.concatenate(out).astype(np.float64)
    return np.where(np.isnan(X_nan), pred, X_nan)


@torch.no_grad()
def ssl_embed(model, X_nan, bs=256):
    out = []
    for b in range(0, len(X_nan), bs):
        xb = X_nan[b:b + bs]
        x = torch.tensor(np.nan_to_num(xb), dtype=torch.float32, device=DEVICE)
        o = torch.tensor(~np.isnan(xb), dtype=torch.float32, device=DEVICE)
        out.append(model.encode(x * o).mean(dim=1).cpu().numpy())      # mean pooling over time
    return np.concatenate(out)


def impute_seq_stat(X_nan, fn):
    stat = fn(X_nan, axis=1, keepdims=True)
    stat = np.where(np.isnan(stat), 0.0, stat)              # variable fully missing in a sequence -> training mean (0)
    return np.where(np.isnan(X_nan), np.broadcast_to(stat, X_nan.shape), X_nan)


def impute_knn(Xtr_nan, Xte_nan, k):
    N, T, D = Xte_nan.shape
    imp = KNNImputer(n_neighbors=k).fit(Xtr_nan.reshape(len(Xtr_nan), -1))
    out = imp.transform(Xte_nan.reshape(N, -1))
    if out.shape[1] != T * D:                                # columns that are empty in training are dropped by sklearn
        raise RuntimeError("A feature is missing in every training sequence; KNN imputation is undefined.")
    return out.reshape(N, T, D)


def impute_lr_time(X_nan):
    N, T, D = X_nan.shape
    t = np.arange(T, dtype=float)[None, :, None]
    obs = ~np.isnan(X_nan); n = obs.sum(1, keepdims=True).clip(min=1)
    y = np.nan_to_num(X_nan)
    tm = (t * obs).sum(1, keepdims=True) / n; ym = (y * obs).sum(1, keepdims=True) / n
    cov = ((t - tm) * (y - ym) * obs).sum(1, keepdims=True); var = (((t - tm) ** 2) * obs).sum(1, keepdims=True)
    pred = ym + np.where(var > 0, cov / np.where(var > 0, var, 1), 0) * (t - tm)
    return np.where(obs, X_nan, pred)


PYPOTS_GRID = {
    "BRITS": [dict(rnn_hidden_size=h) for h in (32, 64, 128)],
    "SAITS": [dict(n_layers=2, d_model=dm, n_heads=4, d_k=dm // 4, d_v=dm // 4, d_ffn=128, dropout=0.1) for dm in (64, 128, 256)],
    "CSDI":  [dict(n_layers=4, n_heads=8, n_channels=ch, d_time_embedding=128, d_feature_embedding=16,
                   d_diffusion_embedding=128, n_diffusion_steps=50) for ch in (32, 64)],
}


# Training schedules. CSDI follows the original study (Tashiro et al., 2021): 200 epochs, batch 16,
# Adam lr 1e-3 with weight decay 1e-6, learning rate multiplied by 0.1 at 75 % and 90 % of training.
PYPOTS_TRAIN = {
    "CSDI": dict(epochs=200, batch_size=16, lr=1e-3, weight_decay=1e-6, milestones=(0.75, 0.90)),
    "BRITS": dict(epochs=100, batch_size=32, lr=1e-3, weight_decay=0.0, milestones=None),
    "SAITS": dict(epochs=100, batch_size=32, lr=1e-3, weight_decay=0.0, milestones=None),
}
# Architecture of the original CSDI study; used when RUN["cv"] is False.
ORIGINAL_HP = {
    "CSDI": dict(n_layers=4, n_heads=8, n_channels=64, d_time_embedding=128, d_feature_embedding=16,
                 d_diffusion_embedding=128, n_diffusion_steps=50),
}


def pypots_model(name, T, D, hp, seed, n_train):
    from pypots.imputation import BRITS, SAITS, CSDI
    from pypots.optim import Adam
    from pypots.optim.lr_scheduler import MultiStepLR
    tc = PYPOTS_TRAIN[name]
    sched = None
    if tc["milestones"]:                                   # PyPOTS steps the scheduler after every batch
        iters = tc["epochs"] * math.ceil(n_train / tc["batch_size"])
        sched = MultiStepLR(milestones=[int(f * iters) for f in tc["milestones"]], gamma=0.1)
    set_seed(seed)
    opt = Adam(lr=tc["lr"], weight_decay=tc["weight_decay"], lr_scheduler=sched)
    return dict(BRITS=BRITS, SAITS=SAITS, CSDI=CSDI)[name](
        **hp, n_steps=T, n_features=D, batch_size=tc["batch_size"], epochs=tc["epochs"], optimizer=opt,
        device=DEVICE, saving_path=None, model_saving_strategy=None, verbose=False)


def pypots_impute(name, Xtr_nan, Xte_nan, hp, seed):
    T, D = Xtr_nan.shape[1:]
    m = pypots_model(name, T, D, hp, seed, n_train=len(Xtr_nan))
    m.fit({"X": Xtr_nan.astype(np.float32)})
    kw = dict(n_sampling_times=CFG["csdi_samples"]) if name == "CSDI" else {}
    imp = np.asarray(m.impute({"X": Xte_nan.astype(np.float32)}, **kw), dtype=np.float64)
    if imp.ndim == 4:                                         # CSDI samples -> median (as in the CSDI paper)
        imp = np.median(imp, axis=1)
    del m; torch.cuda.empty_cache()
    return np.where(np.isnan(Xte_nan), imp, Xte_nan)


def summarize(rows, by, value="rmse"):
    df = pd.DataFrame(rows)
    g = df.groupby(by)[value]
    out = g.agg(["mean", lambda s: s.std(ddof=1), "count"]).rename(columns={"<lambda_0>": "std"})
    out["cell"] = out.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    return out


def confusion(y, p):
    return dict(TP=int(((y == 1) & (p == 1)).sum()), FP=int(((y == 0) & (p == 1)).sum()),
                FN=int(((y == 1) & (p == 0)).sum()), TN=int(((y == 0) & (p == 0)).sum()))


def binary_f1(c):
    d = 2 * c["TP"] + c["FP"] + c["FN"]
    return 0.0 if d == 0 else 2 * c["TP"] / d


def val_split(seed, n_train):
    rng = np.random.default_rng(stable_seed("val", seed))
    val = np.sort(rng.choice(n_train, size=CFG["n_val"], replace=False))
    return np.setdiff1d(np.arange(n_train), val), val


def interp_nan(X):
    X = X.copy()
    for n in range(X.shape[0]):
        for d in range(X.shape[2]):
            v = X[n, :, d]; m = np.isnan(v)
            if m.any() and (~m).any():
                X[n, m, d] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), v[~m])
            elif m.all():
                X[n, :, d] = 0.0
    return X


def train_with_selection(model, Xtr, ytr, Xval, yval, seed):
    # generic loop: Adam, CE, batch 32, 100 epochs, earliest-best validation F1
    set_seed(seed)
    gcpu = torch.Generator(); gcpu.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["head_lr"])
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE); yt = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(Xval, dtype=torch.float32, device=DEVICE)
    best, best_ep, best_state, curve = -np.inf, 0, None, []
    for ep in range(CFG["head_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=gcpu).to(DEVICE)
        for b in range(0, len(Xt), CFG["head_batch"]):
            idx = perm[b:b + CFG["head_batch"]]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).cpu().numpy()
        f1 = binary_f1(confusion(yval, pv)); curve.append(f1)
        if f1 > best:
            best, best_ep = f1, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, best_ep + 1, curve


@torch.no_grad()
def predict_proba(model, X, bs=256):
    out = []
    for b in range(0, len(X), bs):
        out.append(torch.softmax(model(torch.tensor(X[b:b + bs], dtype=torch.float32, device=DEVICE)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


class Head(nn.Module):
    def __init__(self, d): super().__init__(); self.fc = nn.Linear(d, 2)
    def forward(self, z): return self.fc(z)


class LSTMNet(nn.Module):
    def __init__(self, D, h=64):
        super().__init__(); self.lstm = nn.LSTM(D, h, batch_first=True); self.fc = nn.Linear(h, 2)
    def forward(self, x): return self.fc(self.lstm(x)[1][0][-1])


class ResNet1D(nn.Module):                        # Wang et al. (2017)
    def __init__(self, D):
        super().__init__()
        def block(cin, cout):
            return nn.ModuleDict(dict(
                main=nn.Sequential(nn.Conv1d(cin, cout, 8, padding="same"), nn.BatchNorm1d(cout), nn.ReLU(),
                                   nn.Conv1d(cout, cout, 5, padding="same"), nn.BatchNorm1d(cout), nn.ReLU(),
                                   nn.Conv1d(cout, cout, 3, padding="same"), nn.BatchNorm1d(cout)),
                short=nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))))
        self.blocks = nn.ModuleList([block(D, 64), block(64, 128), block(128, 128)]); self.fc = nn.Linear(128, 2)
    def forward(self, x):
        h = x.transpose(1, 2)
        for b in self.blocks: h = F.relu(b["main"](h) + b["short"](h))
        return self.fc(h.mean(-1))


class InceptionModule(nn.Module):
    def __init__(self, cin, nf=32, ks=(39, 19, 9)):
        super().__init__()
        self.bott = nn.Conv1d(cin, nf, 1, bias=False) if cin > 1 else nn.Identity()
        c = nf if cin > 1 else cin
        self.convs = nn.ModuleList([nn.Conv1d(c, nf, k, padding="same", bias=False) for k in ks])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * nf)
    def forward(self, x):
        b = self.bott(x)
        return F.relu(self.bn(torch.cat([c(b) for c in self.convs] + [self.pool(x)], 1)))


class InceptionNet(nn.Module):                    # one InceptionTime network (Ismail Fawaz et al., 2020)
    def __init__(self, D, nf=32, depth=6):
        super().__init__()
        self.mods = nn.ModuleList(); self.res = nn.ModuleList(); cin, rin = D, D
        for i in range(depth):
            self.mods.append(InceptionModule(cin, nf)); cin = 4 * nf
            if i % 3 == 2:
                self.res.append(nn.Sequential(nn.Conv1d(rin, cin, 1, bias=False), nn.BatchNorm1d(cin))); rin = cin
        self.fc = nn.Linear(cin, 2)
    def forward(self, x):
        h = x.transpose(1, 2); r = h; j = 0
        for i, m in enumerate(self.mods):
            h = m(h)
            if i % 3 == 2: h = F.relu(h + self.res[j](r)); r = h; j += 1
        return self.fc(h.mean(-1))


# =============================================================================================
# Hyper-parameter selection for BRITS / SAITS / CSDI (training-set-only K-fold CV, parallel)
# =============================================================================================
def cv_fold_key(dataset, name, hp_idx, fold):
    return f"{dataset}_{name}_hp{hp_idx}_fold{fold}"


def cv_folds(dataset, n):
    return np.array_split(rng_for("cv", dataset).permutation(n), CFG["cv_folds"])


def run_cv_fold(dataset, name, hp_idx, fold):
    key = cv_fold_key(dataset, name, hp_idx, fold)
    hit = cached("cv_folds", key)
    if hit: return hit
    D = get_data(dataset); X = D["Xtr"]
    M = make_mask(dataset, X, "MCAR", 0.3, 0, "train"); Xm = apply_mask(X, M)
    val = cv_folds(dataset, len(X))[fold]; tr = np.setdiff1d(np.arange(len(X)), val)
    hp = PYPOTS_GRID[name][hp_idx]; t0 = time.time()
    imp = pypots_impute(name, Xm[tr], Xm[val], hp, stable_seed("cv", dataset, name, fold))
    res = dict(dataset=dataset, model=name, hp_idx=hp_idx, hp=hp, fold=fold,
               rmse=rmse_at(X[val], imp, M[val]), seconds=round(time.time() - t0, 1))
    save_result("cv_folds", key, res)
    return res


def reduce_cv(dataset, name):
    """Pick the grid entry with the lowest mean held-out-fold RMSE; requires all folds to be finished."""
    scores = []
    for i in range(len(PYPOTS_GRID[name])):
        r = [cached("cv_folds", cv_fold_key(dataset, name, i, k)) for k in range(CFG["cv_folds"])]
        if any(x is None for x in r):
            return None
        scores.append(float(np.mean([x["rmse"] for x in r])))
    best = PYPOTS_GRID[name][int(np.argmin(scores))]
    out = dict(grid=PYPOTS_GRID[name], scores=scores, best=best, folds=CFG["cv_folds"])
    save_result("cv", f"{dataset}_{name}", out)
    return out


def select_hparams(name, dataset, Xtr_full=None):
    hit = cached("cv", f"{dataset}_{name}")
    if hit: return hit["best"]
    if not RUN["cv"]:                                       # original configuration, no CV
        best = ORIGINAL_HP.get(name, PYPOTS_GRID[name][len(PYPOTS_GRID[name]) // 2])
        save_result("cv", f"{dataset}_{name}", dict(grid=PYPOTS_GRID[name], scores=None, best=best, folds=0,
                                                    note="CV disabled: original configuration"))
        return best
    r = reduce_cv(dataset, name)
    if r is None:
        raise RuntimeError(f"CV for {name} on {dataset} is not finished. Run the 'cv' stage first.")
    return r["best"]


# =============================================================================================
# Experiment runners (one task = one cached result file)
# =============================================================================================
def run_imputation(table, dataset, mech, level, seed, method, window=None):
    key = f"{dataset}_{mech}_{level}_seed{seed}_{method}" + (f"_w{window}" if window else "")
    hit = cached(table, key)
    if hit: return hit
    D = get_data(dataset); Xtr, Xte = D["Xtr"], D["Xte"]
    Mtr = make_mask(dataset, Xtr, mech, level, seed, "train")
    Mte = make_mask(dataset, Xte, mech, level, seed, "test")
    Xtr_m, Xte_m = apply_mask(Xtr, Mtr), apply_mask(Xte, Mte)
    t0 = time.time()
    if method == "Mean":
        imp = impute_seq_stat(Xte_m, np.nanmean)
    elif method == "Median":
        imp = impute_seq_stat(Xte_m, np.nanmedian)
    elif method == "KNN":
        imp = impute_knn(Xtr_m, Xte_m, CFG["knn_k"])
    elif method == "Linear regression":
        imp = impute_lr_time(Xte_m)
    elif method in ("BRITS", "SAITS", "CSDI"):
        hp = select_hparams(method, dataset)
        imp = pypots_impute(method, Xtr_m, Xte_m, hp, stable_seed("model", dataset, mech, level, seed, method))
    elif method in ("DTIN", "Mask only", "Local mean", "DTIN-orig"):
        mode = {"DTIN": "dtin", "Mask only": "mask_only", "Local mean": "mean", "DTIN-orig": "orig"}[method]
        model = train_ssl(Xtr_m, mode, stable_seed("model", dataset, mech, level, seed, method), window=window)
        imp = ssl_impute(model, Xte_m); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        raise ValueError(method)
    res = dict(table=table, dataset=dataset, mech=mech, level=level, seed=seed, method=method, window=window,
               rmse=rmse_at(Xte, imp, Mte), n_removed=int(Mte.sum()), seconds=round(time.time() - t0, 1))
    save_result(table, key, res)
    return res


def run_downstream_method(method, seed):
    hit = cached("table5", f"{method}_seed{seed}")
    if hit: return hit
    DOD = get_data("DodgerLoopGame")
    Xtr, Xte = DOD["Xtr"], DOD["Xte"]
    ytr = (DOD["ytr"] == CFG["dodger_positive_label"]).astype(int)
    yte = (DOD["yte"] == CFG["dodger_positive_label"]).astype(int)
    tr, val = val_split(seed, len(Xtr))
    s = stable_seed("downstream", method, seed); sel_ep = None; t0 = time.time()
    if method in ("DTIN", "TST-mask"):
        enc = train_ssl(Xtr, "dtin" if method == "DTIN" else "mask_only", s)      # 20 training sequences, no labels
        Ztr, Zte = ssl_embed(enc, Xtr), ssl_embed(enc, Xte); del enc
        set_seed(s); head = Head(Ztr.shape[1]).to(DEVICE)
        head, sel_ep, _ = train_with_selection(head, Ztr[tr], ytr[tr], Ztr[val], ytr[val], s)
        score = predict_proba(head, Zte); pred = (score >= 0.5).astype(int)
    elif method in ("LSTM", "ResNet", "InceptionTime"):
        Xi_tr, Xi_te = interp_nan(Xtr), interp_nan(Xte)
        n_models = 5 if method == "InceptionTime" else 1                   # InceptionTime = ensemble of 5
        scores, eps = [], []
        for k in range(n_models):
            set_seed(s + k)
            net = dict(LSTM=LSTMNet, ResNet=ResNet1D, InceptionTime=InceptionNet)[method](Xtr.shape[2]).to(DEVICE)
            net, ep, _ = train_with_selection(net, Xi_tr[tr], ytr[tr], Xi_tr[val], ytr[val], s + k)
            scores.append(predict_proba(net, Xi_te)); eps.append(ep)
        score = np.mean(scores, 0); pred = (score >= 0.5).astype(int)
        sel_ep = eps[0] if n_models == 1 else eps
    elif method == "ROCKET":
        from aeon.transformations.collection.convolution_based import Rocket
        Xi_tr, Xi_te = interp_nan(Xtr).transpose(0, 2, 1), interp_nan(Xte).transpose(0, 2, 1)
        rk = Rocket(n_kernels=10_000, random_state=s % (2**31 - 1)).fit(Xi_tr[tr])
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(rk.transform(Xi_tr[tr]), ytr[tr])
        Fte = rk.transform(Xi_te); score = clf.decision_function(Fte); pred = clf.predict(Fte).astype(int)
    else:
        raise ValueError(method)
    c = confusion(yte, pred)
    res = dict(method=method, seed=seed, **c, n_test=int(len(yte)), n_test_positive=int(yte.sum()),
               accuracy=(c["TP"] + c["TN"]) / len(yte), f1=binary_f1(c), auroc=float(roc_auc_score(yte, score)),
               selected_epoch=sel_ep, val_idx=val.tolist(), val_labels=DOD["ytr"][val].tolist(),
               seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.join(CFG["out_dir"], "results", "table5"), exist_ok=True)
    np.savez(os.path.join(CFG["out_dir"], "results", "table5", f"{method}_seed{seed}_predictions.npz"),
             y_true=yte, y_pred=pred, score=score)
    assert abs(res["accuracy"] - (c["TP"] + c["TN"]) / sum(c.values())) < 1e-12
    save_result("table5", f"{method}_seed{seed}", res)
    return res


# =============================================================================================
# Revised downstream protocol (Table 5, "table5_cv")
#   * same frozen encoders / architectures / optimiser as the original protocol
#   * model selection: stratified 5-fold CV on the 20 training sequences chooses the number of
#     training epochs (earliest maximum of the fold-averaged validation binary F1)
#   * final model retrained on all 20 training sequences for that number of epochs, evaluated once
#   * ROCKET: fitted on all 20 training sequences
# =============================================================================================
def _train_epochs(make_model, X, y, seed, epochs, Xval=None):
    """Adam / CE / batch 32. Returns the model and, if Xval is given, the validation
    probabilities of the positive class after every epoch (epochs x n_val)."""
    set_seed(seed); model = make_model()
    gcpu = torch.Generator(); gcpu.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["head_lr"])
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE); yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    probs = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gcpu).to(DEVICE)
        for b in range(0, len(Xt), CFG["head_batch"]):
            idx = perm[b:b + CFG["head_batch"]]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if Xval is not None:
            model.eval(); probs.append(predict_proba(model, Xval))
    model.eval()
    return model, (np.array(probs) if Xval is not None else None)


def run_downstream_cv(method, seed):
    hit = cached("table5_cv", f"{method}_seed{seed}")
    if hit: return hit
    from sklearn.model_selection import StratifiedKFold
    DOD = get_data("DodgerLoopGame")
    Xtr, Xte = DOD["Xtr"], DOD["Xte"]
    ytr = (DOD["ytr"] == CFG["dodger_positive_label"]).astype(int)
    yte = (DOD["yte"] == CFG["dodger_positive_label"]).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=stable_seed("fold", seed) % (2**31 - 1))
    folds = list(skf.split(np.zeros(len(ytr)), ytr))
    s = stable_seed("downstream_cv", method, seed); t0 = time.time(); E = CFG["head_epochs"]
    chosen, mean_curve = None, None
    if method == "ROCKET":
        from aeon.transformations.collection.convolution_based import Rocket
        Xi_tr, Xi_te = interp_nan(Xtr).transpose(0, 2, 1), interp_nan(Xte).transpose(0, 2, 1)
        rk = Rocket(n_kernels=10_000, random_state=s % (2**31 - 1)).fit(Xi_tr)
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(rk.transform(Xi_tr), ytr)
        Fte = rk.transform(Xi_te); score = clf.decision_function(Fte); pred = clf.predict(Fte).astype(int)
    else:
        if method in ("DTIN", "TST-mask"):
            # identical encoder to the original protocol (same seed): only model selection differs
            enc = train_ssl(Xtr, "dtin" if method == "DTIN" else "mask_only", stable_seed("downstream", method, seed))
            Xa, Xb = ssl_embed(enc, Xtr), ssl_embed(enc, Xte); del enc
            make = lambda: Head(Xa.shape[1]).to(DEVICE); n_models = 1
        else:
            Xa, Xb = interp_nan(Xtr), interp_nan(Xte)
            net = dict(LSTM=LSTMNet, ResNet=ResNet1D, InceptionTime=InceptionNet)[method]
            make = lambda: net(Xtr.shape[2]).to(DEVICE)
            n_models = 5 if method == "InceptionTime" else 1
        fold_curves = []
        for k, (tr, va) in enumerate(folds):
            p = np.mean([_train_epochs(make, Xa[tr], ytr[tr], s + 100 * k + j, E, Xval=Xa[va])[1]
                         for j in range(n_models)], axis=0)                     # (epochs, n_val)
            fold_curves.append([binary_f1(confusion(ytr[va], (pe >= 0.5).astype(int))) for pe in p])
        mean_curve = np.mean(fold_curves, axis=0)
        chosen = int(np.argmax(mean_curve)) + 1                                   # earliest maximum
        score = np.mean([predict_proba(_train_epochs(make, Xa, ytr, s + 10_000 + j, chosen)[0], Xb)
                         for j in range(n_models)], axis=0)
        pred = (score >= 0.5).astype(int)
    c = confusion(yte, pred)
    res = dict(method=method, seed=seed, **c, n_test=int(len(yte)), n_test_positive=int(yte.sum()),
               accuracy=(c["TP"] + c["TN"]) / len(yte), f1=binary_f1(c), auroc=float(roc_auc_score(yte, score)),
               selected_epoch=chosen, cv_mean_val_f1=None if mean_curve is None else [round(float(v), 4) for v in mean_curve],
               fold_val_labels=[DOD["ytr"][va].tolist() for _, va in folds], seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.join(CFG["out_dir"], "results", "table5_cv"), exist_ok=True)
    np.savez(os.path.join(CFG["out_dir"], "results", "table5_cv", f"{method}_seed{seed}_predictions.npz"),
             y_true=yte, y_pred=pred, score=score)
    save_result("table5_cv", f"{method}_seed{seed}", res)
    return res


# =============================================================================================
# Original-design downstream protocol (Table 5, "table5_ft"): fine-tuning + MLP classifier
# =============================================================================================
T5_FT_METHODS = ["LSTM", "ResNet", "ROCKET", "InceptionTime", "TST-orig", "DTIN-orig"]

ORIG_SPEC = """# Pre-specified reconstruction of the original DTIN design

Written before any original-design result was computed. Sources: slides "221125_dtin" (25 Nov 2022)
and the first blinded manuscript. Every result obtained under this specification is reported,
together with the results of the implementation that follows the revised manuscript text.
No further variants are tried after these results are seen.

## Pre-training (method "DTIN-orig")
- Architecture, optimiser, epochs, batch size: identical to the revised-manuscript implementation
  (3 layers, d_model 64, 4 heads, FFN 128, dropout 0.1, Adam 1e-3, 100 epochs, batch 128); not specified in the sources.
- Mask strategy: span masking per sequence and variable; span length ~ Uniform{1..20}; span starts
  ~ Bernoulli(r / 10.5); masking ratio r drawn per batch from {0.1, 0.2, 0.3, 0.4, 0.5} (spans may overlap,
  so the realised masked fraction is about 0.09 / 0.25 / 0.39 for r = 0.1 / 0.3 / 0.5). Masked and
  inherently missing positions are set to 0 in the input.
- Target for inherently missing positions: sliding windows of length 20; in each window a Gaussian
  (mean, unbiased variance) of the valid input values; one sample per window; the target is the average
  of the samples of all windows covering the position (fallbacks as in Section 3.3). Resampled every batch.
- Loss: mean squared error over all positions (observed values -> original value; inherently missing -> target).

## TST comparison ("TST-orig")
Same masking and architecture; loss only on artificially masked observed positions.

## Downstream ("table5_ft")
- DTIN-orig / TST-orig: pre-training on the 20 training sequences (no labels); the encoder is fine-tuned
  together with an MLP classifier (64 -> 64 -> 2, ReLU) on mean-pooled outputs.
- Baselines: missing values filled with the per-sequence median (as in the slides).
- Model selection for all methods: stratified 5-fold CV on the 20 training sequences chooses the number of
  epochs (earliest maximum of the fold-averaged validation binary F1); final model retrained on all 20.
  ROCKET fitted on all 20. Adam 1e-3, batch 32, up to 100 epochs. Game Day = positive class.

## Imputation ("DTIN-orig" in Table 8 and Tables 2-3)
Same data, masks, seeds and evaluation as all other methods. At test time inherently missing positions
are input as 0 and the model output is used at those positions.
"""


def write_orig_spec():
    path = os.path.join(CFG["out_dir"], "SPEC_original_design.md")
    if not os.path.exists(path):                       # never overwritten: the first version is the record
        with open(path, "w", encoding="utf-8") as f:
            f.write(ORIG_SPEC + f"\nWritten: {time.ctime()}\n")
    return path


class FTClassifier(nn.Module):
    def __init__(self, encoder, d_model, hidden=64):
        super().__init__()
        self.enc = encoder
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.mlp(self.enc.encode(x).mean(dim=1))


def median_fill(X):
    med = np.nanmedian(X, axis=1, keepdims=True)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(X), np.broadcast_to(med, X.shape), X)


def run_downstream_ft(method, seed):
    hit = cached("table5_ft", f"{method}_seed{seed}")
    if hit: return hit
    from sklearn.model_selection import StratifiedKFold
    DOD = get_data("DodgerLoopGame")
    Xtr, Xte = DOD["Xtr"], DOD["Xte"]
    ytr = (DOD["ytr"] == CFG["dodger_positive_label"]).astype(int)
    yte = (DOD["yte"] == CFG["dodger_positive_label"]).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=stable_seed("fold", seed) % (2**31 - 1))
    folds = list(skf.split(np.zeros(len(ytr)), ytr))
    s = stable_seed("downstream_ft", method, seed); t0 = time.time(); E = CFG["head_epochs"]
    chosen, mean_curve = None, None
    if method == "ROCKET":
        from aeon.transformations.collection.convolution_based import Rocket
        Xi_tr, Xi_te = median_fill(Xtr).transpose(0, 2, 1), median_fill(Xte).transpose(0, 2, 1)
        rk = Rocket(n_kernels=10_000, random_state=s % (2**31 - 1)).fit(Xi_tr)
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(rk.transform(Xi_tr), ytr)
        Fte = rk.transform(Xi_te); score = clf.decision_function(Fte); pred = clf.predict(Fte).astype(int)
    else:
        D = Xtr.shape[2]
        if method in ("DTIN-orig", "TST-orig"):
            enc = train_ssl(Xtr, "orig" if method == "DTIN-orig" else "orig_mask_only", stable_seed("downstream_ft_pre", method, seed))
            enc_state = {k: v.detach().clone() for k, v in enc.state_dict().items()}; del enc
            def make():
                e = DTINNet(D, CFG).to(DEVICE); e.load_state_dict(enc_state)
                return FTClassifier(e, CFG["d_model"]).to(DEVICE)
            Xa, Xb = np.nan_to_num(Xtr), np.nan_to_num(Xte); n_models = 1
        else:
            net = dict(LSTM=LSTMNet, ResNet=ResNet1D, InceptionTime=InceptionNet)[method]
            make = lambda: net(D).to(DEVICE)
            Xa, Xb = median_fill(Xtr), median_fill(Xte)
            n_models = 5 if method == "InceptionTime" else 1
        fold_curves = []
        for k, (tr, va) in enumerate(folds):
            p = np.mean([_train_epochs(make, Xa[tr], ytr[tr], s + 100 * k + j, E, Xval=Xa[va])[1]
                         for j in range(n_models)], axis=0)
            fold_curves.append([binary_f1(confusion(ytr[va], (pe >= 0.5).astype(int))) for pe in p])
        mean_curve = np.mean(fold_curves, axis=0)
        chosen = int(np.argmax(mean_curve)) + 1
        score = np.mean([predict_proba(_train_epochs(make, Xa, ytr, s + 10_000 + j, chosen)[0], Xb)
                         for j in range(n_models)], axis=0)
        pred = (score >= 0.5).astype(int)
    c = confusion(yte, pred)
    res = dict(method=method, seed=seed, **c, n_test=int(len(yte)), n_test_positive=int(yte.sum()),
               accuracy=(c["TP"] + c["TN"]) / len(yte), f1=binary_f1(c), auroc=float(roc_auc_score(yte, score)),
               selected_epoch=chosen, cv_mean_val_f1=None if mean_curve is None else [round(float(v), 4) for v in mean_curve],
               fold_val_labels=[DOD["ytr"][va].tolist() for _, va in folds], seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.join(CFG["out_dir"], "results", "table5_ft"), exist_ok=True)
    np.savez(os.path.join(CFG["out_dir"], "results", "table5_ft", f"{method}_seed{seed}_predictions.npz"),
             y_true=yte, y_pred=pred, score=score)
    save_result("table5_ft", f"{method}_seed{seed}", res)
    return res


def archive_csdi_v1():
    """Move CSDI results obtained with the first (too short) training schedule out of the result folders.
    Runs only once; the archive is kept for transparency."""
    base = os.path.join(CFG["out_dir"], "results"); dest = os.path.join(base, "_archive_csdi_v1")
    marker = os.path.join(dest, "ARCHIVED")
    if os.path.exists(marker):
        return 0
    moved = 0
    for sub in ("cv", "cv_folds", "table8", "tables_2_3", "table4"):
        for p in glob.glob(os.path.join(base, sub, "*CSDI*.json")):
            d = os.path.join(dest, sub); os.makedirs(d, exist_ok=True)
            os.replace(p, os.path.join(d, os.path.basename(p))); moved += 1
    os.makedirs(dest, exist_ok=True)
    with open(marker, "w") as f:
        f.write(f"archived {moved} files on {time.ctime()} (CSDI trained for 100 epochs, batch 32, no LR schedule)\n")
    return moved


# =============================================================================================
# Task list
# =============================================================================================
T5_METHODS = ["LSTM", "ResNet", "ROCKET", "InceptionTime", "TST-mask", "DTIN"]
T8_METHODS = ["Mean", "Median", "KNN", "Linear regression", "CSDI", "DTIN"]
IMP_METHODS = ["Mean", "Median", "KNN", "Linear regression", "BRITS", "SAITS", "CSDI", "DTIN"]
COST = {"CSDI": 100, "SAITS": 15, "BRITS": 15, "DTIN": 5, "DTIN-orig": 5, "Mask only": 5, "Local mean": 5, "KNN": 1}


def build_tasks():
    tasks = []
    def imp(table, dataset, mech, level, seed, method, window=None):
        tid = f"{table}|{dataset}|{mech}|{level}|{seed}|{method}|{window}"
        tasks.append(dict(id=tid, stage="main", kind="imp", table=table, dataset=dataset, mech=mech, level=level,
                          seed=seed, method=method, window=window, cost=COST.get(method, 0.1)))
    seeds = CFG["seeds"]
    if RUN["table5"]:
        for s in seeds:
            for m in T5_METHODS:
                tasks.append(dict(id=f"table5|{m}|{s}", stage="main", kind="down", method=m, seed=s, cost=1000))
    if RUN.get("table5_cv"):
        for s in seeds:
            for m in T5_METHODS:
                tasks.append(dict(id=f"table5_cv|{m}|{s}", stage="main", kind="down_cv", method=m, seed=s, cost=900))
    if RUN.get("orig_table5"):
        for s in seeds:
            for m in T5_FT_METHODS:
                tasks.append(dict(id=f"table5_ft|{m}|{s}", stage="main", kind="down_ft", method=m, seed=s, cost=800))
    if RUN.get("orig_table8"):
        for L in CFG["gap_lengths"]:
            for s in seeds:
                imp("table8", "Siemens", "GAP", L, s, "DTIN-orig")
    if RUN.get("orig_tables_2_3"):
        for ds in ("ECG200", "Siemens"):
            for r in CFG["rates"]:
                for s in seeds:
                    for m in ("Mean", "Median", "KNN", "Linear regression", "DTIN", "DTIN-orig"):
                        imp("tables_2_3", ds, "MCAR", r, s, m)
    if RUN["table8"]:
        for L in CFG["gap_lengths"]:
            for s in seeds:
                for m in T8_METHODS:
                    imp("table8", "Siemens", "GAP", L, s, m)
    if RUN["tables_2_3"]:
        for ds in ("ECG200", "Siemens"):
            for r in CFG["rates"]:
                for s in seeds:
                    for m in IMP_METHODS:
                        imp("tables_2_3", ds, "MCAR", r, s, m)
    if RUN["table4"]:
        for mech in ("MCAR", "MAR", "MNAR"):
            for s in seeds:
                for m in ("BRITS", "SAITS", "CSDI", "DTIN"):
                    imp("table4", "Siemens", mech, CFG["mech_rate"], s, m)
    if RUN["table6"]:
        for ds in ("ECG200", "Siemens"):
            for s in seeds:
                for m in ("Mask only", "Local mean", "DTIN"):
                    imp("table6", ds, "MCAR", 0.4, s, m)
    if RUN["table7"]:
        for w in (5, 10, 20, 30, 40):
            for s in seeds:
                imp("table7", "Siemens", "MCAR", 0.4, s, "DTIN", window=w)
    # CV stage: only for PyPOTS models that the main stage actually needs
    needed = sorted({(t["dataset"], t["method"]) for t in tasks if t.get("method") in PYPOTS_GRID and t["kind"] == "imp"})
    cv = []
    if RUN["cv"]:
        for ds, m in needed:
            for i in range(len(PYPOTS_GRID[m])):
                for k in range(CFG["cv_folds"]):
                    cv.append(dict(id=f"cv|{ds}|{m}|{i}|{k}", stage="cv", kind="cv", dataset=ds, method=m, hp_idx=i,
                                   fold=k, cost=COST[m]))
    seen, uniq = set(), []
    for t in tasks:                                            # drop duplicate tasks (same id)
        if t["id"] not in seen:
            seen.add(t["id"]); uniq.append(t)
    tasks = uniq
    allt = cv + tasks
    # order: cv stage; then downstream tasks; then cheap imputation runs; CSDI last (longest)
    prio = lambda t: 0 if t["kind"] in ("down", "down_cv", "down_ft") else (2 if t.get("method") == "CSDI" else 1)
    allt.sort(key=lambda t: (t["stage"] != "cv", prio(t), -t["cost"]))
    return allt, needed


def task_result_path(t):
    if t["kind"] == "cv":
        return result_path("cv_folds", cv_fold_key(t["dataset"], t["method"], t["hp_idx"], t["fold"]))
    if t["kind"] == "down":
        return result_path("table5", f"{t['method']}_seed{t['seed']}")
    if t["kind"] == "down_cv":
        return result_path("table5_cv", f"{t['method']}_seed{t['seed']}")
    if t["kind"] == "down_ft":
        return result_path("table5_ft", f"{t['method']}_seed{t['seed']}")
    key = f"{t['dataset']}_{t['mech']}_{t['level']}_seed{t['seed']}_{t['method']}" + (f"_w{t['window']}" if t["window"] else "")
    return result_path(t["table"], key)


def run_task(t):
    if t["kind"] == "cv":
        return run_cv_fold(t["dataset"], t["method"], t["hp_idx"], t["fold"])
    if t["kind"] == "down":
        return run_downstream_method(t["method"], t["seed"])
    if t["kind"] == "down_cv":
        return run_downstream_cv(t["method"], t["seed"])
    if t["kind"] == "down_ft":
        return run_downstream_ft(t["method"], t["seed"])
    return run_imputation(t["table"], t["dataset"], t["mech"], t["level"], t["seed"], t["method"], t["window"])


# ------------------------------------------------------------------ claiming (safe across processes)
def _safe(tid):
    return hashlib.md5(tid.encode()).hexdigest()


def lock_path(t):
    d = os.path.join(CFG["out_dir"], "locks"); os.makedirs(d, exist_ok=True)
    return os.path.join(d, _safe(t["id"]) + ".lock")


def error_path(t):
    d = os.path.join(CFG["out_dir"], "errors"); os.makedirs(d, exist_ok=True)
    return os.path.join(d, _safe(t["id"]) + ".txt")


def claim(t, worker):
    try:
        fd = os.open(lock_path(t), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(dict(task=t["id"], worker=worker, pid=os.getpid(), host=socket.gethostname(), start=time.time()), f)
    return True


def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return True                       # cannot tell: treat as alive (never delete a lock we are unsure about)


def clear_stale_locks():
    """Remove locks of tasks that have no result and whose worker process no longer exists."""
    removed = 0
    for p in glob.glob(os.path.join(CFG["out_dir"], "locks", "*.lock")):
        try:
            with open(p) as f:
                info = json.load(f)
        except Exception:
            info = {}
        if info.get("host") == socket.gethostname() and not _pid_alive(info.get("pid", -1)):
            os.remove(p); removed += 1
    return removed


def worker_loop(stage, worker, log=print, exclude=()):
    tasks = [t for t in json.load(open(os.path.join(CFG["out_dir"], "tasks.json")))
             if t["stage"] == stage and t.get("method") not in set(exclude)]
    done = 0
    for t in tasks:
        if os.path.exists(task_result_path(t)) or os.path.exists(error_path(t)):
            continue
        if not claim(t, worker):
            continue
        t0 = time.time(); log(f"[{time.strftime('%H:%M:%S')}] start {t['id']}")
        try:
            run_task(t); done += 1
            log(f"[{time.strftime('%H:%M:%S')}] done  {t['id']} ({time.time() - t0:.0f}s)")
        except Exception:
            with open(error_path(t), "w") as f:
                f.write(t["id"] + "\n" + traceback.format_exc())
            log(f"[{time.strftime('%H:%M:%S')}] ERROR {t['id']}\n{traceback.format_exc()}")
        finally:
            try: os.remove(lock_path(t))
            except FileNotFoundError: pass
    log(f"worker {worker} finished stage '{stage}' ({done} tasks run)")


def progress(stage=None):
    tasks = json.load(open(os.path.join(CFG["out_dir"], "tasks.json")))
    rows = []
    for t in tasks:
        if stage and t["stage"] != stage: continue
        grp = {"cv": "cv", "down": "table5", "down_cv": "table5_cv", "down_ft": "table5_ft"}.get(t["kind"], t.get("table"))
        state = ("done" if os.path.exists(task_result_path(t)) else
                 "error" if os.path.exists(error_path(t)) else
                 "running" if os.path.exists(lock_path(t)) else "waiting")
        rows.append(dict(group=grp, method=t.get("method"), state=state))
    df = pd.DataFrame(rows)
    if df.empty: return df
    return df.pivot_table(index=["group", "method"], columns="state", aggfunc="size", fill_value=0)


def list_errors():
    out = []
    for p in glob.glob(os.path.join(CFG["out_dir"], "errors", "*.txt")):
        with open(p) as f:
            txt = f.read()
        out.append(txt)
    return out


def clear_errors():
    for p in glob.glob(os.path.join(CFG["out_dir"], "errors", "*.txt")):
        os.remove(p)


# =============================================================================================
# Aggregation into manuscript tables (reads only result files)
# =============================================================================================
def _load_table(table):
    rows = []
    for p in glob.glob(os.path.join(CFG["out_dir"], "results", table, "*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def _paired_wilcoxon(xa, xb):
    d = np.asarray(xb) - np.asarray(xa)
    if len(d) == 0 or np.all(d == 0):
        return None, None
    w = wilcoxon(xa, xb, alternative="two-sided")
    return float(w.statistic), float(w.pvalue)


def make_table5(table="table5"):
    rows = [r for r in _load_table(table) if r["seed"] in CFG["seeds"]]
    if not rows: return None, None
    per_run = pd.DataFrame(rows).sort_values(["method", "seed"])
    per_run.to_csv(os.path.join(CFG["out_dir"], f"{table}_per_run_confusion_matrices.csv"), index=False)
    cnt = per_run.groupby("method")["seed"].count()
    t5 = pd.DataFrame({k: summarize(rows, "method", k)["cell"] for k in ("accuracy", "f1", "auroc")})
    t5["n_runs"] = cnt
    order = T5_METHODS + [m for m in T5_FT_METHODS if m not in T5_METHODS]
    t5 = t5.reindex([m for m in order if m in t5.index])
    t5.to_csv(os.path.join(CFG["out_dir"], f"{table}.csv"))
    w = None
    da, db = ("TST-orig", "DTIN-orig") if table == "table5_ft" else ("TST-mask", "DTIN")
    if {da, db} <= set(per_run.method):
        a = per_run[per_run.method == da].set_index("seed")["f1"]
        b = per_run[per_run.method == db].set_index("seed")["f1"]
        common = sorted(set(a.index) & set(b.index))
        stat, p = _paired_wilcoxon(a[common].values, b[common].values)
        w = dict(comparison=f"DodgerLoopGame F1 ({table}): {db} vs {da}", n_pairs=len(common),
                 tstmask_mean=float(a[common].mean()), dtin_mean=float(b[common].mean()),
                 n_dtin_higher=int((b[common].values > a[common].values).sum()),
                 n_ties=int((b[common].values == a[common].values).sum()), statistic=stat, p_two_sided=p)
        save_result("table9", f"DodgerLoopGame_F1_{table}", w)
    return t5, w


def make_imputation_table(table, index, columns, filename, order=None, **filt):
    rows = [r for r in _load_table(table) if r["seed"] in CFG["seeds"]]
    rows = [r for r in rows if all(r.get(k) == v for k, v in filt.items())]
    if not rows: return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CFG["out_dir"], filename.replace(".csv", "_per_seed.csv")), index=False)
    s = summarize(rows, [index, columns])
    t = s["cell"].unstack(); n = s["count"].unstack()
    if order: t = t[[c for c in order if c in t.columns]] if columns == "method" else t.reindex([o for o in order if o in t.index])
    t.to_csv(os.path.join(CFG["out_dir"], filename)); n.to_csv(os.path.join(CFG["out_dir"], filename.replace(".csv", "_n_runs.csv")))
    return t


def make_table9_imputation():
    out = {}
    specs = [("tables_2_3", "ECG200", "MCAR", 0.3), ("tables_2_3", "Siemens", "MCAR", 0.3),
             ("table4", "Siemens", "MCAR", 0.3), ("table4", "Siemens", "MAR", 0.3), ("table4", "Siemens", "MNAR", 0.3)]
    for table, ds, mech, lv in specs:
        rows = [r for r in _load_table(table) if r["dataset"] == ds and r["mech"] == mech and r["level"] == lv and r["seed"] in CFG["seeds"]]
        a = {r["seed"]: r["rmse"] for r in rows if r["method"] == "CSDI"}
        b = {r["seed"]: r["rmse"] for r in rows if r["method"] == "DTIN"}
        common = sorted(set(a) & set(b))
        if len(common) < 2: continue
        xa, xb = [a[s] for s in common], [b[s] for s in common]
        stat, p = _paired_wilcoxon(xa, xb)
        res = dict(comparison=f"{ds} {mech} {int(lv * 100)}% RMSE: DTIN vs CSDI", n_pairs=len(common),
                   csdi_mean=float(np.mean(xa)), dtin_mean=float(np.mean(xb)),
                   n_dtin_lower=int(sum(y < x for x, y in zip(xa, xb))), statistic=stat, p_two_sided=p)
        save_result("table9", f"{table}_{ds}_{mech}", res); out[f"{table}_{ds}_{mech}"] = res
    return out


def build_package():
    out = CFG["out_dir"]; zpath = os.path.join(out, "verification_package.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(os.path.join(out, "**", "*"), recursive=True):
            if os.path.isfile(p) and not p.endswith(".zip") and os.sep + "locks" + os.sep not in p:
                z.write(p, os.path.relpath(p, out))
        for extra in ("dtin_core.py", "dtin_worker.py", "DTIN_parallel.ipynb"):
            if os.path.exists(extra):
                z.write(extra, os.path.join("code", extra))
    return zpath
