"""
dtin_improve.py - phase 1: improve DTIN with as few changes as possible, on the SAME datasets and
benchmarks as the original study (ECG200, Siemens, DodgerLoopGame; Tables 2, 3, 4, 5, 8).

* Development data only: each training set is split into dev-train / val-stop / val-report.
  The original test sets are not touched in this phase.
* The variants change ONE thing at a time; the plan and the selection rule are written to disk first.
* Every job writes one result file.

Worker:  python dtin_improve.py --worker A
"""
import os, sys, json, time, math, glob, socket, hashlib, argparse, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dtin_core as dc

# ===================================================================================== plan
SEEDS = [0, 1, 2]
IMP_CONDITIONS = {
    "ECG200": [("MCAR", r) for r in (0.1, 0.2, 0.3, 0.4, 0.5)],                          # Table 2
    "Siemens": [("MCAR", r) for r in (0.1, 0.2, 0.3, 0.4, 0.5)]                         # Table 3
               + [("MAR", 0.3), ("MNAR", 0.3)]                                          # Table 4
               + [("GAP", L) for L in (10, 20, 30, 40, 50)],                            # Table 8
}
BASELINES = ["Mean", "Median", "KNN", "Linear regression"]          # simple baselines of the original study
TRAIN = dict(lr=1e-3, batch=64, max_steps=10_000, warmup=500, eval_every=250, patience=8,
             point_ratio=0.15, window=20)
VARIANTS = {
    "V0_DTIN":       None,                                                 # exactly as specified in the manuscript
    "V1_converged":  dict(mask_input=False, revin=False, lam=1.0),         # + training to convergence
    "V2_maskinput":  dict(mask_input=True,  revin=False, lam=1.0),         # + observation mask as input
    "V3_revin":      dict(mask_input=True,  revin=True,  lam=1.0),         # + per-sequence normalisation
    "V3_lambda0.3":  dict(mask_input=True,  revin=True,  lam=0.3),         # pseudo-target weight 0.3
    "V3_lambda0":    dict(mask_input=True,  revin=True,  lam=0.0),         # no pseudo-target (ablation)
}
DODGER_STEPS = 2_000              # downstream pre-training: fixed number of steps (only 20 sequences, no val-stop)
ARCH = dict(d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.1)

PLAN = f"""# DTIN phase 1 - minimal changes on the original benchmarks (pre-specified)

Written before any phase-1 result was computed. The original test sets are not used in this phase.

Data: training sets of ECG200, Siemens and DodgerLoopGame. ECG200/Siemens are split into dev-train 80 % /
val-stop 10 % (early stopping) / val-report 10 % (reported); Siemens by batch.
Imputation conditions (same as the original tables): {json.dumps(IMP_CONDITIONS)}.
Baselines: {BASELINES}. Seeds: {SEEDS}. Metric: RMSE at the synthetic missing positions of val-report.

Variants (architecture {json.dumps(ARCH)}; one change at a time):
- V0_DTIN: the manuscript specification (zeros at missing, 15 % point masking, Gaussian pseudo-targets,
  Adam 1e-3, batch 128, 100 epochs).
- V1_converged: V0 trained to convergence: {json.dumps(TRAIN)}; warm-up + cosine; early stopping on val-stop RMSE.
- V2_maskinput: V1 + observation mask as extra input channels.
- V3_revin: V2 + per-sequence, per-variable normalisation of the visible values (RevIN).
- V3_lambda0.3 / V3_lambda0: V3 with pseudo-target weight 0.3 / 0 (lambda = 1 is the original DTIN loss).
Loss: per-sequence weighted MSE over artificially masked (weight 1) and inherently missing (weight lambda)
positions, averaged over sequences (Eq. 10-11 when lambda = 1).

Downstream (DodgerLoopGame, 20 training sequences only): pre-training on the 20 sequences
(V0: manuscript schedule; V1-V3: {DODGER_STEPS} steps, same schedule, no early stopping), then stratified
5-fold CV inside the 20 sequences: (a) linear probe on frozen mean-pooled representations and (b) fine-tuning
of encoder + linear head; Adam 1e-3, batch 32, up to 100 epochs; score = maximum of the fold-averaged
validation binary F1 (Game Day positive). The 138 test sequences are not used.

Selection rule: the variant with the best mean rank over the 17 imputation conditions plus the fine-tuning F1
is selected. It proceeds to the final test evaluation (all original baselines incl. BRITS/SAITS/CSDI,
original test sets, evaluated once) only if its mean val-report RMSE is lower than the best simple baseline
in at least 12 of the 17 imputation conditions. All results are reported.
"""


def out(*p):
    d = os.path.join(dc.CFG["out_dir"], "improve", *p[:-1]); os.makedirs(d, exist_ok=True)
    return os.path.join(d, p[-1])


def write_plan():
    p = out("PLAN_phase1.md")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(PLAN + f"\nWritten: {time.ctime()}\n")
    return p


# ===================================================================================== splits / masks
_SPLITS = {}


def dev_split(name):
    if name in _SPLITS: return _SPLITS[name]
    X = dc.get_data(name)["Xtr"]; n = len(X); rng = np.random.default_rng(dc.stable_seed("devsplit", name))
    if name == "Siemens":
        w = pd.read_csv(os.path.join(dc.CFG["out_dir"], "siemens_windows.csv"))
        groups = w[w.split == "train"].batch.to_numpy(); assert len(groups) == n
        val_b, cnt = [], 0
        for b in rng.permutation(np.unique(groups)):
            if cnt >= 0.2 * n: break
            val_b.append(b); cnt += int((groups == b).sum())
        half = set(val_b[: len(val_b) // 2])
        stop = np.flatnonzero(np.isin(groups, list(half)))
        report = np.flatnonzero(np.isin(groups, [b for b in val_b if b not in half]))
        train = np.flatnonzero(~np.isin(groups, val_b))
    else:
        perm = rng.permutation(n); k = int(round(0.1 * n))
        stop, report, train = np.sort(perm[:k]), np.sort(perm[k:2 * k]), np.sort(perm[2 * k:])
    _SPLITS[name] = dict(train=train, stop=stop, report=report)
    return _SPLITS[name]


def dev_mask(name, X, mech, level, seed, part):
    return dc.make_mask(f"dev-{name}", X, mech, level, seed, part)       # same generators as the original tables


# ===================================================================================== model
class DTINv2(nn.Module):
    """DTIN encoder with two optional changes: observation mask as input, per-sequence normalisation."""
    def __init__(self, D, mask_input, revin, max_len=2048):
        super().__init__()
        self.mask_input, self.revin = mask_input, revin; dm = ARCH["d_model"]
        self.inp = nn.Linear(D * (2 if mask_input else 1), dm)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dm, 2).float() * (-math.log(10000.0) / dm))
        pe = torch.zeros(max_len, dm); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(dm, ARCH["n_heads"], ARCH["d_ff"], ARCH["dropout"], batch_first=True)
        self.enc = nn.TransformerEncoder(layer, ARCH["n_layers"], enable_nested_tensor=False)
        self.pred = nn.Linear(dm, D)

    def _norm(self, x, v):
        cnt = v.sum(1, keepdim=True)
        mu = (x * v).sum(1, keepdim=True) / cnt.clamp(min=1)
        sd = torch.sqrt((((x - mu) * v) ** 2).sum(1, keepdim=True) / cnt.clamp(min=1) + 1e-5)
        mu = torch.where(cnt > 0, mu, torch.zeros_like(mu)); sd = torch.where(cnt > 1, sd, torch.ones_like(sd))
        return (x - mu) / sd * v, mu, sd

    def encode(self, x, v):
        x = x * v
        if self.revin: x, _, _ = self._norm(x, v)
        h = torch.cat([x, v], -1) if self.mask_input else x
        return self.enc(self.inp(h) + self.pe[: x.shape[1]])

    def forward(self, x, v):
        x = x * v
        if self.revin:
            xn, mu, sd = self._norm(x, v)
            h = torch.cat([xn, v], -1) if self.mask_input else xn
            return self.pred(self.enc(self.inp(h) + self.pe[: x.shape[1]])) * sd + mu
        h = torch.cat([x, v], -1) if self.mask_input else x
        return self.pred(self.enc(self.inp(h) + self.pe[: x.shape[1]]))


@torch.no_grad()
def impute(model, X_nan, bs=256):
    model.eval(); res = []
    for b in range(0, len(X_nan), bs):
        xb = X_nan[b:b + bs]
        x = torch.tensor(np.nan_to_num(xb), dtype=torch.float32, device=dc.DEVICE)
        v = torch.tensor(~np.isnan(xb), dtype=torch.float32, device=dc.DEVICE)
        res.append(model(x, v).cpu().numpy())
    pred = np.concatenate(res).astype(np.float64)
    return np.where(np.isnan(X_nan), pred, X_nan)


def train_variant(Xtr_nan, var, seed, Xstop_true=None, Mstop=None, max_steps=None):
    """V1-V3 training. With Xstop_true/Mstop: early stopping on val-stop RMSE; otherwise fixed steps."""
    cfg = dict(TRAIN); max_steps = max_steps or cfg["max_steps"]
    dc.set_seed(seed); gen = torch.Generator(device=dc.DEVICE); gen.manual_seed(seed)
    N, T, D = Xtr_nan.shape
    X = torch.tensor(np.nan_to_num(Xtr_nan), dtype=torch.float32, device=dc.DEVICE)
    O = torch.tensor(~np.isnan(Xtr_nan), dtype=torch.float32, device=dc.DEVICE)
    mu_tr, var_tr = dc.train_stats(Xtr_nan)
    model = DTINv2(D, var["mask_input"], var["revin"]).to(dc.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps)))
    Xstop_nan = dc.apply_mask(Xstop_true, Mstop) if Xstop_true is not None else None
    best, best_state, best_step, bad, curve, step = np.inf, None, 0, 0, [], 0
    for step in range(1, max_steps + 1):
        model.train()
        idx = torch.randint(N, (min(cfg["batch"], N),), generator=gen, device=dc.DEVICE)
        x, o = X[idx], O[idx]
        a = (torch.rand(o.shape, generator=gen, device=dc.DEVICE) < cfg["point_ratio"]).float() * o
        v = o * (1 - a)
        xhat = model(x, v)
        w = a
        target = x
        if var["lam"] > 0:
            mean, vr = dc.local_stats(x, v, cfg["window"], mu_tr, var_tr)
            z = mean + vr.sqrt() * torch.randn(o.shape, generator=gen, device=dc.DEVICE)
            target = torch.where(o.bool(), x, z); w = a + var["lam"] * (1 - o)
        cnt = w.sum((1, 2))
        loss = ((((xhat - target) ** 2) * w).sum((1, 2)) / cnt.clamp(min=1e-8))[cnt > 0].mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if Xstop_nan is not None and step % cfg["eval_every"] == 0:
            r = dc.rmse_at(Xstop_true, impute(model, Xstop_nan), Mstop); curve.append((step, r))
            if r < best - 1e-6:
                best, best_step, bad = r, step, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model, dict(best_stop_rmse=None if best == np.inf else best, best_step=best_step, last_step=step, curve=curve)


# ===================================================================================== downstream (dev only)
class FTWrap(nn.Module):
    """Encoder + linear head on mean-pooled outputs. Input: concat([x, v]) along the variable axis."""
    def __init__(self, enc, D):
        super().__init__(); self.enc, self.D = enc, D; self.fc = nn.Linear(ARCH["d_model"], 2)

    def forward(self, xa):
        x, v = xa[..., : self.D], xa[..., self.D:]
        h = self.enc.encode(x, v) if isinstance(self.enc, DTINv2) else self.enc.encode(x * v)
        return self.fc(h.mean(dim=1))


def cv_score(make, Xa, y, folds, seed):
    curves, accs = [], []
    for k, (tr, va) in enumerate(folds):
        p = dc._train_epochs(make, Xa[tr], y[tr], seed + 100 * k, dc.CFG["head_epochs"], Xval=Xa[va])[1]
        curves.append([dc.binary_f1(dc.confusion(y[va], (pe >= 0.5).astype(int))) for pe in p])
        accs.append([float(((pe >= 0.5).astype(int) == y[va]).mean()) for pe in p])
    f1, acc = np.mean(curves, 0), np.mean(accs, 0); e = int(np.argmax(f1))
    return dict(f1=float(f1[e]), acc=float(acc[e]), epoch=e + 1)


def run_downstream(variant, seed):
    from sklearn.model_selection import StratifiedKFold
    import copy
    DOD = dc.get_data("DodgerLoopGame"); X = DOD["Xtr"]
    y = (DOD["ytr"] == dc.CFG["dodger_positive_label"]).astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=dc.stable_seed("devfold", seed) % (2**31 - 1))
                 .split(np.zeros(len(y)), y))
    s = dc.stable_seed("dev-down", variant, seed); D = X.shape[2]
    if VARIANTS[variant] is None:
        enc = dc.train_ssl(X, "dtin", s); info = {}
    else:
        enc, info = train_variant(X, VARIANTS[variant], s, max_steps=DODGER_STEPS)
    xs = np.nan_to_num(X); vs = (~np.isnan(X)).astype(float)
    with torch.no_grad():
        xt = torch.tensor(xs, dtype=torch.float32, device=dc.DEVICE); vt = torch.tensor(vs, dtype=torch.float32, device=dc.DEVICE)
        h = enc.encode(xt, vt) if isinstance(enc, DTINv2) else enc.encode(xt * vt)
        Z = h.mean(1).cpu().numpy()
    probe = cv_score(lambda: dc.Head(Z.shape[1]).to(dc.DEVICE), Z, y, folds, s)
    state = {k: t.detach().clone() for k, t in enc.state_dict().items()}
    def make_ft():
        e = copy.deepcopy(enc); e.load_state_dict(state); return FTWrap(e, D).to(dc.DEVICE)
    ft = cv_score(make_ft, np.concatenate([xs, vs], -1), y, folds, s)
    del enc; torch.cuda.empty_cache()
    return dict(probe_f1=probe["f1"], probe_acc=probe["acc"], probe_epoch=probe["epoch"],
                ft_f1=ft["f1"], ft_acc=ft["acc"], ft_epoch=ft["epoch"], pretrain=info)


# ===================================================================================== jobs
def job_list():
    jobs = []
    for s in SEEDS:
        for v in VARIANTS:
            jobs.append(dict(kind="down", dataset="DodgerLoopGame", cond="downstream", level=0, seed=s, method=v,
                             id=f"down|{v}|{s}", cost=3))
    for ds, conds in IMP_CONDITIONS.items():
        for mech, lv in conds:
            for s in SEEDS:
                for m in BASELINES + list(VARIANTS):
                    jobs.append(dict(kind="imp", dataset=ds, cond=mech, level=lv, seed=s, method=m,
                                     id=f"{ds}|{mech}|{lv}|{s}|{m}", cost=2 if m in VARIANTS else 1))
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


def write_jobs():
    p = out("jobs_phase1.json")
    with open(p, "w") as f: json.dump(job_list(), f, indent=1)
    return p


def _key(j): return hashlib.md5(j["id"].encode()).hexdigest()
def res_path(j): return out("phase1", "results", _key(j) + ".json")
def lock_path(j): return out("phase1", "locks", _key(j) + ".lock")
def err_path(j): return out("phase1", "errors", _key(j) + ".txt")
def folder(sub): return os.path.dirname(out("phase1", sub, "x"))


def run_job(j):
    t0 = time.time()
    if j["kind"] == "down":
        info = run_downstream(j["method"], j["seed"]); res = dict(j, **info)
    else:
        X = dc.get_data(j["dataset"])["Xtr"]; sp = dev_split(j["dataset"])
        Xtr, Xstop, Xrep = X[sp["train"]], X[sp["stop"]], X[sp["report"]]
        Mtr = dev_mask(j["dataset"], Xtr, j["cond"], j["level"], j["seed"], "train")
        Mstop = dev_mask(j["dataset"], Xstop, j["cond"], j["level"], j["seed"], "stop")
        Mrep = dev_mask(j["dataset"], Xrep, j["cond"], j["level"], j["seed"], "report")
        Xtr_m, Xrep_m = dc.apply_mask(Xtr, Mtr), dc.apply_mask(Xrep, Mrep)
        m, info = j["method"], {}
        if m == "Mean": imp = dc.impute_seq_stat(Xrep_m, np.nanmean)
        elif m == "Median": imp = dc.impute_seq_stat(Xrep_m, np.nanmedian)
        elif m == "KNN": imp = dc.impute_knn(Xtr_m, Xrep_m, dc.CFG["knn_k"])
        elif m == "Linear regression": imp = dc.impute_lr_time(Xrep_m)
        elif VARIANTS[m] is None:
            model = dc.train_ssl(Xtr_m, "dtin", dc.stable_seed("dev", j["id"])); imp = dc.ssl_impute(model, Xrep_m); del model
        else:
            model, info = train_variant(Xtr_m, VARIANTS[m], dc.stable_seed("dev", j["id"]), Xstop, Mstop)
            imp = impute(model, Xrep_m); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        res = dict(j, rmse=dc.rmse_at(Xrep, imp, Mrep), n_removed=int(Mrep.sum()), **info)
    res["seconds"] = round(time.time() - t0, 1)
    tmp = res_path(j) + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f: json.dump(res, f, default=float)
    os.replace(tmp, res_path(j))
    return res


def worker(name, log=print):
    for j in json.load(open(out("jobs_phase1.json"))):
        if os.path.exists(res_path(j)) or os.path.exists(err_path(j)): continue
        try:
            fd = os.open(lock_path(j), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(dict(pid=os.getpid(), host=socket.gethostname(), worker=name), f)
        t0 = time.time(); log(f"[{time.strftime('%H:%M:%S')}] start {j['id']}")
        try:
            r = run_job(j)
            msg = f"rmse={r['rmse']:.4f}" if "rmse" in r else f"ft_f1={r['ft_f1']:.3f} probe_f1={r['probe_f1']:.3f}"
            log(f"[{time.strftime('%H:%M:%S')}] done  {j['id']} {msg} ({time.time() - t0:.0f}s)")
        except Exception:
            with open(err_path(j), "w") as f: f.write(j["id"] + "\n" + traceback.format_exc())
            log(f"ERROR {j['id']}\n{traceback.format_exc()}")
        finally:
            try: os.remove(lock_path(j))
            except FileNotFoundError: pass
    log(f"worker {name} finished")


def clear_stale_locks():
    import psutil
    n = 0
    for p in glob.glob(os.path.join(folder("locks"), "*.lock")):
        try: info = json.load(open(p))
        except Exception: info = {}
        if info.get("host") == socket.gethostname() and not psutil.pid_exists(info.get("pid", -1)):
            os.remove(p); n += 1
    return n


def progress():
    jobs = json.load(open(out("jobs_phase1.json")))
    st = ["done" if os.path.exists(res_path(j)) else "error" if os.path.exists(err_path(j)) else
          "running" if os.path.exists(lock_path(j)) else "waiting" for j in jobs]
    df = pd.DataFrame(dict(dataset=[j["dataset"] for j in jobs], method=[j["method"] for j in jobs], state=st))
    return df.pivot_table(index=["dataset", "method"], columns="state", aggfunc="size", fill_value=0)


# ===================================================================================== summaries
def results():
    return pd.DataFrame([json.load(open(p)) for p in glob.glob(os.path.join(folder("results"), "*.json"))])


def imputation_table(dataset):
    df = results()
    if df.empty or "rmse" not in df: return None
    df = df[(df.kind == "imp") & (df.dataset == dataset)].copy()
    if df.empty: return None
    df["condition"] = df.cond + "-" + df.level.map(lambda l: f"{float(l):g}")
    g = df.groupby(["method", "condition"]).rmse
    t = (g.mean().map("{:.4f}".format) + " ± " + g.std(ddof=1).map("{:.4f}".format)).unstack()
    rows = [m for m in BASELINES + list(VARIANTS) if m in t.index]
    cols = [f"{c}-{float(l):g}" for c, l in IMP_CONDITIONS[dataset] if f"{c}-{float(l):g}" in t.columns]
    return t.loc[rows, cols]


def downstream_table():
    df = results()
    if df.empty or "ft_f1" not in df: return None
    df = df[df.kind == "down"]
    g = df.groupby("method")[["probe_f1", "ft_f1"]]
    fmt = lambda d: d.apply(lambda c: c.map("{:.4f}".format))
    t = fmt(g.mean()) + " ± " + fmt(g.std(ddof=1))
    return t.reindex([v for v in VARIANTS if v in t.index])


def selection():
    """Mean rank of the variants (17 imputation conditions + fine-tuning F1) and the phase-2 gate."""
    df = results()
    if df.empty: return None, None
    imp = df[df.kind == "imp"]
    mean = imp.groupby(["method", "dataset", "cond", "level"]).rmse.mean().unstack(["dataset", "cond", "level"])
    var_rows = [v for v in VARIANTS if v in mean.index]
    ranks = mean.loc[var_rows].rank(axis=0)                                   # lower RMSE = better rank
    if "ft_f1" in df:
        ft = df[df.kind == "down"].groupby("method").ft_f1.mean().reindex(var_rows)
        ranks[("DodgerLoopGame", "downstream", "ft_f1")] = (-ft).rank()
    summary = pd.DataFrame(dict(mean_rank=ranks.mean(axis=1), n_criteria=ranks.notna().sum(axis=1))).sort_values("mean_rank")
    best_base = mean.loc[[b for b in BASELINES if b in mean.index]].min()
    gate = {v: int((mean.loc[v] < best_base).sum()) for v in var_rows}
    summary["wins_vs_best_baseline"] = pd.Series(gate)
    summary["of"] = int(best_base.notna().sum())
    selected = summary.index[0]
    return summary, dict(selected=selected, wins=gate[selected], needed=12,
                         proceed_to_final_test=gate[selected] >= 12)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("dtin_outputs", "improve", "config_improve.json"))
    ap.add_argument("--worker", default="I0")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads > 0: torch.set_num_threads(a.threads)
    dc.init(a.config)
    logf = open(out("phase1", "logs", f"worker_{a.worker}.log"), "a", buffering=1, encoding="utf-8")
    def log(m): print(m, flush=True); logf.write(m + "\n")
    worker(a.worker, log)
