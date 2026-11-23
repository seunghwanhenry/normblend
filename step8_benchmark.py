"""
step8_benchmark.py - Step 8: benchmark extension of NormBlend (written 2026-11-19, after Steps 1-7; descriptive).

What it adds
  E1  more imputers: BRITS, TimesNet and ImputeFormer (PyPOTS) next to KNN, SAITS and the mask-aware transformer
  E2  a third dataset with changing operating conditions: C-MAPSS FD002 + FD004 (six operating conditions; CMAPSS_M)
  E3  cross-domain test: every model trained on C-MAPSS FD001 + FD003 also imputes the CMAPSS_M test windows (same
      windows and masks as the CMAPSS_M jobs), without retraining or recalibration
  E4  normalization handled inside one model: SAITS and ImputeFormer with the window statistics as extra input
      channels ("G+stats"); Non-stationary Transformer; RevIN-SCINet; TimesNet with its non-stationary normalization
  E5  missing-rate sweep: MCAR 10%, 50% and 90% (30% and 70% are part of E1)
Protocol: the clean protocol of Step 7 - global statistics from the observed values of the training units only; the
validation units are split by unit into an early-stopping half and a calibration half that estimates the NormBlend
weights. Seeds 0-4. For C-MAPSS and MetroPT the splits, masks and model seeds are those of Steps 5 and 7, so KNN,
SAITS and the transformer reproduce Step 7 (checked by `check` and in the report). The MetroPT results of linear
regression, KNN, SAITS and the transformer under the four Step 5 conditions are read from step7_outputs (not rerun).
Hyperparameters of the added models are fixed from their original studies or PyPOTS defaults and are not tuned.

Commands (from the DTIN folder):
  python step8_benchmark.py plan        # writes step8_outputs/PLAN_step8.md (once)
  python step8_benchmark.py prepare     # builds the C-MAPSS FD002 + FD004 windows from data/cmapss
  python step8_benchmark.py check       # CPU, about 1 min: KNN on C-MAPSS must equal Step 7
  python step8_benchmark.py smoke       # every model and variant once on a tiny subset (a few minutes)
  python step8_benchmark.py run [--worker k --n-workers K]
  python step8_benchmark.py status
  python step8_benchmark.py report
"""
import argparse, glob, hashlib, importlib.util, inspect, json, os, platform, sys, time, traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


s5 = _load("step5_confirm", "step5_confirm.py")
s7 = _load("step7_robustness", "step7_robustness.py")

OUT = "step8_outputs"
STEP7_OUT = "step7_outputs"
SEEDS = [0, 1, 2, 3, 4]
FIRST_SEEDS = [0, 1, 2]                      # run first, so that a complete three-seed design exists early
DATASETS = ["CMAPSS", "MetroPT", "CMAPSS_M"]
CONDS = list(s5.CONDITIONS)                  # MCAR 30%, MCAR 70%, MNAR 30%, gaps 40%
SWEEP = [("MCAR", 0.1), ("MCAR", 0.5), ("MCAR", 0.9)]
SWEEP_DS = ["CMAPSS", "MetroPT"]
SWEEP_MODELS = ["KNN", "SAITS", "ImputeFormer"]
BACKBONES = ["KNN", "SAITS", "Transformer", "BRITS", "TimesNet", "ImputeFormer"]
NEW_BACKBONES = ["BRITS", "TimesNet", "ImputeFormer"]
STEP7_REUSE = {"Linear regression", "KNN", "SAITS", "Transformer"}     # MetroPT, Step 5 conditions only
GSTAT = ["SAITS", "ImputeFormer"]                                         # single model + window statistics
OWN = ["NSTransformer", "RevIN-SCINet", "TimesNet-NS"]                   # normalization inside the model
NORMS = ["global", "seq"]
CMAPSS_M_SUBSETS = ["FD002", "FD004"]
s5.GAP_L.setdefault("CMAPSS_M", s5.GAP_L["CMAPSS"]); s5.T_LEN.setdefault("CMAPSS_M", s5.T_LEN["CMAPSS"])

TRAIN = dict(epochs=300, patience=20, batch_size=32, lr=1e-3, weight_decay=0.0)    # as SAITS in Steps 5-7
PYPOTS = {   # class name, hyperparameters (original studies / PyPOTS defaults; not tuned)
    "BRITS": ("BRITS", dict(rnn_hidden_size=64)),
    "TimesNet": ("TimesNet", dict(n_layers=2, top_k=3, d_model=64, d_ffn=64, n_kernels=6, dropout=0.1)),
    "ImputeFormer": ("ImputeFormer", dict(n_layers=3, d_input_embed=32, d_learnable_embed=96, d_proj=8, d_ffn=256,
                                          n_temporal_heads=4, dropout=0.1)),
    "NSTransformer": ("NonstationaryTransformer", dict(n_layers=2, d_model=128, n_heads=4, d_ffn=256,
                                                      d_projector_hidden=[128, 128], n_projector_hidden_layers=2,
                                                      dropout=0.1)),
    "RevIN-SCINet": ("RevIN_SCINet", dict(n_stacks=1, n_levels=1, n_groups=1, n_decoder_layers=1, d_hidden=1,
                                          dropout=0.1)),
    "TimesNet-NS": ("TimesNet", dict(n_layers=2, top_k=3, d_model=64, d_ffn=64, n_kernels=6, dropout=0.1,
                                     apply_nonstationary_norm=True)),
}
FAST = dict(on=False)                         # smoke test only: tiny subsets and budgets


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def script_sha():
    return sha_file(os.path.abspath(__file__))


PLAN = """# Step 8 - benchmark extension of NormBlend (descriptive)

Written on 19 November 2026, after Steps 1-7 and the first full draft of the paper, before any Step 8 job was run.
Script step8_benchmark.py, SHA-256 {sha}; it imports the unchanged step5_confirm.py ({sha5}) and
step7_robustness.py ({sha7}). Step 8 has no confirmatory decision rule; every result is reported.

## Why
A pre-submission reading of the draft asked for (1) more imputers, (2) data with changing operating conditions,
(3) a test under a real distribution shift, (4) comparisons with methods that handle normalization inside a single
model, and (5) results over the missing rate.

## Design
* Protocol: Step 7 clean protocol (observed-only training statistics; calibration half of the validation units for
  the NormBlend weights; early-stopping half for checkpoints). NormBlend as frozen in Step 4 (per-variable weights,
  window guard with kappa = 1). Seeds {seeds}; seeds {first} are run first.
* Datasets: C-MAPSS FD001 + FD003 and MetroPT-3 (Steps 5 and 7) and C-MAPSS FD002 + FD004 (new; same 14 sensors,
  30-cycle windows, units = engines, 30% test, 10% validation).
* Imputers with global (G) and per-sequence (S) normalization: KNN, SAITS, the transformer, BRITS, TimesNet,
  ImputeFormer. Hyperparameters of the added models: {hp}. Training budget of the PyPOTS models: {train}.
* Single-model alternatives (one training): SAITS and ImputeFormer with the visible mean and log standard deviation
  of each variable as extra input channels (G+stats); Non-stationary Transformer; RevIN-SCINet; TimesNet with its
  non-stationary normalization (TimesNet-NS).
* Cross-domain: the C-MAPSS FD001 + FD003 models impute the FD002 + FD004 test windows of the same seed with the
  masks of the FD002 + FD004 jobs; errors on the FD002 + FD004 reporting scale.
* Missing-rate sweep: MCAR 10%, 50%, 90% for KNN, SAITS and ImputeFormer on C-MAPSS and MetroPT.
* Reused: the MetroPT results of linear regression, KNN, SAITS and the transformer under the Step 5 conditions are
  read from step7_outputs. C-MAPSS is rerun (needed for the cross-domain test) and must reproduce Step 7.
* Reported: RMSE and MAE at the removed positions; G, S, NormBlend, NormBlend without guard, pooled and equal
  weights, validation-selected normalization, oracle; weights; guard activation; run time and parameter counts.
"""


# ------------------------------------------------------------------------------------------- data
def load_cmapss_m():
    W, meta = [], []
    for k, sub in enumerate(CMAPSS_M_SUBSETS, 1):
        a = pd.read_csv(s5.find(s5.CMAPSS_DIR, f"train_{sub}.txt"), sep=r"\s+", header=None).to_numpy(np.float64)
        assert a.shape[1] == 26, a.shape
        for u in np.unique(a[:, 0]).astype(int):
            r = a[a[:, 0] == u]; r = r[np.argsort(r[:, 1])]
            assert (np.diff(r[:, 1]) == 1).all(), (sub, u)
            life = int(r[-1, 1]); X = r[:, [4 + s for s in s5.CMAPSS_SENSORS]]; L = s5.T_LEN["CMAPSS"]
            op = np.round(r[:, 2:5], 0)                     # operating settings -> condition label
            for i in range(len(r) // L):
                c0, c1 = int(r[i * L, 1]), int(r[(i + 1) * L - 1, 1])
                n_op = len({tuple(x) for x in op[i * L:(i + 1) * L]})
                W.append(X[i * L:(i + 1) * L])
                meta.append(dict(unit=1000 * (k - 1) + u, subset=sub, start=c0, end=c1, life=life, n_op=n_op))
    return np.stack(W), pd.DataFrame(meta), [f"s{s}" for s in s5.CMAPSS_SENSORS]


def cache8(ds):
    return os.path.join(OUT, "cache", f"{ds}.npz"), os.path.join(OUT, "cache", f"{ds}_windows.csv")


def prepare():
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    X, meta, cols = load_cmapss_m()
    p, q = cache8("CMAPSS_M"); np.savez_compressed(p, X=X, cols=np.array(cols)); meta.to_csv(q, index=False)
    print(f"CMAPSS_M: windows {X.shape[0]}, length {X.shape[1]}, variables {X.shape[2]}, units {meta.unit.nunique()}, "
          f"non-finite values {int((~np.isfinite(X)).sum())}, operating conditions per window: "
          f"median {meta.n_op.median():.0f}, range {meta.n_op.min()}-{meta.n_op.max()}")


_C8 = {}


def setting8(dc, ds, seed):
    if ds in s5.DATASETS: return s5.setting(dc, ds, seed)
    if (ds, seed) in _C8: return _C8[(ds, seed)]
    if ds not in _C8:
        p, q = cache8(ds)
        if not os.path.exists(p): sys.exit("run first: python step8_benchmark.py prepare")
        _C8[ds] = (np.load(p)["X"], pd.read_csv(q))
    X, meta = _C8[ds]
    itr, ist, ite = s5.split_idx(ds, meta, seed); itrain = np.r_[itr, ist]
    Xa, Xte, sc = dc.standardize(X[itrain], X[ite])
    S = dict(Xa=Xa, Xte=Xte, tr=np.arange(len(itr)), st=np.arange(len(itr), len(itrain)), itrain=itrain, ite=ite,
             meta=meta, scale=sc)
    _C8[(ds, seed)] = S
    return S


def scale_of(ds, seed, S):
    """mean and sd that map raw values to the reporting scale of (ds, seed)"""
    if "scale" in S: return np.array(S["scale"]["mean"]), np.array(S["scale"]["std"])
    X, _ = s5._CACHE[ds]
    mu = np.nanmean(X[S["itrain"]], axis=(0, 1)); sd = np.nanstd(X[S["itrain"]], axis=(0, 1))
    return mu, np.where(sd > 0, sd, 1.0)


def parts(dc, ds, mech, lv, seed):
    S = setting8(dc, ds, seed); Mtr, Mte = s5.masks(dc, ds, S, mech, lv, seed)
    es, cal = s7.es_cal(ds, S, seed); mu, sd = s7.clean_stats(S["Xa"], Mtr, S["tr"])
    return S, Mtr, Mte, es, cal, mu, sd


def cross_domain(dc, mech, lv, seed):
    """FD002 + FD004 test windows of this seed on the C-MAPSS (FD001 + FD003) reporting scale, with the masks of the
    CMAPSS_M job; also the arrays needed to report errors on the CMAPSS_M scale"""
    Sc = s5.setting(dc, "CMAPSS", seed); mu_c, sd_c = scale_of("CMAPSS", seed, Sc)
    Sm = setting8(dc, "CMAPSS_M", seed); _, Mte_m = s5.masks(dc, "CMAPSS_M", Sm, mech, lv, seed)
    mu_m, sd_m = scale_of("CMAPSS_M", seed, Sm)
    raw = Sm["Xte"] * sd_m + mu_m
    return (raw - mu_c) / sd_c, Mte_m, Sm, (mu_c, sd_c, mu_m, sd_m)


# ------------------------------------------------------------------------------------------- models
def win_stats(Z):
    """visible mean and log sd per window and variable, repeated over time: (N, T, 2D)"""
    m, s = s5.seq_stats(Z)
    return np.concatenate([np.repeat(m, Z.shape[1], 1), np.repeat(np.log(s), Z.shape[1], 1)], axis=2)


def fit_pypots(dc, method, Xtrain_nan, Xval_true, Mval, Xtest_nan, seed):
    import torch
    import pypots.imputation as PI
    from pypots.optim import Adam
    cls_name, hp = PYPOTS[method]; cls = getattr(PI, cls_name); T, D = Xtrain_nan.shape[1:]
    tr = dict(TRAIN, epochs=2, patience=None) if FAST["on"] else TRAIN
    dc.set_seed(seed)
    kw = dict(hp, n_steps=T, n_features=D, batch_size=tr["batch_size"], epochs=tr["epochs"], patience=tr["patience"],
              optimizer=Adam(lr=tr["lr"], weight_decay=tr["weight_decay"]), device=dc.DEVICE, saving_path=None,
              model_saving_strategy=None, verbose=False)
    ok = inspect.signature(cls.__init__).parameters
    model = cls(**{k: v for k, v in kw.items() if k in ok})
    Xval_nan = Xval_true.copy(); Xval_nan[Mval] = np.nan
    t0 = time.time()
    model.fit({"X": Xtrain_nan.astype(np.float32)},
              val_set={"X": Xval_nan.astype(np.float32), "X_ori": Xval_true.astype(np.float32)})
    t1 = time.time()
    r = np.asarray(model.impute({"X": Xtest_nan.astype(np.float32)}), dtype=np.float64)
    if r.ndim == 4: r = np.median(r, axis=1)
    t2 = time.time()
    n_par = int(sum(p.numel() for p in model.model.parameters()))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.where(np.isnan(Xtest_nan), r, Xtest_nan), dict(train_seconds=t1 - t0, infer_seconds=t2 - t1, n_params=n_par)


def fit_impute(dc, v2, ev, method, Ztr, Zes_true, Mes, Ztests, s):
    """train on Ztr (masked), early-stop on (Zes_true, Mes), impute every array of Ztests (dict)"""
    names = list(Ztests); sizes = [len(Ztests[n]) for n in names]; info = {}
    if method == "SAITS":
        kw = dict(train_override=dict(epochs=2, patience=None)) if FAST["on"] else {}
        t0 = time.time()
        Zall, val = ev.pypots_fit_impute("SAITS", ev.select_saits()["cfg"], Ztr, Zes_true, Mes,
                                         np.concatenate([Ztests[n] for n in names]), s, **kw)
        info.update(seconds_fit_impute=time.time() - t0, val_es_rmse_model_space=val)
        return dict(zip(names, np.split(Zall, np.cumsum(sizes)[:-1]))), info
    if method == "Transformer":
        cfg = dict(s5.TRANSFORMER, **(dict(max_steps=4, eval_every=2, warmup=1) if FAST["on"] else {}))
        t0 = time.time(); model = v2.DTINv2Imputer(v2.imputation_config(**cfg), seed=s).fit(Ztr, Zes_true, Mes)
        t1 = time.time(); out = {n: model.impute(z) for n, z in Ztests.items()}
        info.update(train_seconds=t1 - t0, infer_seconds=time.time() - t1,
                    n_params=int(sum(p.numel() for p in model.model.parameters())))
        del model
        return out, info
    Zall, info = fit_pypots(dc, method, Ztr, Zes_true, Mes, np.concatenate([Ztests[n] for n in names]), s)
    return dict(zip(names, np.split(Zall, np.cumsum(sizes)[:-1]))), info


# ------------------------------------------------------------------------------------------- jobs
def _job(ds, mech, lv, seed, method, variant, tier):
    j = dict(dataset=ds, mech=mech, level=lv, seed=seed, method=method, norm=variant, tier=tier)
    j["id"] = f"{ds}|{mech}|{lv}|{seed}|{method}|{variant}"
    return j


def setting_jobs(ds, mech, lv, seed, tier):
    out = []
    if tier == 1:
        models = BACKBONES if ds in ("CMAPSS", "CMAPSS_M") else NEW_BACKBONES
        if ds != "MetroPT": out.append(_job(ds, mech, lv, seed, "Linear regression", "global", tier))
        out += [_job(ds, mech, lv, seed, m, n, tier) for m in models for n in NORMS]
    elif tier == 2:
        out += [_job(ds, mech, lv, seed, m, "gstat", tier) for m in GSTAT]
        out += [_job(ds, mech, lv, seed, m, "own", tier) for m in OWN]
    else:
        out.append(_job(ds, mech, lv, seed, "Linear regression", "global", tier))
        out += [_job(ds, mech, lv, seed, m, n, tier) for m in SWEEP_MODELS for n in NORMS]
    return out


def job_list():
    jobs = []
    for seeds in (FIRST_SEEDS, [s for s in SEEDS if s not in FIRST_SEEDS]):
        for tier in (1, 2, 3):
            for seed in seeds:
                for (mech, lv) in (SWEEP if tier == 3 else CONDS):
                    for ds in (SWEEP_DS if tier == 3 else DATASETS):
                        jobs += setting_jobs(ds, mech, lv, seed, tier)
    return jobs


def res_path(j, ext, out=None):
    return os.path.join(out or OUT, "results", hashlib.md5(j["id"].encode()).hexdigest() + ext)


def run_job(j, dc, v2, ev, out=None):
    ds, mech, lv, seed, method, variant = j["dataset"], j["mech"], j["level"], j["seed"], j["method"], j["norm"]
    S, Mtr, Mte, es, cal, mu, sd = parts(dc, ds, mech, lv, seed)
    Xa, Xte, tr = S["Xa"], S["Xte"], S["tr"]
    if FAST["on"]:
        tr, es, cal, Xte, Mte = tr[:64], es[:16], cal[:16], Xte[:24], Mte[:24]
    fw = lambda Z: (Z - mu) / sd
    bw = lambda Z: Z * sd + mu
    Za, Zte = fw(Xa), fw(Xte)
    Za_m, Zte_m = dc.apply_mask(Za, Mtr), dc.apply_mask(Zte, Mte)
    inputs = {"te": Zte_m, "cal": Za_m[cal]}
    masks = {"te": Mte, "cal": Mtr[cal]}
    if ds == "CMAPSS" and j["tier"] in (1, 2):
        Xx, Mx, _, _ = cross_domain(dc, mech, lv, seed)
        if FAST["on"]: Xx, Mx = Xx[:24], Mx[:24]
        inputs["xd"] = dc.apply_mask(fw(Xx), Mx); masks["xd"] = Mx
    s = s5.sseed("step5-init", ds, mech, lv, seed, method); info = {}
    D = Xa.shape[2]
    if method == "Linear regression":
        imp = {n: dc.impute_lr_time(Z) for n, Z in inputs.items()}
    else:
        if variant in ("global", "own"):
            Ztr, Zes_true, Mes = Za_m, Za[es], Mtr[es]
            prep = lambda Z: (Z, (lambda W: W))
        elif variant == "seq":
            m_tr, s_tr = s5.seq_stats(Za_m); Ztr = (Za_m - m_tr) / s_tr
            m_es, s_es = s5.seq_stats(Za_m[es]); Zes_true = (Za[es] - m_es) / s_es; Mes = Mtr[es]
            def prep(Z):
                m, sc = s5.seq_stats(Z)
                return (Z - m) / sc, (lambda W, m=m, sc=sc: W * sc + m)
        elif variant == "gstat":
            Ztr = np.concatenate([Za_m, win_stats(Za_m)], axis=2)
            Zes_true = np.concatenate([Za[es], win_stats(Za_m[es])], axis=2)
            Mes = np.concatenate([Mtr[es], np.zeros(Mtr[es].shape, bool), np.zeros(Mtr[es].shape, bool)], axis=2)
            prep = lambda Z: (np.concatenate([Z, win_stats(Z)], axis=2), (lambda W: W[..., :D]))
        else:
            raise ValueError(variant)
        P = {n: prep(Z) for n, Z in inputs.items()}
        if method == "KNN":
            ref = Ztr[np.r_[tr, es]]
            t0 = time.time(); Zi = {n: dc.impute_knn(ref, z, dc.CFG["knn_k"]) for n, (z, _) in P.items()}
            info["seconds_fit_impute"] = time.time() - t0
        else:
            Zi, info = fit_impute(dc, v2, ev, method, Ztr[tr], Zes_true, Mes, {n: z for n, (z, _) in P.items()}, s)
        imp = {n: P[n][1](Zi[n]) for n in P}
    imp = {n: np.where(np.isnan(inputs[n]), bw(imp[n]), bw(inputs[n])) for n in imp}     # reporting scale
    arrs = {n: imp[n][masks[n]].astype(np.float32) for n in imp}
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    os.makedirs(os.path.join(out or OUT, "results"), exist_ok=True)
    np.savez_compressed(res_path(j, ".npz", out), **arrs)
    res = dict(j, rmse=dc.rmse_at(Xte, imp["te"], Mte), rmse_cal=dc.rmse_at(Xa[cal], imp["cal"], Mtr[cal]),
               n_es_windows=int(len(es)), n_cal_windows=int(len(cal)), **info)
    return res


# ------------------------------------------------------------------------------------------- running
def environment():
    import torch
    try:
        import pypots; pv = pypots.__version__
    except Exception:
        pv = "?"
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, pypots=pv,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


def _attempt(j, dc, v2, ev, out, errdir):
    res = None
    for strict in (True, False):
        s5.strict_mode(strict)
        try:
            res = run_job(j, dc, v2, ev, out); res["deterministic"] = strict; break
        except RuntimeError as e:
            if strict and "deterministic" in str(e).lower(): continue
            open(os.path.join(errdir, hashlib.md5(j["id"].encode()).hexdigest() + ".txt"), "w").write(j["id"] + "\n" + traceback.format_exc()); break
        except Exception:
            open(os.path.join(errdir, hashlib.md5(j["id"].encode()).hexdigest() + ".txt"), "w").write(j["id"] + "\n" + traceback.format_exc()); break
    return res


def run(worker=0, n_workers=1):
    plan = os.path.join(OUT, "PLAN_step8.md")
    if not os.path.exists(plan): sys.exit("The plan is missing: python step8_benchmark.py plan")
    if not os.path.exists(cache8("CMAPSS_M")[0]): sys.exit("run first: python step8_benchmark.py prepare")
    sha = script_sha()
    if sha not in open(plan, encoding="utf-8").read():
        print("WARNING: this script differs from the version recorded in PLAN_step8.md; document the change.")
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True); os.makedirs(os.path.join(OUT, "errors"), exist_ok=True)
    dc, v2, ev = s5.modules(); import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // max(1, n_workers)))
    jobs = [j for i, j in enumerate(job_list()) if i % n_workers == worker]
    for i, j in enumerate(jobs, 1):
        p = res_path(j, ".json")
        if os.path.exists(p): continue
        print(f"[{i}/{len(jobs)}] {j['id']}", flush=True); t0 = time.time()
        res = _attempt(j, dc, v2, ev, None, os.path.join(OUT, "errors"))
        if res is not None:
            res.update(seconds=round(time.time() - t0, 1), script_sha256=sha); tmp = p + ".tmp"
            json.dump(res, open(tmp, "w"), indent=1, default=float); os.replace(tmp, p)
    json.dump(environment(), open(os.path.join(OUT, "environment.json"), "w"), indent=1)


def status():
    jobs = job_list(); done = [os.path.exists(res_path(j, ".json")) for j in jobs]
    errs = glob.glob(os.path.join(OUT, "errors", "*.txt"))
    print(f"finished {sum(done)} of {len(jobs)} jobs; errors: {len(errs)}")
    df = pd.DataFrame(jobs).assign(done=done)
    first = df.seed.isin(FIRST_SEEDS)
    print(f"  seeds {FIRST_SEEDS}: {int(df[first].done.sum())} of {int(first.sum())}; "
          f"other seeds: {int(df[~first].done.sum())} of {int((~first).sum())}")
    print(df.groupby(["tier", "method"]).done.agg(["sum", "count"]).to_string())
    for e in errs[:5]: print("  ", open(e).read().splitlines()[0])


def check(dc=None):
    """C-MAPSS, KNN, two settings: the test imputations must equal the Step 7 files"""
    if dc is None: dc, _, _ = s5.modules()
    out = os.path.join(OUT, "check"); bad = 0; n = 0
    for key in ("CMAPSS|MNAR|0.3|0", "CMAPSS|MCAR|0.7|3"):
        ds, mech, lv, seed = key.split("|")
        for norm in NORMS:
            j = _job(ds, mech, float(lv), int(seed), "KNN", norm, 1); j["id"] = f"{key}|KNN|{norm}"
            p7 = s7.res_path(j, ".npz")
            if not os.path.exists(p7): print("Step 7 file missing:", j["id"]); continue
            r = run_job(j, dc, None, None, out=out)
            a = np.load(res_path(j, ".npz", out))["te"].astype(float); b = np.load(p7)["te"].astype(float)
            d = float(np.abs(a - b).max()); bad += d > 1e-6; n += 1
            print(f"{j['id']}: max |difference| to Step 7 = {d:.2e}; RMSE {r['rmse']:.6f}")
    print("reproduction check:", "passed" if (bad == 0 and n > 0) else "FAILED")


def smoke():
    """every model and variant once on a tiny subset with two epochs (results in step8_outputs/smoke)"""
    FAST["on"] = True
    dc, v2, ev = s5.modules(); out = os.path.join(OUT, "smoke"); err = os.path.join(out, "errors")
    os.makedirs(err, exist_ok=True); ok = 0; todo = []
    for m in BACKBONES:
        for n in NORMS: todo.append(_job("CMAPSS", "MCAR", 0.3, 0, m, n, 1))
    todo += [_job("CMAPSS", "MCAR", 0.3, 0, m, "gstat", 2) for m in GSTAT]
    todo += [_job("CMAPSS", "MCAR", 0.3, 0, m, "own", 2) for m in OWN]
    todo += [_job("CMAPSS_M", "GAP", 0.4, 0, "SAITS", "seq", 1), _job("MetroPT", "MNAR", 0.3, 0, "ImputeFormer", "global", 1),
             _job("MetroPT", "MCAR", 0.9, 0, "BRITS", "seq", 3), _job("CMAPSS", "MCAR", 0.1, 0, "Linear regression", "global", 3)]
    for j in todo:
        t0 = time.time(); r = _attempt(j, dc, v2, ev, out, err)
        if r is None: print(f"FAILED  {j['id']} (see {err})"); continue
        ok += 1; print(f"ok      {j['id']}: {time.time() - t0:5.1f} s, RMSE {r['rmse']:.3f}, deterministic {r['deterministic']}")
    print(f"smoke test: {ok} of {len(todo)} passed")


# ------------------------------------------------------------------------------------------- report
def gmean(x):
    x = np.asarray(x, float); return float(np.exp(np.log(x).mean()))


def _load_pred(j):
    p = res_path(j, ".npz")
    if os.path.exists(p): return np.load(p), "step8"
    if j["dataset"] == "MetroPT" and j["method"] in STEP7_REUSE and (j["mech"], j["level"]) in CONDS:
        q = s7.res_path(dict(id=j["id"]), ".npz")
        if os.path.exists(q): return np.load(q), "step7"
    return None, None


def _metrics(p, y):
    e = p - y
    return float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean())


def _variants(g, s_, a, d, guard_n, cal_g, cal_s, y_cal, d_cal):
    """G, S and the combinations, for prediction arrays at the removed positions"""
    an = np.where(guard_n, 0.0, a[d])
    Eg, Es = ((cal_g - y_cal) ** 2).mean(), ((cal_s - y_cal) ** 2).mean()
    pooled = s5.dual_weights(y_cal, cal_g, cal_s, np.zeros_like(d_cal), 1)[0]
    ap = np.where(guard_n, 0.0, pooled)
    return {"global": g, "seq": s_, "normblend": an * g + (1 - an) * s_, "nb_noguard": a[d] * g + (1 - a[d]) * s_,
            "nb_pooled": ap * g + (1 - ap) * s_, "nb_equal": 0.5 * (g + s_), "valsel": g if Eg <= Es else s_}


def evaluate(dc):
    rows, wrows, grows, xrows, repro = [], [], [], [], []
    settings = [(ds, mech, lv, seed) for seed in SEEDS for ds in DATASETS for (mech, lv) in CONDS]
    settings += [(ds, mech, lv, seed) for seed in SEEDS for ds in SWEEP_DS for (mech, lv) in SWEEP]
    for ds, mech, lv, seed in settings:
        cond = f"{mech} {lv:g}"; sweep = (mech, lv) in SWEEP
        S, Mtr, Mte, es, cal, mu, sd = parts(dc, ds, mech, lv, seed)
        Za_m = dc.apply_mask((S["Xa"] - mu) / sd, Mtr); Zte_m = dc.apply_mask((S["Xte"] - mu) / sd, Mte)
        y = S["Xte"][Mte]; n, _, d = np.nonzero(Mte); D = S["Xa"].shape[2]
        y_c = S["Xa"][cal][Mtr[cal]]; _, _, d_c = np.nonzero(Mtr[cal])
        guard = s5.window_guard(Za_m[S["tr"]], Zte_m)
        grows.append(dict(dataset=ds, cond=cond, seed=seed, guard_share=float(guard.mean())))
        xd = None
        if ds == "CMAPSS" and not sweep:
            Xx, Mx, Sm, (mu_c, sd_c, mu_m, sd_m) = cross_domain(dc, mech, lv, seed)
            Zx_m = dc.apply_mask((Xx - mu) / sd, Mx); gx = s5.window_guard(Za_m[S["tr"]], Zx_m)
            nx, _, dxv = np.nonzero(Mx); y_x = Sm["Xte"][Mx]
            to_m = lambda P: ((P * sd_c[dxv] + mu_c[dxv]) - mu_m[dxv]) / sd_m[dxv]     # C-MAPSS -> CMAPSS_M scale
            xd = (gx, nx, dxv, y_x, to_m)
            grows.append(dict(dataset="CMAPSS->CMAPSS_M", cond=cond, seed=seed, guard_share=float(gx.mean())))
        base = dict(dataset=ds, cond=cond, seed=seed)
        def get(method, variant):
            return _load_pred(_job(ds, mech, lv, seed, method, variant, 0))
        z, src = get("Linear regression", "global")
        if z is not None:
            r_, a_ = _metrics(z["te"].astype(float), y)
            rows.append(dict(base, model="Linear regression", variant="-", rmse=r_, mae=a_, source=src))
        for mdl in (SWEEP_MODELS if sweep else BACKBONES):
            (zg, sg), (zs, ss) = get(mdl, "global"), get(mdl, "seq")
            if zg is None or zs is None: continue
            g, s_ = zg["te"].astype(float), zs["te"].astype(float)
            cg, cs = zg["cal"].astype(float), zs["cal"].astype(float)
            a = s5.dual_weights(y_c, cg, cs, d_c, D)
            for v, p in _variants(g, s_, a, d, guard[n], cg, cs, y_c, d_c).items():
                r_, a_ = _metrics(p, y)
                rows.append(dict(base, model=mdl, variant=v, rmse=r_, mae=a_, source=sg))
            for k in range(D):
                wrows.append(dict(base, model=mdl, var=k, weight=float(a[k])))
            if ds == "CMAPSS" and mdl in ("KNN", "SAITS", "Transformer") and not sweep:      # reproduction of Step 7
                for nm, zz in (("global", zg), ("seq", zs)):
                    q = s7.res_path(dict(id=f"{ds}|{mech}|{lv}|{seed}|{mdl}|{nm}"), ".npz")
                    if os.path.exists(q):
                        repro.append(dict(base, model=mdl, norm=nm, max_abs_diff=float(np.abs(np.load(q)["te"].astype(float) - zz["te"].astype(float)).max())))
            if xd is not None and "xd" in zg.files and "xd" in zs.files:
                gx, nx, dxv, y_x, to_m = xd
                for v, p in _variants(to_m(zg["xd"].astype(float)), to_m(zs["xd"].astype(float)), a, dxv, gx[nx],
                                      cg, cs, y_c, d_c).items():
                    r_, a_ = _metrics(p, y_x)
                    xrows.append(dict(base, dataset="CMAPSS->CMAPSS_M", model=mdl, variant=v, rmse=r_, mae=a_))
        if not sweep:
            for mdl, var in [(m, "gstat") for m in GSTAT] + [(m, "own") for m in OWN]:
                z, src = get(mdl, var)
                if z is None: continue
                r_, a_ = _metrics(z["te"].astype(float), y)
                rows.append(dict(base, model=mdl if var == "own" else f"{mdl}+stats", variant=var, rmse=r_, mae=a_, source=src))
                if xd is not None and "xd" in z.files:
                    gx, nx, dxv, y_x, to_m = xd
                    r_, a_ = _metrics(to_m(z["xd"].astype(float)), y_x)
                    xrows.append(dict(base, dataset="CMAPSS->CMAPSS_M", model=mdl if var == "own" else f"{mdl}+stats",
                                      variant=var, rmse=r_, mae=a_))
    return (pd.DataFrame(rows), pd.DataFrame(wrows), pd.DataFrame(grows), pd.DataFrame(xrows), pd.DataFrame(repro))


def examples(dc, picks=(("MetroPT", "GAP", 0.4, 0, "SAITS"), ("CMAPSS_M", "MCAR", 0.3, 0, "SAITS"),
                        ("CMAPSS", "MCAR", 0.3, 0, "ImputeFormer"))):
    """full windows (truth, mask, G, S, NormBlend) for figures"""
    out = {}
    for ds, mech, lv, seed, mdl in picks:
        S, Mtr, Mte, es, cal, mu, sd = parts(dc, ds, mech, lv, seed)
        (zg, _), (zs, _) = _load_pred(_job(ds, mech, lv, seed, mdl, "global", 0)), _load_pred(_job(ds, mech, lv, seed, mdl, "seq", 0))
        if zg is None or zs is None: continue
        Za_m = dc.apply_mask((S["Xa"] - mu) / sd, Mtr); Zte_m = dc.apply_mask((S["Xte"] - mu) / sd, Mte)
        n, _, d = np.nonzero(Mte); D = S["Xa"].shape[2]
        y_c = S["Xa"][cal][Mtr[cal]]; _, _, d_c = np.nonzero(Mtr[cal])
        a = s5.dual_weights(y_c, zg["cal"].astype(float), zs["cal"].astype(float), d_c, D)
        guard = s5.window_guard(Za_m[S["tr"]], Zte_m); an = np.where(guard[n], 0.0, a[d])
        g, s_ = zg["te"].astype(float), zs["te"].astype(float)
        diff = pd.Series(np.abs(g - s_)).groupby(n).mean()
        pick = sorted(set(range(min(4, len(S["Xte"])))) | set(diff.sort_values().index[-4:].tolist()))
        key = f"{ds}|{mech}|{lv}|{seed}|{mdl}"
        for nm, v in (("G", g), ("S", s_), ("NB", an * g + (1 - an) * s_)):
            F = np.full(S["Xte"].shape, np.nan); F[Mte] = v; out[f"{key}|{nm}"] = F[pick]
        out[f"{key}|truth"] = S["Xte"][pick]; out[f"{key}|mask"] = Mte[pick]; out[f"{key}|weights"] = a
        out[f"{key}|windows"] = np.array(pick)
    np.savez_compressed(os.path.join(OUT, "step8_examples.npz"), **out)


def report():
    dc, _, _ = s5.modules()
    R, W, G, X, P = evaluate(dc)
    if R.empty: sys.exit("no results yet")
    R.to_csv(os.path.join(OUT, "step8_runs.csv"), index=False); W.to_csv(os.path.join(OUT, "step8_weights.csv"), index=False)
    G.to_csv(os.path.join(OUT, "step8_guard.csv"), index=False); X.to_csv(os.path.join(OUT, "step8_cross.csv"), index=False)
    P.to_csv(os.path.join(OUT, "step8_reproduction.csv"), index=False)
    jr = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "results", "*.json"))]
    if jr: pd.DataFrame(jr).to_csv(os.path.join(OUT, "step8_jobs.csv"), index=False)
    try:
        examples(dc)
    except Exception:
        traceback.print_exc()
    L = ["# Step 8 report (benchmark extension, descriptive)", "",
         f"Script versions in the results: {sorted({r.get('script_sha256') for r in jr})}", ""]
    main = R[~R.cond.isin([f"MCAR {lv:g}" for _, lv in SWEEP])]
    m = main[main.variant.isin(["global", "seq", "normblend", "valsel"])].groupby(["dataset", "model", "variant"]).rmse.mean().unstack()
    if {"global", "seq", "normblend"} <= set(m.columns):
        m["oracle"] = m[["global", "seq"]].min(axis=1); m["NB/oracle"] = m.normblend / m.oracle
        L += ["## Mean RMSE over conditions and seeds", "", "```", m.round(4).to_string(), "```", ""]
        c = main.groupby(["dataset", "cond", "model", "variant"]).rmse.mean().unstack()
        c = c.dropna(subset=["global", "seq", "normblend"]); c["ratio"] = c.normblend / c[["global", "seq"]].min(axis=1)
        L += [f"Cells: {len(c)}; NormBlend <= 1.02 x oracle in {int((c.ratio <= 1.02).sum())}; below the oracle in "
              f"{int((c.ratio < 1).sum())}; geometric mean ratio {gmean(c.ratio):.3f}; worst {c.ratio.max():.3f}", ""]
    o = main[~main.variant.isin(["global", "seq", "normblend", "nb_noguard", "nb_pooled", "nb_equal", "valsel"])]
    if not o.empty:
        L += ["## Single-model alternatives (mean RMSE)", "", "```",
              o.groupby(["dataset", "model"]).rmse.mean().unstack(0).round(4).to_string(), "```", ""]
    if not X.empty:
        L += ["## Cross-domain C-MAPSS FD001+FD003 -> FD002+FD004 (mean RMSE, FD002+FD004 scale)", "", "```",
              X.groupby(["model", "variant"]).rmse.mean().unstack().round(4).to_string(), "```", ""]
    sw = R[R.cond.isin([f"MCAR {lv:g}" for _, lv in SWEEP] + ["MCAR 0.3", "MCAR 0.7"]) & R.dataset.isin(SWEEP_DS)]
    if not sw.empty:
        L += ["## Missing-rate sweep (mean RMSE)", "", "```",
              sw[sw.variant.isin(["global", "seq", "normblend"]) & sw.model.isin(SWEEP_MODELS)]
              .groupby(["dataset", "model", "variant", "cond"]).rmse.mean().unstack().round(4).to_string(), "```", ""]
    L += ["## Guard activation (share of test windows)", "", "```",
          G.groupby(["dataset", "cond"]).guard_share.mean().round(4).unstack().to_string(), "```", ""]
    if not P.empty:
        L += [f"Reproduction of Step 7 on C-MAPSS: {len(P)} comparisons, largest |difference| {P.max_abs_diff.max():.2e}", ""]
    if jr:
        J = pd.DataFrame(jr)
        L += [f"Runs without strict determinism: {int((~J.deterministic.astype(bool)).sum())} of {len(J)}", ""]
    txt = "\n".join(L); open(os.path.join(OUT, "REPORT_step8.md"), "w", encoding="utf-8").write(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "prepare", "check", "smoke", "run", "status", "report"])
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "plan":
        os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, "PLAN_step8.md")
        if os.path.exists(p): sys.exit(f"{p} already exists; it must not be changed after results exist")
        txt = PLAN.format(sha=script_sha(), sha5=sha_file(os.path.join(HERE, "step5_confirm.py")),
                          sha7=sha_file(os.path.join(HERE, "step7_robustness.py")), seeds=SEEDS, first=FIRST_SEEDS,
                          hp=json.dumps({k: v[1] for k, v in PYPOTS.items()}), train=json.dumps(TRAIN))
        n = len(job_list())
        open(p, "w", encoding="utf-8").write(txt + f"\nJobs: {n}\n\nWritten: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(open(p, encoding="utf-8").read())
    elif a.cmd == "prepare": prepare()
    elif a.cmd == "check": check()
    elif a.cmd == "smoke": smoke()
    elif a.cmd == "run": run(a.worker, a.n_workers)
    elif a.cmd == "status": status()
    else: report()
