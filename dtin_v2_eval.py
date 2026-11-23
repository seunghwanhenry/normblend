"""
dtin_v2_eval.py - Table 5 and Table 8 on the ORIGINAL test sets with the final DTIN-v2 (dtin_v2.py).

Frozen configuration (selected on development data, phases 1-2):
  imputation  dtin_v2.imputation_config(): d_model 64, 3 layers, mixed point/span masking, lr 1e-3,
              mask input + per-sequence normalisation, no pseudo-target, early stopping on val-stop.
  downstream  dtin_v2.pretraining_config(): same encoder, pseudo-target weight 1, 2,000 pre-training steps;
              fine-tuning (encoder + linear head), epochs chosen by stratified 5-fold CV on the 20 training
              sequences, final model trained on all 20, evaluated once on the 138 test sequences.
              Ablation row: identical pipeline with pseudo-target weight 0 (masked reconstruction only).

Table 8 (Siemens, consecutive gaps, 40 % per variable, lengths 10-50, 10 seeds): same masks as before.
  DTIN-v2 is trained on dev-train (80 % of the training windows, by batch) with early stopping on val-stop
  (10 %); Mean / Median / Linear regression use only the test sequence; KNN uses the full training set.
  Metrics: RMSE over all variables (primary) and without PI12002_PV_Out (secondary, test/train variance 10.9x).
Table 5 baselines are read from the earlier stratified-CV run (results/table5_cv), which used the same folds.

Worker:  python dtin_v2_eval.py --worker A
"""
import os, json, time, glob, socket, hashlib, argparse, traceback
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
import dtin_core as dc
import dtin_improve as di
import dtin_v2 as v2

SEEDS = list(range(10))
GAP_LENGTHS = [10, 20, 30, 40, 50]
T8_BASELINES = ["Mean", "Median", "KNN", "Linear regression"]
T5_LAMBDAS = {"DTIN-v2": 1.0, "DTIN-v2 (no pseudo-target)": 0.0}
SHIFTED_VAR = "PI12002_PV_Out"

PLAN = f"""# Final evaluation of DTIN-v2 on Tables 5 and 8 (pre-specified)

Written before any test-set result of DTIN-v2 was computed. The configuration is frozen
(dtin_v2.imputation_config / dtin_v2.pretraining_config) and is not changed after these results.
Seeds: {SEEDS}. Table 8 gap lengths: {GAP_LENGTHS} (40 % of each variable removed).
Table 8 primary metric: RMSE over all variables; secondary: RMSE without {SHIFTED_VAR}.
Table 5: Game Day = positive class; accuracy, binary F1, AUROC from each run's own predictions; mean and
sample std over seeds. Paired two-sided Wilcoxon tests by seed.
BRITS / SAITS / CSDI are added later after tuning with a comparable budget. All results are reported.
"""


def out(*p):
    d = os.path.join(dc.CFG["out_dir"], "final_v2", *p[:-1]); os.makedirs(d, exist_ok=True)
    return os.path.join(d, p[-1])


def folder(sub): return os.path.dirname(out(sub, "x"))


def write_plan():
    p = out("PLAN_final_v2.md")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f: f.write(PLAN + f"\nWritten: {time.ctime()}\n")
    return p


# ===================================================================================== Table 8
def shifted_index():
    cols = dc.get_data("Siemens")["info"]["var_cols"]
    return [i for i, c in enumerate(cols) if c.startswith(SHIFTED_VAR)][0]


def t8_job(method, L, seed):
    S = dc.get_data("Siemens"); Xtr, Xte = S["Xtr"], S["Xte"]
    Mtr = dc.make_mask("Siemens", Xtr, "GAP", L, seed, "train")          # identical to the earlier Table 8 runs
    Mte = dc.make_mask("Siemens", Xte, "GAP", L, seed, "test")
    Xtr_m, Xte_m = dc.apply_mask(Xtr, Mtr), dc.apply_mask(Xte, Mte)
    info = {}
    if method == "Mean": imp = dc.impute_seq_stat(Xte_m, np.nanmean)
    elif method == "Median": imp = dc.impute_seq_stat(Xte_m, np.nanmedian)
    elif method == "Linear regression": imp = dc.impute_lr_time(Xte_m)
    elif method == "KNN": imp = dc.impute_knn(Xtr_m, Xte_m, dc.CFG["knn_k"])
    elif method == "DTIN-v2":
        sp = di.dev_split("Siemens")
        model = v2.DTINv2Imputer(v2.imputation_config(), seed=dc.stable_seed("v2final", "table8", L, seed))
        model.fit(Xtr_m[sp["train"]], Xtr[sp["stop"]], Mtr[sp["stop"]])
        imp = model.impute(Xte_m); info = dict(best_step=model.info["best_step"], best_val_rmse=model.info["best_val_rmse"])
        del model; torch.cuda.empty_cache()
    else:
        raise ValueError(method)
    j = shifted_index(); M2 = Mte.copy(); M2[:, :, j] = False
    return dict(info, rmse=dc.rmse_at(Xte, imp, Mte), rmse_wo_shifted=dc.rmse_at(Xte, imp, M2))


# ===================================================================================== Table 5
def t5_job(method, seed):
    D = dc.get_data("DodgerLoopGame"); Xtr, Xte = D["Xtr"], D["Xte"]
    ytr = (D["ytr"] == dc.CFG["dodger_positive_label"]).astype(int)
    yte = (D["yte"] == dc.CFG["dodger_positive_label"]).astype(int)
    lam = T5_LAMBDAS[method]; s = dc.stable_seed("v2final", "table5", method, seed)
    enc = v2.pretrain_encoder(Xtr, v2.pretraining_config(pseudo_weight=lam), seed=s)     # 20 sequences, no labels
    fold_seed = dc.stable_seed("fold", seed) % (2**31 - 1)                               # same folds as table5_cv
    epochs, curve = v2.select_epochs_cv(enc, Xtr, ytr, n_splits=5, max_epochs=dc.CFG["head_epochs"], seed=fold_seed)
    clf = v2.fit_classifier(enc, Xtr, ytr, epochs, seed=s)
    prob = v2.predict_proba(clf, Xte)[:, 1]; pred = (prob >= 0.5).astype(int)
    c = dc.confusion(yte, pred)
    np.savez(out("results", f"table5_{method}_seed{seed}_predictions.npz"), y_true=yte, y_pred=pred, prob=prob)
    del enc, clf; torch.cuda.empty_cache()
    return dict(**c, n_test=int(len(yte)), n_test_positive=int(yte.sum()), accuracy=(c["TP"] + c["TN"]) / len(yte),
                f1=dc.binary_f1(c), auroc=float(roc_auc_score(yte, prob)), selected_epoch=epochs,
                cv_curve=[round(float(x), 4) for x in curve])


# ===================================================================================== deep baselines (Table 8)
# Same data and training protocol as DTIN-v2: dev-train (80 %) for training, val-stop (10 %) for early stopping.
DEEP = ["BRITS", "SAITS", "CSDI"]
BASE_TRAIN = {                       # max epochs / patience / optimiser; CSDI schedule from its original study
    "BRITS": dict(epochs=300, patience=20, batch_size=32, lr=1e-3, weight_decay=0.0, milestones=None),
    "SAITS": dict(epochs=300, patience=20, batch_size=32, lr=1e-3, weight_decay=0.0, milestones=None),
    "CSDI":  dict(epochs=200, patience=20, batch_size=16, lr=1e-3, weight_decay=1e-6, milestones=(0.75, 0.90)),
}
BASE_DEFAULT = {                     # configurations of the original studies
    "BRITS": dict(rnn_hidden_size=64),
    "CSDI": dict(n_layers=4, n_heads=8, n_channels=64, d_time_embedding=128, d_feature_embedding=16,
                 d_diffusion_embedding=128, n_diffusion_steps=50),
}
SAITS_GRID = [dict(n_layers=nl, d_model=dm, n_heads=4, d_k=dm // 4, d_v=dm // 4, d_ffn=dm // 2, dropout=0.1)
              for nl in (2, 3) for dm in (128, 256)]                 # light tuning of the strongest competitor
SAITS_TUNE_CONDS = [("ECG200", "MCAR", 0.3), ("Siemens", "MCAR", 0.3), ("Siemens", "MNAR", 0.3), ("Siemens", "GAP", 30)]
DEV_SEEDS = [0, 1, 2]

BASELINE_PLAN = f"""# Deep baselines for Table 8 (pre-specified)

Written before any deep-baseline result in this setting was computed.
Training protocol identical to DTIN-v2: dev-train for training, val-stop for early stopping; same masks and seeds.
BRITS and CSDI: configurations of the original studies {json.dumps(BASE_DEFAULT)}.
SAITS (strongest competitor): light tuning over {len(SAITS_GRID)} configurations {json.dumps(SAITS_GRID)}
on the development conditions {SAITS_TUNE_CONDS} (seeds {DEV_SEEDS}), selected by mean rank of the val-stop RMSE,
exactly as for DTIN-v2.  Training budgets: {json.dumps(BASE_TRAIN)}.
Tuning budget for the paper: DTIN-v2 6 variants (phase 1) + 24 configurations (phase 2); SAITS 4 configurations;
BRITS, CSDI, Mean, Median, KNN (k = 5), Linear regression: no tuning.
"""


def write_baseline_plan():
    p = out("PLAN_baselines.md")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f: f.write(BASELINE_PLAN + f"\nWritten: {time.ctime()}\n")
    return p


def pypots_fit_impute(name, hp, Xtrain_nan, Xval_true, Mval, Xtest_nan, seed, use_val=True, train_override=None):
    import math
    from pypots.imputation import BRITS, SAITS, CSDI
    from pypots.optim import Adam
    from pypots.optim.lr_scheduler import MultiStepLR
    tr = dict(BASE_TRAIN[name], **(train_override or {})); T, D = Xtrain_nan.shape[1:]
    sched = None
    if tr["milestones"]:
        iters = tr["epochs"] * math.ceil(len(Xtrain_nan) / tr["batch_size"])
        sched = MultiStepLR(milestones=[int(f * iters) for f in tr["milestones"]], gamma=0.1)
    dc.set_seed(seed)
    opt = Adam(lr=tr["lr"], weight_decay=tr["weight_decay"], lr_scheduler=sched)
    model = dict(BRITS=BRITS, SAITS=SAITS, CSDI=CSDI)[name](
        **hp, n_steps=T, n_features=D, batch_size=tr["batch_size"], epochs=tr["epochs"],
        patience=tr["patience"] if use_val else None,
        optimizer=opt, device=dc.DEVICE, saving_path=None, model_saving_strategy=None, verbose=False)
    Xval_nan = Xval_true.copy(); Xval_nan[Mval] = np.nan
    if use_val:     # early stopping / checkpoint selection on the validation loss
        model.fit({"X": Xtrain_nan.astype(np.float32)},
                  val_set={"X": Xval_nan.astype(np.float32), "X_ori": Xval_true.astype(np.float32)})
    else:           # original CSDI protocol: full schedule, final model
        model.fit({"X": Xtrain_nan.astype(np.float32)})
    def imp(X):
        kw = dict(n_sampling_times=dc.CFG["csdi_samples"]) if name == "CSDI" else {}
        r = np.asarray(model.impute({"X": X.astype(np.float32)}, **kw), dtype=np.float64)
        if r.ndim == 4: r = np.median(r, axis=1)
        return np.where(np.isnan(X), r, X)
    val_rmse = dc.rmse_at(Xval_true, imp(Xval_nan), Mval)
    test_imp = imp(Xtest_nan) if Xtest_nan is not None else None
    del model; torch.cuda.empty_cache()
    return test_imp, val_rmse


def saits_dev_job(ds, mech, lv, seed, cfg_idx):
    X = dc.get_data(ds)["Xtr"]; sp = di.dev_split(ds)
    Xtr, Xst = X[sp["train"]], X[sp["stop"]]
    Mtr = di.dev_mask(ds, Xtr, mech, lv, seed, "train"); Mst = di.dev_mask(ds, Xst, mech, lv, seed, "stop")
    _, val = pypots_fit_impute("SAITS", SAITS_GRID[cfg_idx], dc.apply_mask(Xtr, Mtr), Xst, Mst, None,
                               dc.stable_seed("saits-dev", ds, mech, lv, seed, cfg_idx))
    return dict(val_stop_rmse=val)


def select_saits():
    p = out("selected_saits.json")
    if os.path.exists(p): return json.load(open(p))
    df = results(); df = df[df.table == "saits_dev"] if not df.empty else df
    need = len(SAITS_GRID) * len(SAITS_TUNE_CONDS) * len(DEV_SEEDS)
    if len(df) < need: raise RuntimeError(f"SAITS tuning not finished: {len(df)}/{need}")
    df["condition"] = df.dataset + " " + df.cond + "-" + df.level.map(lambda l: f"{float(l):g}")
    m = df.groupby(["cfg_idx", "condition"]).val_stop_rmse.mean().unstack()
    rank = m.rank(axis=0).mean(axis=1).sort_values()
    best = int(rank.index[0])
    sel = dict(cfg_idx=best, cfg=SAITS_GRID[best], mean_rank=rank.to_dict(), written=time.ctime())
    with open(p, "w") as f: json.dump(sel, f, indent=1, default=float)
    return sel


def deep_t8_job(method, L, seed):
    S = dc.get_data("Siemens"); Xtr, Xte = S["Xtr"], S["Xte"]; sp = di.dev_split("Siemens")
    Mtr = dc.make_mask("Siemens", Xtr, "GAP", L, seed, "train")
    Mte = dc.make_mask("Siemens", Xte, "GAP", L, seed, "test")
    hp = select_saits()["cfg"] if method == "SAITS" else BASE_DEFAULT[method]
    kw = CSDI_PILOT[decide_csdi()["variant"]] if method == "CSDI" else {}
    imp, val = pypots_fit_impute(method, hp, dc.apply_mask(Xtr, Mtr)[sp["train"]], Xtr[sp["stop"]], Mtr[sp["stop"]],
                                 dc.apply_mask(Xte, Mte), dc.stable_seed("deep-final", method, L, seed), **kw)
    j = shifted_index(); M2 = Mte.copy(); M2[:, :, j] = False
    return dict(rmse=dc.rmse_at(Xte, imp, Mte), rmse_wo_shifted=dc.rmse_at(Xte, imp, M2), val_stop_rmse=val)


# --------------------------------------------------------------------- CSDI pilot (development data only)
CSDI_PILOT = {"early_stop": dict(use_val=True), "full_schedule": dict(use_val=False)}
CSDI_PILOT_PLAN = """# CSDI pilot (pre-specified)

Reason: on Table 8, CSDI trained with early stopping on its validation (noise-prediction) loss gave
unstable results (31/50 runs with test RMSE > 1; validation RMSE 0.36-5.3).
Pilot on development data only (Siemens, GAP-30, dev split, seeds 0-2):
  early_stop     = current protocol (checkpoint chosen by validation loss, patience 20);
  full_schedule  = original CSDI protocol (full 200-epoch schedule, final model, no validation set).
Decision: the variant with the lower mean val-report RMSE is used for all Table 8 CSDI runs; no further changes.
"""


def csdi_pilot_job(variant, seed):
    X = dc.get_data("Siemens")["Xtr"]; sp = di.dev_split("Siemens")
    Xtr, Xst, Xrep = X[sp["train"]], X[sp["stop"]], X[sp["report"]]
    Mtr = di.dev_mask("Siemens", Xtr, "GAP", 30, seed, "train"); Mst = di.dev_mask("Siemens", Xst, "GAP", 30, seed, "stop")
    Mrep = di.dev_mask("Siemens", Xrep, "GAP", 30, seed, "report")
    imp, val = pypots_fit_impute("CSDI", BASE_DEFAULT["CSDI"], dc.apply_mask(Xtr, Mtr), Xst, Mst, dc.apply_mask(Xrep, Mrep),
                                 dc.stable_seed("csdi-pilot", variant, seed), **CSDI_PILOT[variant])
    return dict(val_stop_rmse=val, report_rmse=dc.rmse_at(Xrep, imp, Mrep))


def write_csdi_pilot():
    p = out("PLAN_csdi_pilot.md")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f: f.write(CSDI_PILOT_PLAN + f"\nWritten: {time.ctime()}\n")
    jobs = [dict(table="csdi_pilot", method="CSDI", variant=v, seed=s, id=f"csdi_pilot|{v}|{s}", cost=1)
            for v in CSDI_PILOT for s in DEV_SEEDS]
    with open(out("jobs_csdi_pilot.json"), "w") as f: json.dump(jobs, f, indent=1)
    return p


def csdi_pilot_table():
    df = results()
    if df.empty: return None
    df = df[df.table == "csdi_pilot"]
    if df.empty: return None
    return df.pivot(index="seed", columns="variant", values="report_rmse").round(4), \
           df.groupby("variant")[["report_rmse", "val_stop_rmse"]].agg(["mean", "std"]).round(4)


def decide_csdi():
    """Pilot decision (pre-specified rule): lower mean val-report RMSE. Written once."""
    p = out("selected_csdi_variant.json")
    if os.path.exists(p): return json.load(open(p))
    res = csdi_pilot_table()
    if res is None: raise RuntimeError("CSDI pilot results not found")
    per_seed, summary = res
    if per_seed.notna().sum().min() < len(DEV_SEEDS): raise RuntimeError("CSDI pilot not finished")
    means = per_seed.mean()
    sel = dict(variant=str(means.idxmin()), mean_report_rmse=means.round(4).to_dict(), written=time.ctime())
    with open(p, "w") as f: json.dump(sel, f, indent=1)
    return sel


def archive_csdi_table8():
    """Move the Table 8 CSDI results obtained with early stopping to an archive folder (once)."""
    dest = os.path.join(folder("results"), "_archive_csdi_early_stop"); marker = os.path.join(dest, "ARCHIVED")
    if os.path.exists(marker): return 0
    os.makedirs(dest, exist_ok=True); moved = 0
    for p in glob.glob(os.path.join(folder("results"), "*.json")):
        r = json.load(open(p))
        if r.get("table") == "table8" and r.get("method") == "CSDI":
            os.replace(p, os.path.join(dest, os.path.basename(p))); moved += 1
    with open(marker, "w") as f: f.write(f"archived {moved} CSDI runs (early stopping on validation loss) {time.ctime()}\n")
    return moved


def baseline_jobs(stage):
    if stage == "A":     # SAITS tuning (dev) + BRITS and CSDI on the test benchmark
        jobs = [dict(table="saits_dev", method="SAITS", dataset=ds, cond=m, level=lv, seed=s, cfg_idx=i,
                     id=f"saits_dev|{ds}|{m}|{lv}|{s}|{i}", cost=1)
                for ds, m, lv in SAITS_TUNE_CONDS for s in DEV_SEEDS for i in range(len(SAITS_GRID))]
        jobs += [dict(table="table8", method=m, level=L, seed=s, id=f"table8|{m}|{L}|{s}", cost=5 if m == "CSDI" else 2)
                 for L in GAP_LENGTHS for s in SEEDS for m in ("CSDI", "BRITS")]
    else:                # selected SAITS on the test benchmark
        jobs = [dict(table="table8", method="SAITS", level=L, seed=s, id=f"table8|SAITS|{L}|{s}", cost=2)
                for L in GAP_LENGTHS for s in SEEDS]
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


# ===================================================================================== jobs
def job_list():
    jobs = [dict(table="table5", method=m, level=0, seed=s, id=f"table5|{m}|{s}", cost=3) for s in SEEDS for m in T5_LAMBDAS]
    jobs += [dict(table="table8", method=m, level=L, seed=s, id=f"table8|{m}|{L}|{s}", cost=2 if m == "DTIN-v2" else 1)
             for L in GAP_LENGTHS for s in SEEDS for m in ["DTIN-v2"] + T8_BASELINES]
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


def write_jobs(name="jobs"):
    jobs = job_list() if name == "jobs" else baseline_jobs(name.split("_")[-1])
    p = out(f"{name}.json")
    with open(p, "w") as f: json.dump(jobs, f, indent=1)
    return p


def _key(j): return hashlib.md5(j["id"].encode()).hexdigest()
def res_path(j): return out("results", _key(j) + ".json")
def lock_path(j): return out("locks", _key(j) + ".lock")
def err_path(j): return out("errors", _key(j) + ".txt")


def run_job(j):
    t0 = time.time()
    if j["table"] == "table5": info = t5_job(j["method"], j["seed"])
    elif j["table"] == "saits_dev": info = saits_dev_job(j["dataset"], j["cond"], j["level"], j["seed"], j["cfg_idx"])
    elif j["table"] == "csdi_pilot": info = csdi_pilot_job(j["variant"], j["seed"])
    elif j["method"] in DEEP: info = deep_t8_job(j["method"], j["level"], j["seed"])
    else: info = t8_job(j["method"], j["level"], j["seed"])
    res = dict(j, **info, seconds=round(time.time() - t0, 1))
    tmp = res_path(j) + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f: json.dump(res, f, default=float)
    os.replace(tmp, res_path(j))
    return res


def worker(name, log=print, jobs_file="jobs"):
    for j in json.load(open(out(f"{jobs_file}.json"))):
        if os.path.exists(res_path(j)) or os.path.exists(err_path(j)): continue
        try: fd = os.open(lock_path(j), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError: continue
        with os.fdopen(fd, "w") as f: json.dump(dict(pid=os.getpid(), host=socket.gethostname(), worker=name), f)
        t0 = time.time(); log(f"[{time.strftime('%H:%M:%S')}] start {j['id']}")
        try:
            r = run_job(j)
            msg = (f"rmse={r['rmse']:.4f}" if j["table"] == "table8" else f"val_stop={r['val_stop_rmse']:.4f}"
                   if j["table"] in ("saits_dev", "csdi_pilot") else f"F1={r['f1']:.3f} AUROC={r['auroc']:.3f}")
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


def progress(jobs_file="jobs"):
    jobs = json.load(open(out(f"{jobs_file}.json")))
    st = ["done" if os.path.exists(res_path(j)) else "error" if os.path.exists(err_path(j)) else
          "running" if os.path.exists(lock_path(j)) else "waiting" for j in jobs]
    df = pd.DataFrame(dict(table=[j["table"] for j in jobs], method=[j["method"] for j in jobs], state=st))
    return df.pivot_table(index=["table", "method"], columns="state", aggfunc="size", fill_value=0)


# ===================================================================================== tables
def results():
    return pd.DataFrame([json.load(open(p)) for p in glob.glob(os.path.join(folder("results"), "*.json"))])


def _cell(s): return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"


def _wilcoxon(a, b):
    d = np.asarray(b) - np.asarray(a)
    if len(d) < 2 or np.all(d == 0): return None
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)


def table8(metric="rmse"):
    df = results()
    if df.empty: return None, None
    df = df[df.table == "table8"]
    order = T8_BASELINES + DEEP + ["DTIN-v2"]
    t = df.groupby(["level", "method"])[metric].apply(_cell).unstack().reindex(columns=[m for m in order if m in set(df.method)])
    t.index.name = "interval length"
    tests = []
    for L in GAP_LENGTHS:
        d = df[df.level == L]
        if "DTIN-v2" not in set(d.method): continue
        base = d[d.method.isin(T8_BASELINES + DEEP)].groupby("method")[metric].mean()
        if base.empty: continue
        bb = base.idxmin()
        a = d[d.method == bb].set_index("seed")[metric]; b = d[d.method == "DTIN-v2"].set_index("seed")[metric]
        common = sorted(set(a.index) & set(b.index))
        tests.append(dict(interval=L, best_baseline=bb, baseline_mean=a[common].mean(), dtin_v2_mean=b[common].mean(),
                          n_pairs=len(common), n_dtin_lower=int((b[common].values < a[common].values).sum()),
                          p_two_sided=_wilcoxon(a[common].values, b[common].values)))
    t.to_csv(out(f"table8_{metric}.csv")); df.to_csv(out("table8_per_seed.csv"), index=False)
    return t, pd.DataFrame(tests)


def table5():
    df = results()
    rows = []
    if not df.empty:
        rows += [r for r in df[df.table == "table5"].to_dict("records")]
    prev = glob.glob(os.path.join(dc.CFG["out_dir"], "results", "table5_cv", "*.json"))
    base = pd.DataFrame([json.load(open(p)) for p in prev])
    if not base.empty:
        base = base[base.method.isin(["LSTM", "ResNet", "ROCKET", "InceptionTime"])]
        rows += base.to_dict("records")
    if not rows: return None, None
    allr = pd.DataFrame(rows)
    allr.to_csv(out("table5_per_seed.csv"), index=False)
    order = ["LSTM", "ResNet", "ROCKET", "InceptionTime", "DTIN-v2 (no pseudo-target)", "DTIN-v2"]
    t = pd.DataFrame({k: allr.groupby("method")[k].apply(_cell) for k in ("accuracy", "f1", "auroc")})
    t["n_runs"] = allr.groupby("method").seed.count()
    t = t.reindex([m for m in order if m in t.index])
    tests = []
    for other in ("DTIN-v2 (no pseudo-target)", "ROCKET"):
        a = allr[allr.method == other].set_index("seed").f1; b = allr[allr.method == "DTIN-v2"].set_index("seed").f1
        common = sorted(set(a.index) & set(b.index))
        if len(common) >= 2:
            tests.append(dict(comparison=f"F1: DTIN-v2 vs {other}", n_pairs=len(common), other_mean=a[common].mean(),
                              dtin_v2_mean=b[common].mean(), n_dtin_higher=int((b[common].values > a[common].values).sum()),
                              p_two_sided=_wilcoxon(a[common].values, b[common].values)))
    t.to_csv(out("table5.csv"))
    return t, pd.DataFrame(tests)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("dtin_outputs", "final_v2", "config_final.json"))
    ap.add_argument("--worker", default="F0")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--jobs", default="jobs", help="jobs | jobs_baselines_A | jobs_baselines_B")
    a = ap.parse_args()
    if a.threads > 0: torch.set_num_threads(a.threads)
    dc.init(a.config)
    logf = open(out("logs", f"worker_{a.worker}.log"), "a", buffering=1, encoding="utf-8")
    def log(m): print(m, flush=True); logf.write(m + "\n")
    worker(a.worker, log, a.jobs)
