"""GBM ranging-error mitigation with features {d_hat, PR, R_MaxNoise} (Ferreira et al., IEEE TIM 2021)
on the public Bregar UWB dataset, channel 5, leave-one-environment-out cross-validation.
Ranging accuracy + localisation (LS, GBM + original WLS, GBM + improved WLS) + computational cost."""
import os, sys, time, json, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

SP = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
SEED, EPS, NITER = 0, 1e-6, 1000
os.makedirs(OUT, exist_ok=True)

meta = pd.read_pickle(os.path.join(SP, "ch5_meta.pkl"))
meta["PR"] = meta.RSS_FP / meta.RSS                    # Eq. (6) in [Ferreira]: P_FP / P_T
meta["RMaxNoise"] = meta.MAX_NOISE / meta.FP_POINT2     # Eq. (4) in [Ferreira]: P_MaxNoise / F2
FEATS = ["RANGE", "PR", "RMaxNoise"]
valid = ~((meta.env == 2) & (meta.anchor == "A6"))     # faulty anchor skipped, as in Bregar's code

# ---------------- training (LOEO) ----------------
meta["e_GBM"] = np.nan
train_time = []
for k in range(4):
    tr, te = (meta.env != k) & valid, meta.env == k
    gbm = GradientBoostingRegressor(loss="squared_error", learning_rate=0.1, max_depth=3,
                                    n_estimators=100, random_state=SEED)   # settings from [Ferreira]
    t0 = time.time()
    gbm.fit(meta.loc[tr, FEATS].to_numpy(), meta.loc[tr, "range_error"].to_numpy())
    train_time.append(time.time() - t0)
    meta.loc[te, "e_GBM"] = gbm.predict(meta.loc[te, FEATS].to_numpy())
    pickle.dump(gbm, open(os.path.join(OUT, f"gbm_test_env{k}.pkl"), "wb"))
    print(f"fold env{k}: trained in {train_time[-1]:.1f}s, importances "
          + ", ".join(f"{f}={v:.2f}" for f, v in zip(FEATS, gbm.feature_importances_)), flush=True)
meta[["env", "pos", "anchor", "meas", "RANGE", "true_range", "range_error", "PR", "RMaxNoise", "e_GBM"]].to_csv(
    os.path.join(OUT, "gbm_predictions.csv"), index=False)

# ---------------- ranging accuracy ----------------
rm = lambda e: np.sqrt(np.mean(e ** 2)) * 100
rows = []
for k in range(4):
    g = meta[(meta.env == k) & valid]
    r = g.range_error - g.e_GBM
    rows += [dict(env=k, method="UWB", RMSE=rm(g.range_error), MAE=np.abs(g.range_error).mean() * 100),
             dict(env=k, method="GBM", RMSE=rm(r), MAE=np.abs(r).mean() * 100)]
rng = pd.DataFrame(rows)
rng.to_csv(os.path.join(OUT, "ranging_per_env.csv"), index=False)
print("\nRANGING per environment (cm)\n", rng.pivot(index="env", columns="method").round(2))
print("RANGING mean ± std over environments\n", rng.groupby("method")[["RMSE", "MAE"]].agg(["mean", "std"]).round(2))

# ---------------- localisation ----------------
def solve(A, d, w=None):
    H = np.column_stack([-2 * A[:, 0], -2 * A[:, 1], np.ones(len(A))])
    b = d ** 2 - A[:, 0] ** 2 - A[:, 1] ** 2
    if w is None:
        return np.linalg.lstsq(H, b, rcond=None)[0][:2]
    HW = H.T * w
    return np.linalg.solve(HW @ H, HW @ b)[:2]

rs = np.random.default_rng(SEED)
pos_rows, all_err = [], []
for k in range(4):
    g = meta[meta.env == k]
    anchors = sorted(g.anchor.unique())
    A = np.array([g[g.anchor == a][["X_ANCHOR", "Y_ANCHOR"]].iloc[0].to_numpy() for a in anchors], float)
    errs = {m: [] for m in ["LS", "GBM + original WLS", "GBM + improved WLS"]}
    for pos, gp in g.groupby("pos"):
        true = gp[["X_TAG", "Y_TAG"]].iloc[0].to_numpy(float)
        per_a = [gp[gp.anchor == a] for a in anchors]
        idx = [rs.integers(0, len(p), NITER) for p in per_a]      # random range combinations, as in Bregar's code
        D = np.stack([p.RANGE.to_numpy()[i] for p, i in zip(per_a, idx)], 1)
        E = np.stack([p.e_GBM.to_numpy()[i] for p, i in zip(per_a, idx)], 1)
        for d, e in zip(D, E):
            dc = d - e
            w_o = 1.0 / (np.maximum(np.abs(d), 0.01) * (e ** 2 + EPS))    # original weights (thesis Eq. 9)
            w_i = 1.0 / (np.maximum(np.abs(dc), 0.01) * (e ** 2 + EPS))   # improved weights (thesis Eq. 11)
            errs["LS"].append(np.linalg.norm(solve(A, d) - true))
            errs["GBM + original WLS"].append(np.linalg.norm(solve(A, dc, w_o) - true))
            errs["GBM + improved WLS"].append(np.linalg.norm(solve(A, dc, w_i) - true))
    for m, e in errs.items():
        e = np.asarray(e) * 100
        pos_rows.append(dict(env=k, method=m, RMSE=np.sqrt(np.mean(e ** 2)), MAE=e.mean(), STD=e.std()))
    print(f"localisation env{k} done", flush=True)
pos = pd.DataFrame(pos_rows)
pos.to_csv(os.path.join(OUT, "localisation_per_env.csv"), index=False)
print("\nLOCALISATION per environment (cm)\n", pos.pivot(index="env", columns="method", values="RMSE").round(2))
print("LOCALISATION mean ± std over environments\n",
      pos.groupby("method")[["RMSE", "MAE", "STD"]].agg(["mean", "std"]).round(2))

# ---------------- computational cost ----------------
gbm = pickle.load(open(os.path.join(OUT, "gbm_test_env0.pkl"), "rb"))
n_nodes = sum(t[0].tree_.node_count for t in gbm.estimators_)
n_leaves = sum(t[0].tree_.n_leaves for t in gbm.estimators_)
n_split = n_nodes - n_leaves
x1 = meta.loc[[0], FEATS].to_numpy()
for _ in range(50): gbm.predict(x1)
t0 = time.perf_counter()
for _ in range(2000): gbm.predict(x1)
cost = dict(trees=len(gbm.estimators_), max_depth=3, split_nodes=int(n_split), leaves=int(n_leaves),
            stored_values=int(2 * n_split + n_leaves),
            model_bytes_float32=int(n_split * 8 + n_leaves * 4),   # feature id + threshold per split, value per leaf
            comparisons_per_inference=3 * len(gbm.estimators_), additions_per_inference=len(gbm.estimators_),
            feature_extraction_ops="2 divisions on DW1000 diagnostic registers (RSS, RSS_FP, MAX_NOISE, FP_POINT2)",
            sklearn_us_per_sample_pc=(time.perf_counter() - t0) / 2000 * 1e6,
            train_time_s_mean=float(np.mean(train_time)))
json.dump(cost, open(os.path.join(OUT, "gbm_cost.json"), "w"), indent=2)
print("\nCOST", json.dumps(cost, indent=2))
