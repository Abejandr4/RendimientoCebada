import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Y = "RENDIMIENTO_T_HA"
K_SHRINK, K_VEC, K_INNER = 5, 5, 5
rng = np.random.default_rng(0)

# ---------- 1. Datos
df = pd.read_parquet("df_master.parquet")
S2 = [c for c in df.columns if c.endswith("_s2") and not c.startswith("n_obs")]
fen = pd.read_csv("fenologia_features.csv")
fen["ciclo_sin_senescencia_s2"] = fen["sen50_doy_s2"].isna().astype(int)
FEN = [c for c in fen.columns if c not in ("ID_POLIGONO", "sen50_doy_s2", "dur50_dias_s2")]
df = df.merge(fen, on="ID_POLIGONO", how="left")
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in df.columns]
C = S2 + FEN + GEO

train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
yv = train[Y].to_numpy(); n = len(train)
coords = train[["x_m", "y_m"]].to_numpy()

# ---------- 2. Priors (sin fuga para las filas de entrenamiento)
def _mun(ytr, mtr, mte):
    g = ytr.mean()
    suma = pd.Series(ytr).groupby(mtr).sum(); cnt = pd.Series(ytr).groupby(mtr).count()
    s = pd.Series(mte).map(suma).fillna(0).to_numpy()
    c = pd.Series(mte).map(cnt).fillna(0).to_numpy()
    return (s + K_SHRINK * g) / (c + K_SHRINK)

def priors(dtr, dte):
    """Devuelve P_tr y P_te con columnas [prior_mun, prior_vec]."""
    ytr = dtr[Y].to_numpy()
    mtr, mte = dtr["Municipio"].to_numpy(), dte["Municipio"].to_numpy()
    ctr, cte = dtr[["x_m", "y_m"]].to_numpy(), dte[["x_m", "y_m"]].to_numpy()
    tree = cKDTree(ctr)
    pm_te = _mun(ytr, mtr, mte)
    pv_te = ytr[tree.query(cte, k=K_VEC)[1]].mean(axis=1)
    pm_tr = np.zeros(len(ytr))                                   # K-fold interno: evita fuga
    for a, b in KFold(K_INNER, shuffle=True, random_state=0).split(ctr):
        pm_tr[b] = _mun(ytr[a], mtr[a], mtr[b])
    pv_tr = ytr[tree.query(ctr, k=K_VEC + 1)[1][:, 1:]].mean(axis=1)   # sin la propia parcela
    return np.c_[pm_tr, pv_tr], np.c_[pm_te, pv_te]

# ---------- 3. Configuraciones (fijadas de antemano)
def modelo(kind):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-1, 3, 25)))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(200, min_samples_leaf=3, max_features=0.5,
                                               random_state=0, n_jobs=1))

CFG = {
    "prior mix (referencia)":         dict(mode="mix"),
    "res ridge S2+FEN+GEO (previa)":  dict(mode="res", cols=C, model="ridge"),
    "A. stack: solo priors":          dict(mode="in", cols=[], model="ridge"),
    "B. priors-in ridge S2+FEN+GEO":  dict(mode="in", cols=C, model="ridge"),
    "C. priors-in residual ridge":    dict(mode="res_in", cols=C, model="ridge"),
    "D. priors-in residual rf":       dict(mode="res_in", cols=C, model="rf"),
}
BASE = "prior mix (referencia)"

def predecir(cfg, dtr, dte):
    P_tr, P_te = priors(dtr, dte)
    mix_tr, mix_te = P_tr.mean(axis=1), P_te.mean(axis=1)
    if cfg["mode"] == "mix":
        return mix_te
    ytr = dtr[Y].to_numpy()
    F_tr = dtr[cfg["cols"]].to_numpy(dtype=float) if cfg["cols"] else np.empty((len(dtr), 0))
    F_te = dte[cfg["cols"]].to_numpy(dtype=float) if cfg["cols"] else np.empty((len(dte), 0))
    if cfg["mode"] == "res":                                     # sin priors como entrada
        return mix_te + modelo(cfg["model"]).fit(F_tr, ytr - mix_tr).predict(F_te)
    X_tr, X_te = np.c_[P_tr, F_tr], np.c_[P_te, F_te]
    if cfg["mode"] == "in":                                      # el modelo aprende los pesos
        return modelo(cfg["model"]).fit(X_tr, ytr).predict(X_te)
    return mix_te + modelo(cfg["model"]).fit(X_tr, ytr - mix_tr).predict(X_te)   # res_in

# ---------- 4. Validación
split_alea = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))
split_esp = []
for rep in range(3):
    gr = KMeans(15, n_init=10, random_state=rep).fit_predict(coords)
    split_esp += list(GroupKFold(5).split(coords, yv, gr))

def cv(cfg, splits):
    se, cnt = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        p = predecir(cfg, train.iloc[tr], train.iloc[te])
        se[te] += (yv[te] - p) ** 2; cnt[te] += 1
    return se / cnt

for esq, sp in {"aleatorio": split_alea, "espacial": split_esp}.items():
    print(f"\n===== CV {esq} (RMSE agrupado; dif y IC95 vs. '{BASE}') =====", flush=True)
    res = {nombre: cv(cfg, sp) for nombre, cfg in CFG.items()}
    base = res[BASE]; idx = rng.integers(0, n, (2000, n))
    for nombre, se in res.items():
        d = np.sqrt(base[idx].mean(1)) - np.sqrt(se[idx].mean(1))
        lo, hi = np.percentile(d, [2.5, 97.5])
        marca = "" if nombre == BASE else ("REAL" if lo > 0 else ("peor" if hi < 0 else "ruido"))
        print(f"  {nombre:32s} RMSE {np.sqrt(se.mean()):.3f}  dif {np.sqrt(base.mean())-np.sqrt(se.mean()):+.3f}"
              f"  IC95 [{lo:+.3f}, {hi:+.3f}]  {marca}", flush=True)

# ---------- 5. ¿Qué pesos aprende el stack? (con las 138 parcelas)
P_all, _ = priors(train, train)
m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 25))).fit(P_all, yv)
coef = m[-1].coef_ / m[0].scale_
interc = m[-1].intercept_ - (coef * m[0].mean_).sum()
print(f"\nStack: y = {interc:+.2f} + {coef[0]:.2f}*prior_mun + {coef[1]:.2f}*prior_vec"
      f"   (suma de pesos = {coef.sum():.2f})")
print("Corr(prior_mun, prior_vec) =", round(np.corrcoef(P_all.T)[0, 1], 2))