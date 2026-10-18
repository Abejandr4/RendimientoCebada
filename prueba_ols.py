import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from modelo_lib import Y, prior_mix, nuevo_modelo

# ---------- Datos (mismo orden que entrenar_y_guardar.py)
df = pd.read_parquet("df_master.parquet")
S2 = [c for c in df.columns if c.endswith("_s2") and not c.startswith("n_obs")]
fen = pd.read_csv("fenologia_features.csv")
fen["ciclo_sin_senescencia_s2"] = fen["sen50_doy_s2"].isna().astype(int)
FEN = [c for c in fen.columns if c not in ("ID_POLIGONO", "sen50_doy_s2", "dur50_dias_s2")]
df = df.merge(fen, on="ID_POLIGONO", how="left")
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in df.columns]
TODAS = S2 + FEN + GEO
POCAS = ["ndvi_p95_s2", "ndvi_p05_s2", "ndvi_auc_s2", "gu50_doy_s2", "dias_sobre90_s2",
         "elev_mean", "pend_mean", "area_ha_geom"]          # definidas de antemano

def modelo(kind):
    if kind == "ols":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    return nuevo_modelo()                                   # Ridge con penalización elegida por CV

CFG = {"Ridge, 34 variables (actual)": ("ridge", TODAS),
       "OLS, 34 variables":            ("ols", TODAS),
       "Ridge, 8 variables":           ("ridge", POCAS),
       "OLS, 8 variables":             ("ols", POCAS)}
BASE = "Ridge, 34 variables (actual)"

train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
yv = train[Y].to_numpy(); n = len(train)
coords = train[["x_m", "y_m"]].to_numpy()
rng = np.random.default_rng(0)

split_alea = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))
split_esp = []
for rep in range(3):
    gr = KMeans(15, n_init=10, random_state=rep).fit_predict(coords)
    split_esp += list(GroupKFold(5).split(coords, yv, gr))

def err_cuad(kind, cols, splits):
    se, cnt = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        dtr, dte = train.iloc[tr], train.iloc[te]
        p_tr, p_te = prior_mix(dtr, dte)
        m = modelo(kind).fit(dtr[cols], dtr[Y].to_numpy() - p_tr)
        se[te] += (yv[te] - (p_te + m.predict(dte[cols]))) ** 2; cnt[te] += 1
    return se / cnt

# ---------- Penalización que elige Ridge con todos los datos
p_tr, _ = prior_mix(train, train.iloc[:1])
m = nuevo_modelo().fit(train[TODAS], yv - p_tr)
print(f"alpha elegido por RidgeCV: {m[-1].alpha_:.1f} (rango probado: 0.1 a 1000)")

for esq, sp in {"aleatorio": split_alea, "espacial": split_esp}.items():
    print(f"\n===== CV {esq} (RMSE agrupado; dif y IC95 vs. '{BASE}') =====", flush=True)
    res = {k: err_cuad(kind, cols, sp) for k, (kind, cols) in CFG.items()}
    base = res[BASE]; idx = rng.integers(0, n, (2000, n))
    for k, se in res.items():
        d = np.sqrt(base[idx].mean(1)) - np.sqrt(se[idx].mean(1))
        lo, hi = np.percentile(d, [2.5, 97.5])
        marca = "" if k == BASE else ("REAL" if lo > 0 else ("peor" if hi < 0 else "ruido"))
        print(f"  {k:30s} RMSE {np.sqrt(se.mean()):.3f}  dif {np.sqrt(base.mean())-np.sqrt(se.mean()):+.3f}"
              f"  IC95 [{lo:+.3f}, {hi:+.3f}]  {marca}", flush=True)