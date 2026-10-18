import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from modelo_lib import Y, prior_mix

# ---------- Datos (mismo armado que prueba_ols.py)
df = pd.read_parquet("df_master.parquet")
S2 = [c for c in df.columns if c.endswith("_s2") and not c.startswith("n_obs")]
fen = pd.read_csv("fenologia_features.csv")
fen["ciclo_sin_senescencia_s2"] = fen["sen50_doy_s2"].isna().astype(int)
FEN = [c for c in fen.columns if c not in ("ID_POLIGONO", "sen50_doy_s2", "dur50_dias_s2")]
df = df.merge(fen, on="ID_POLIGONO", how="left")
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in df.columns]
TODAS = S2 + FEN + GEO

train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
yv = train[Y].to_numpy(); n = len(train)
coords = train[["x_m", "y_m"]].to_numpy()
rng = np.random.default_rng(0)

split_alea = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))
split_esp = []
for rep in range(3):
    gr = KMeans(15, n_init=10, random_state=rep).fit_predict(coords)
    split_esp += list(GroupKFold(5).split(coords, yv, gr))

def modelo(alfa):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alfa))

def err_cuad(alfa, splits):
    """alfa=None -> solo el prior (corrección = 0)."""
    se, cnt = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        dtr, dte = train.iloc[tr], train.iloc[te]
        p_tr, p_te = prior_mix(dtr, dte)
        pred = p_te if alfa is None else p_te + modelo(alfa).fit(dtr[TODAS], dtr[Y].to_numpy() - p_tr).predict(dte[TODAS])
        se[te] += (yv[te] - pred) ** 2; cnt[te] += 1
    return se / cnt

ALFAS = [None, 100, 300, 1000, 3000, 10000, 100000]
BASE = 1000

p_all, _ = prior_mix(train, train.iloc[:1])
print("Tamaño de la corrección ajustada con las 138 parcelas (desv. estándar, t/ha):")
for a in ALFAS[1:]:
    c = modelo(a).fit(train[TODAS], yv - p_all).predict(train[TODAS])
    print(f"  alpha {a:>6}: {c.std():.3f}   rango [{c.min():+.3f}, {c.max():+.3f}]")

for esq, sp in {"aleatorio": split_alea, "espacial": split_esp}.items():
    print(f"\n===== CV {esq} (RMSE agrupado; dif y IC95 vs. alpha={BASE}) =====", flush=True)
    res = {a: err_cuad(a, sp) for a in ALFAS}
    base = res[BASE]; idx = rng.integers(0, n, (2000, n))
    for a, se in res.items():
        d = np.sqrt(base[idx].mean(1)) - np.sqrt(se[idx].mean(1))
        lo, hi = np.percentile(d, [2.5, 97.5])
        marca = "" if a == BASE else ("REAL" if lo > 0 else ("peor" if hi < 0 else "ruido"))
        nombre = "solo prior" if a is None else f"alpha {a}"
        print(f"  {nombre:12s} RMSE {np.sqrt(se.mean()):.3f}  dif {np.sqrt(base.mean())-np.sqrt(se.mean()):+.3f}"
              f"  IC95 [{lo:+.3f}, {hi:+.3f}]  {marca}", flush=True)