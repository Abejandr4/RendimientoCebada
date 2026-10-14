import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K_SHRINK = 5
rng = np.random.default_rng(0)

df = pd.read_parquet("df_master.parquet")
train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
y = train["RENDIMIENTO_T_HA"].to_numpy()
mun = train["Municipio"]
coords = train[["x_m", "y_m"]].to_numpy()
n = len(train)

S2_all = [c for c in train.columns if c.endswith("_s2")]
L8_all = [c for c in train.columns if c.endswith("_l8")]
S2 = [c for c in S2_all if not c.startswith("n_obs")]
L8 = [c for c in L8_all if not c.startswith("n_obs")]
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in train.columns]
BIO = ["GDD_antesis", "GDD_llenado", "DAP_antesis_est", "ETo_acum", "ETa_acum", "Deficit_Hidrico_Total",
       "KS_prom_antesis", "KS_prom_llenado", "fPAR_integrado", "Biomasa_Antesis_Sim",
       "Biomasa_Total_Sim", "HI_penalizado", "s_sim"]
EXT = ["Deficit_ETc", "ciclo_incompleto", "caida_fpar_no_alcanzada"]

# ---------- Validaciones y utilidades
split_alea = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))
split_esp = []
for rep in range(3):
    gr = KMeans(15, n_init=10, random_state=rep).fit_predict(coords)
    split_esp += list(GroupKFold(5).split(coords, y, gr))
esquemas = {"aleatorio": split_alea, "espacial": split_esp}

def prior_mun(tr, te):
    ytr = pd.Series(y[tr]); mtr = mun.iloc[tr].reset_index(drop=True); mte = mun.iloc[te]
    g = ytr.mean()
    suma, cnt = ytr.groupby(mtr).sum(), ytr.groupby(mtr).count()
    p_te = ((mte.map(suma).fillna(0) + K_SHRINK * g) / (mte.map(cnt).fillna(0) + K_SHRINK)).to_numpy()
    s_i, n_i = mtr.map(suma).to_numpy(), mtr.map(cnt).to_numpy()
    p_tr = (s_i - ytr.to_numpy() + K_SHRINK * g) / (n_i - 1 + K_SHRINK)   # sin la propia parcela
    return p_tr, p_te

def modelo(kind):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(200, min_samples_leaf=3, max_features=0.5,
                                               random_state=0, n_jobs=-1))

def err_parcela(cols, kind, splits, residual):
    """Error cuadrático por parcela, promediado sobre las repeticiones del CV."""
    suma, cuenta = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        p_tr, p_te = prior_mun(tr, te)
        if not cols:
            pred = p_te
        elif residual:
            pred = p_te + modelo(kind).fit(train[cols].iloc[tr], y[tr] - p_tr).predict(train[cols].iloc[te])
        else:
            pred = modelo(kind).fit(train[cols].iloc[tr], y[tr]).predict(train[cols].iloc[te])
        suma[te] += (y[te] - pred) ** 2; cuenta[te] += 1
    return suma / cuenta

rmse = lambda e: float(np.sqrt(e.mean()))

# ---------- A. ¿n_obs_* son proxies de ubicación?
print("n_obs medio por municipio:")
print(train.groupby("Municipio")[["n_obs_s2", "n_obs_l8"]].mean().round(1).to_string())

# ---------- B. Ablación (features directos, SIN prior geográfico)
variantes = {
    "n_obs_s2 solo":               ["n_obs_s2"],
    "n_obs_l8 solo":               ["n_obs_l8"],
    "L8 solo (vi6t, sin n_obs)":   L8,
    "S2 con n_obs":                S2_all,
    "S2 sin n_obs":                S2,
    "S2+GEO+L8 con n_obs":         S2_all + GEO + L8_all,
    "S2+GEO+L8 sin n_obs":         S2 + GEO + L8,
    "BIO+EXT solo":                BIO + EXT,
    "S2+GEO+L8 sin n_obs + BIO":   S2 + GEO + L8 + BIO + EXT,
}
filas = []
for nombre, cols in variantes.items():
    fila = {"variante": nombre}
    for esq, sp in esquemas.items():
        for kind in ["ridge", "rf"]:
            fila[f"{esq[:4]}_{kind}"] = rmse(err_parcela(cols, kind, sp, residual=False))
    filas.append(fila)
    print(f"  listo: {nombre}", flush=True)
print("\n=== B. Ablación (RMSE; referencia: media 0.85 aleatorio) ===")
print(pd.DataFrame(filas).round(3).to_string(index=False))

# ---------- C. ¿Las mejoras sobre el prior municipal son reales? (bootstrap pareado)
print("\n=== C. Residual sobre prior municipal vs. prior solo (diferencia de RMSE; >0 = mejora) ===")
cands = {"res + S2": S2, "res + L8": L8, "res + S2+GEO+L8": S2 + GEO + L8,
         "res + S2+GEO+L8+BIO": S2 + GEO + L8 + BIO + EXT}
for esq, sp in esquemas.items():
    base = err_parcela([], "rf", sp, residual=True)
    print(f"[{esq}] prior municipio solo: RMSE {rmse(base):.3f}")
    for nombre, cols in cands.items():
        for kind in ["ridge", "rf"]:
            e = err_parcela(cols, kind, sp, residual=True)
            idx = rng.integers(0, n, (2000, n))
            d = np.sqrt(base[idx].mean(1)) - np.sqrt(e[idx].mean(1))
            lo, hi = np.percentile(d, [2.5, 97.5])
            marca = "REAL" if lo > 0 else ("peor" if hi < 0 else "ruido")
            print(f"   {nombre:22s} {kind:5s} RMSE {rmse(e):.3f}  dif {rmse(base)-rmse(e):+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]  {marca}")

# ---------- D. Señal DENTRO del municipio (municipios con >=10 parcelas)
tam = mun.value_counts()
g = train[mun.isin(tam[tam >= 10].index)].reset_index(drop=True)
resid = g["RENDIMIENTO_T_HA"] - g.groupby("Municipio")["RENDIMIENTO_T_HA"].transform("mean")
cand = list(dict.fromkeys(S2_all + L8_all + GEO + BIO + EXT))
rho = {}
for c in cand:
    x = g[c] - g.groupby("Municipio")[c].transform("mean")
    rho[c] = x.corr(resid, method="spearman")
rho = pd.Series(rho).dropna()
print(f"\n=== D. Spearman dentro del municipio (n={len(g)}; |rho|>0.17 ~ p<0.05, ojo con comparaciones múltiples) ===")
print(rho.reindex(rho.abs().sort_values(ascending=False).index).head(15).round(2).to_string())

# ---------- E. Parcelas alargadas
raros = ["AGC_062", "AGC_073", "AGC_115", "AGC_139"]
r = train[train.ID_POLIGONO.isin(raros)][["ID_POLIGONO", "Municipio", "RENDIMIENTO_T_HA"]]
r["media_municipio"] = r["Municipio"].map(train.groupby("Municipio")["RENDIMIENTO_T_HA"].mean())
print("\n=== E. Parcelas alargadas (en entrenamiento) ===")
print(r.round(2).to_string(index=False))
print("\nDistribución de y:\n", train["RENDIMIENTO_T_HA"].describe().round(2).to_string())