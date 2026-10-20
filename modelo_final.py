import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONFIG_FINAL = "res ridge S2+FEN+GEO"  # cambia tras ver la tabla 
K_SHRINK, K_VEC = 5, 5
Y = "RENDIMIENTO_T_HA"
rng = np.random.default_rng(0)

# ---------- 1. Datos
df = pd.read_parquet("df_master.parquet")
S2 = [c for c in df.columns if c.endswith("_s2") and not c.startswith("n_obs")]
fen = pd.read_csv("fenologia_features.csv")
fen["ciclo_sin_senescencia_s2"] = fen["sen50_doy_s2"].isna().astype(int)
FEN = [c for c in fen.columns if c not in ("ID_POLIGONO", "sen50_doy_s2", "dur50_dias_s2")]
df = df.merge(fen, on="ID_POLIGONO", how="left")
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in df.columns]
BIO = ["GDD_antesis", "GDD_llenado", "DAP_antesis_est", "ETo_acum", "ETa_acum", "Deficit_Hidrico_Total",
       "KS_prom_antesis", "KS_prom_llenado", "fPAR_integrado", "Biomasa_Antesis_Sim",
       "Biomasa_Total_Sim", "HI_penalizado", "s_sim"]
EXT = ["Deficit_ETc", "ciclo_incompleto", "caida_fpar_no_alcanzada"]
print(f"S2={len(S2)}  FEN={len(FEN)}  GEO={len(GEO)}  BIO={len(BIO)}  EXT={len(EXT)}")

train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
test = df[df["CONJUNTO"] == "PREDICCION"].reset_index(drop=True)
yv = train[Y].to_numpy(); n = len(train)
print(f"Entrenamiento {n} | Predicción {len(test)}")

# Revisión: municipios de las 59 parcelas vs. datos de entrenamiento disponibles
chk = pd.DataFrame({"n_pred": test["Municipio"].value_counts()})
chk["n_entren"] = chk.index.map(train["Municipio"].value_counts()).fillna(0).astype(int)
print("\nMunicipios a predecir:\n", chk.to_string())

# ---------- 2. Priors geográficos (siempre con datos de entrenamiento del fold)
def prior_mun(dtr, dte):
    ytr, mtr, mte = dtr[Y].to_numpy(), dtr["Municipio"].to_numpy(), dte["Municipio"].to_numpy()
    g = ytr.mean()
    suma, cnt = pd.Series(ytr).groupby(mtr).sum(), pd.Series(ytr).groupby(mtr).count()
    p_te = (pd.Series(mte).map(suma).fillna(0).to_numpy() + K_SHRINK * g) / \
           (pd.Series(mte).map(cnt).fillna(0).to_numpy() + K_SHRINK)
    s_i, n_i = pd.Series(mtr).map(suma).to_numpy(), pd.Series(mtr).map(cnt).to_numpy()
    p_tr = (s_i - ytr + K_SHRINK * g) / (n_i - 1 + K_SHRINK)      # excluye la propia parcela
    return p_tr, p_te

def prior_vec(dtr, dte):
    ctr, cte = dtr[["x_m", "y_m"]].to_numpy(), dte[["x_m", "y_m"]].to_numpy()
    ytr = dtr[Y].to_numpy(); tree = cKDTree(ctr)
    p_te = ytr[tree.query(cte, k=K_VEC)[1]].mean(axis=1)
    p_tr = ytr[tree.query(ctr, k=K_VEC + 1)[1][:, 1:]].mean(axis=1)   # el primero es la propia parcela
    return p_tr, p_te

def prior_mix(dtr, dte):
    a, b = prior_mun(dtr, dte), prior_vec(dtr, dte)
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

# ---------- 3. Configuraciones (fijadas de antemano)
def modelo(kind):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-1, 3, 25)))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(200, min_samples_leaf=3, max_features=0.5,
                                               random_state=0, n_jobs=1))

CFG = {
    "prior municipio":          dict(prior=prior_mun, cols=[], models=[]),
    "prior mix (mun+vecinos)":  dict(prior=prior_mix, cols=[], models=[]),
    "res ridge S2":             dict(prior=prior_mix, cols=S2, models=["ridge"]),
    "res ridge S2+FEN":         dict(prior=prior_mix, cols=S2 + FEN, models=["ridge"]),
    "res ridge S2+FEN+GEO":     dict(prior=prior_mix, cols=S2 + FEN + GEO, models=["ridge"]),
    "res ensemble S2+FEN+GEO":  dict(prior=prior_mix, cols=S2 + FEN + GEO, models=["ridge", "rf"]),
    "Arq.1 res ens. +BIO":      dict(prior=prior_mix, cols=S2 + FEN + GEO + BIO + EXT, models=["ridge", "rf"]),
}
BASE = "prior mix (mun+vecinos)"

def predecir(cfg, dtr, dte):
    p_tr, p_te = cfg["prior"](dtr, dte)
    if not cfg["cols"]:
        return p_te
    r = dtr[Y].to_numpy() - p_tr
    c = cfg["cols"]
    corr = [modelo(k).fit(dtr[c], r).predict(dte[c]) for k in cfg["models"]]
    return p_te + np.mean(corr, axis=0)

# ---------- 4. Validación
coords = train[["x_m", "y_m"]].to_numpy()
split_alea = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))
split_esp = []
for rep in range(3):
    gr = KMeans(15, n_init=10, random_state=rep).fit_predict(coords)
    split_esp += list(GroupKFold(5).split(coords, yv, gr))
esquemas = {"aleatorio": split_alea, "espacial": split_esp}

def cv(cfg, splits):
    se, pr, cnt, abs_all = np.zeros(n), np.zeros(n), np.zeros(n), []
    for tr, te in splits:
        p = predecir(cfg, train.iloc[tr], train.iloc[te])
        e = yv[te] - p
        se[te] += e ** 2; pr[te] += p; cnt[te] += 1; abs_all.append(np.abs(e))
    return se / cnt, pr / cnt, np.concatenate(abs_all)

resultados = {}
for esq, sp in esquemas.items():
    print(f"\n===== CV {esq} (RMSE agrupado; dif y IC95 vs. '{BASE}') =====", flush=True)
    for nombre, cfg in CFG.items():
        resultados[(esq, nombre)] = cv(cfg, sp)
    base_se = resultados[(esq, BASE)][0]
    idx = rng.integers(0, n, (2000, n))
    for nombre in CFG:
        se = resultados[(esq, nombre)][0]
        d = np.sqrt(base_se[idx].mean(1)) - np.sqrt(se[idx].mean(1))
        lo, hi = np.percentile(d, [2.5, 97.5])
        marca = "" if nombre == BASE else ("REAL" if lo > 0 else ("peor" if hi < 0 else "ruido"))
        print(f"  {nombre:26s} RMSE {np.sqrt(se.mean()):.3f}  dif {np.sqrt(base_se.mean())-np.sqrt(se.mean()):+.3f}"
              f"  IC95 [{lo:+.3f}, {hi:+.3f}]  {marca}", flush=True)

# ---------- 5. Modelo final con las 138 parcelas
cfg = CFG[CONFIG_FINAL]
pred = predecir(cfg, train, test)
_, oof, abs_err = resultados[("aleatorio", CONFIG_FINAL)]
q80, q90 = np.quantile(abs_err, [0.80, 0.90])
print(f"\nModelo final: {CONFIG_FINAL} | semiancho de intervalo: 80% = ±{q80:.2f}, 90% = ±{q90:.2f} t/ha")

out = pd.DataFrame({"ID_POLIGONO": test["ID_POLIGONO"], "Municipio": test["Municipio"],
                    "PREDICCION": pred, "lo80": pred - q80, "hi80": pred + q80,
                    "lo90": pred - q90, "hi90": pred + q90})
out.to_csv("predicciones_59.csv", index=False)
pd.DataFrame({"ID_POLIGONO": train["ID_POLIGONO"], "Municipio": train["Municipio"],
              "real": yv, "oof_pred": oof}).to_csv("oof_train.csv", index=False)
print(out["PREDICCION"].describe().round(2).to_string())
print("Entrenamiento y: media %.2f, std %.2f | Predicción: media %.2f, std %.2f"
      % (yv.mean(), yv.std(), pred.mean(), pred.std()))
print("Guardados: predicciones_59.csv, oof_train.csv")