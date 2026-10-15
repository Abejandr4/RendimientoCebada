import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, RepeatedKFold
from modelo_lib import Y, predecir

# ---------- Datos (mismo orden que entrenar_y_guardar.py)
df = pd.read_parquet("df_master.parquet")
S2 = [c for c in df.columns if c.endswith("_s2") and not c.startswith("n_obs")]
fen = pd.read_csv("fenologia_features.csv")
fen["ciclo_sin_senescencia_s2"] = fen["sen50_doy_s2"].isna().astype(int)
FEN = [c for c in fen.columns if c not in ("ID_POLIGONO", "sen50_doy_s2", "dur50_dias_s2")]
df = df.merge(fen, on="ID_POLIGONO", how="left")
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in df.columns]
BASE = S2 + FEN + GEO

BIO_SEL = ["KS_prom_llenado", "Deficit_Hidrico_Total", "Deficit_ETc", "s_sim"]   # definidas de antemano
BIO_ALL = ["GDD_antesis", "GDD_llenado", "DAP_antesis_est", "ETo_acum", "ETa_acum", "Deficit_Hidrico_Total",
           "KS_prom_antesis", "KS_prom_llenado", "fPAR_integrado", "Biomasa_Antesis_Sim",
           "Biomasa_Total_Sim", "HI_penalizado", "s_sim",
           "Deficit_ETc", "ciclo_incompleto", "caida_fpar_no_alcanzada"]

CFG = {
    "Base (S2+FEN+GEO)":             BASE,
    "Base + BIO seleccionadas":      BASE + BIO_SEL,
    "Base + BIO completas":          BASE + BIO_ALL,
    "GEO + BIO seleccionadas (sin S2)": GEO + BIO_SEL,
}

train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
yv = train[Y].to_numpy(); n = len(train)
coords = train[["x_m", "y_m"]].to_numpy()
rng = np.random.default_rng(0)

split_alea = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))
split_esp = []
for rep in range(3):
    gr = KMeans(15, n_init=10, random_state=rep).fit_predict(coords)
    split_esp += list(GroupKFold(5).split(coords, yv, gr))

def err_cuad(cols, splits):
    se, cnt = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        p, _, _ = predecir(train.iloc[tr], train.iloc[te], cols)
        se[te] += (yv[te] - p) ** 2; cnt[te] += 1
    return se / cnt

for esq, sp in {"aleatorio": split_alea, "espacial": split_esp}.items():
    print(f"\n===== CV {esq} (RMSE agrupado; dif y IC95 vs. Base) =====", flush=True)
    res = {k: err_cuad(c, sp) for k, c in CFG.items()}
    base = res["Base (S2+FEN+GEO)"]; idx = rng.integers(0, n, (2000, n))
    for k, se in res.items():
        d = np.sqrt(base[idx].mean(1)) - np.sqrt(se[idx].mean(1))
        lo, hi = np.percentile(d, [2.5, 97.5])
        marca = "" if k.startswith("Base (") else ("REAL" if lo > 0 else ("peor" if hi < 0 else "ruido"))
        print(f"  {k:34s} RMSE {np.sqrt(se.mean()):.3f}  dif {np.sqrt(base.mean())-np.sqrt(se.mean()):+.3f}"
              f"  IC95 [{lo:+.3f}, {hi:+.3f}]  {marca}", flush=True)

print("\nCorrelación de las BIO seleccionadas con el rendimiento (entrenamiento):")
print(train[BIO_SEL].corrwith(train[Y]).round(2).to_string())