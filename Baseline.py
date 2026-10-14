import warnings; warnings.filterwarnings("ignore")
import traceback
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUTA = "DataSet/DATASET_RETO_AGRO_2026/"
K_SHRINK = 5     # suavizado del prior por municipio hacia la media global

def leer_bio(path):
    with open(path, encoding="utf-8-sig") as f:
        primera = f.readline()
    sep = ";" if primera.count(";") > primera.count(",") else ","
    d = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
    d.columns = d.columns.str.strip()
    d = d.rename(columns={"ID_parcela": "ID_POLIGONO"})
    d["ID_POLIGONO"] = d["ID_POLIGONO"].str.strip()
    for c in d.columns.drop("ID_POLIGONO"):
        s = d[c].str.strip()
        low = s.str.lower()
        if low.isin(["true", "false", "verdadero", "falso", "1", "0"]).all():
            d[c] = low.isin(["true", "verdadero", "1"]).astype(int)
        else:
            d[c] = pd.to_numeric(s.str.replace(",", ".", regex=False))
    return d

# ---------- 1. Tabla maestra
parcelas = pd.read_csv(RUTA + "ID_area_rendimiento_70_30_Reto_AgroCebada.csv")
parcelas = parcelas.loc[:, ~parcelas.columns.str.startswith("Unnamed")]
feats = pd.read_csv("features_satelitales.csv")
geo = pd.read_csv("geo_features.csv")
bio = leer_bio(RUTA + "biofisicas_197_parcelas.csv")
ext = leer_bio(RUTA + "biofisicas_extras_197_parcelas.csv")

df = (parcelas.merge(feats, on="ID_POLIGONO", how="left")
              .merge(geo, on="ID_POLIGONO", how="left")
              .merge(bio, on="ID_POLIGONO", how="left")
              .merge(ext, on="ID_POLIGONO", how="left"))
assert len(df) == 197, len(df)
df["s_sim"] = df["Biomasa_Total_Sim"] * df["HI_penalizado"] / 1000   # t/ha
df.to_parquet("df_master.parquet")

train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
y = train["RENDIMIENTO_T_HA"]
coords = train[["x_m", "y_m"]].to_numpy()
print(f"Entrenamiento: {len(train)} parcelas | y media {y.mean():.2f}, std {y.std():.2f}"
      f" | corr(sim, y) = {train.s_sim.corr(y):.3f}", flush=True)
ss_tot = ((y - y.mean()) ** 2).sum()
ss_in = train.groupby("Municipio")["RENDIMIENTO_T_HA"].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
print(f"R2 EN MUESTRA de Municipio solo (optimista): {1 - ss_in / ss_tot:.2f}", flush=True)

# ---------- 2. Grupos de variables
S2 = [c for c in train.columns if c.endswith("_s2")]
L8 = [c for c in train.columns if c.endswith("_l8")]
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in train.columns]
BIO = [c for c in bio.columns if c != "ID_POLIGONO"] + ["s_sim"]
EXT = [c for c in ext.columns if c != "ID_POLIGONO"]
print(f"S2={len(S2)}  L8={len(L8)}  GEO={len(GEO)}  BIO={len(BIO)}  EXT={len(EXT)}", flush=True)
assert len(S2) > 0, "No hay columnas _s2: features_satelitales.csv no es el nuevo"

# ---------- 3. Priors geográficos (siempre calculados con datos de entrenamiento del fold)
def prior_municipio(tr, te):
    """Media por municipio suavizada. Para filas de entrenamiento se excluye la propia parcela."""
    ytr, mtr, mte = y.iloc[tr], train["Municipio"].iloc[tr], train["Municipio"].iloc[te]
    g = ytr.mean()
    suma, n = ytr.groupby(mtr).sum(), ytr.groupby(mtr).count()
    p_te = ((mte.map(suma).fillna(0) + K_SHRINK * g) / (mte.map(n).fillna(0) + K_SHRINK)).to_numpy()
    s_i, n_i = mtr.map(suma).to_numpy(), mtr.map(n).to_numpy()
    p_tr = (s_i - ytr.to_numpy() + K_SHRINK * g) / (n_i - 1 + K_SHRINK)
    return p_tr, p_te

def prior_knn(tr, te, k=5):
    """Media del rendimiento de las k parcelas de entrenamiento más cercanas."""
    ctr, cte, ytr = coords[tr], coords[te], y.iloc[tr].to_numpy()
    tree = cKDTree(ctr)
    p_te = ytr[tree.query(cte, k=k)[1]].mean(axis=1)
    p_tr = ytr[tree.query(ctr, k=k + 1)[1][:, 1:]].mean(axis=1)    # el primero es la propia parcela
    return p_tr, p_te

variantes = {
    "media":                        ([], "media"),
    "prior municipio":              ([], "prior_mun"),
    "prior vecinos (k=5)":          ([], "prior_knn"),
    "solo simulador calib.":        ([], "sim"),
    "GEO":                          (GEO, "directo"),
    "S2":                           (S2, "directo"),
    "S2+GEO":                       (S2 + GEO, "directo"),
    "S2+GEO+L8":                    (S2 + GEO + L8, "directo"),
    "S2+GEO+bio (Arq.1)":           (S2 + GEO + BIO + EXT, "directo"),
    "Arq.2 residual sim (S2+GEO)":  (S2 + GEO, "residual"),
    "res. prior mun (S2+GEO)":      (S2 + GEO, "res_mun"),
    "res. prior mun (S2+GEO+L8)":   (S2 + GEO + L8, "res_mun"),
    "res. prior mun (+bio)":        (S2 + GEO + BIO + EXT, "res_mun"),
    "res. prior vecinos (S2+GEO)":  (S2 + GEO, "res_knn"),
}

def modelo(kind):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(200, min_samples_leaf=3, max_features=0.5,
                                               random_state=0, n_jobs=-1))

def predecir(cols, modo, kind, tr, te):
    ytr = y.iloc[tr].to_numpy()
    if modo == "media":
        return np.full(len(te), ytr.mean())
    if modo == "prior_mun":
        return prior_municipio(tr, te)[1]
    if modo == "prior_knn":
        return prior_knn(tr, te)[1]
    if modo in ("sim", "residual"):          # calibración a+b*s DENTRO del fold
        sim_tr, sim_te = train[["s_sim"]].iloc[tr], train[["s_sim"]].iloc[te]
        lr = LinearRegression().fit(sim_tr, ytr)
        b_tr, b_te = lr.predict(sim_tr), lr.predict(sim_te)
        if modo == "sim":
            return b_te
        return b_te + modelo(kind).fit(train[cols].iloc[tr], ytr - b_tr).predict(train[cols].iloc[te])
    if modo in ("res_mun", "res_knn"):
        f = prior_municipio if modo == "res_mun" else prior_knn
        p_tr, p_te = f(tr, te)
        return p_te + modelo(kind).fit(train[cols].iloc[tr], ytr - p_tr).predict(train[cols].iloc[te])
    return modelo(kind).fit(train[cols].iloc[tr], ytr).predict(train[cols].iloc[te])

# ---------- 4. Esquemas de validación
split_alea = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))
split_esp = []
for rep in range(3):
    grupos = KMeans(15, n_init=10, random_state=rep).fit_predict(coords)
    split_esp += list(GroupKFold(5).split(coords, y, grupos))
split_est = list(GroupKFold(train["Estado"].nunique()).split(coords, y, train["Estado"]))
esquemas = {"aleatorio": split_alea, "espacial": split_esp, "estado (LOO)": split_est}

# ---------- 5. Evaluación (imprime cada resultado apenas termina)
filas = []
for esq, splits in esquemas.items():
    for nombre, (cols, modo) in variantes.items():
        kinds = ["-"] if modo in ("media", "sim", "prior_mun", "prior_knn") else ["ridge", "rf"]
        for kind in kinds:
            try:
                rm, r2 = [], []
                for tr, te in splits:
                    p = predecir(cols, modo, kind, tr, te)
                    rm.append(mean_squared_error(y.iloc[te], p) ** 0.5)
                    r2.append(r2_score(y.iloc[te], p))
                filas.append({"CV": esq, "variante": nombre, "modelo": kind,
                              "RMSE": np.mean(rm), "RMSE_std": np.std(rm), "R2": np.mean(r2)})
                print(f"[{esq:12s}] {nombre:30s} {kind:5s} RMSE {np.mean(rm):.3f}  R2 {np.mean(r2):+.3f}", flush=True)
            except Exception:
                print(f"[{esq}] ERROR en {nombre} / {kind}", flush=True)
                traceback.print_exc()

res = pd.DataFrame(filas)
res.to_csv("resultados_cv.csv", index=False)
for esq in esquemas:
    print(f"\n===== CV {esq} =====")
    print(res[res.CV == esq].drop(columns="CV").sort_values("RMSE").round(3).to_string(index=False))