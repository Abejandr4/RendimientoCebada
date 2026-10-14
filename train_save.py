import json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, RepeatedKFold
from modelo_lib import Y, predecir, ajustar

RUTA = "DataSet/DATASET_RETO_AGRO_2026/"                                             # AJUSTA
SHP = RUTA + "Parcelas_Reto_AGC_CONJUNTO_70_30/Parcelas_Reto_AGC_CONJUNTO.shp"      # AJUSTA
OUT = Path("app_data"); OUT.mkdir(exist_ok=True)

# ---------- 1. Datos (mismo orden que modelo_final.py: S2 se define ANTES de unir fenología)
df = pd.read_parquet("df_master.parquet")
S2 = [c for c in df.columns if c.endswith("_s2") and not c.startswith("n_obs")]
fen = pd.read_csv("fenologia_features.csv")
fen["ciclo_sin_senescencia_s2"] = fen["sen50_doy_s2"].isna().astype(int)
FEN = [c for c in fen.columns if c not in ("ID_POLIGONO", "sen50_doy_s2", "dur50_dias_s2")]
df = df.merge(fen, on="ID_POLIGONO", how="left")
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in df.columns]
COLS = S2 + FEN + GEO
print(f"S2={len(S2)} FEN={len(FEN)} GEO={len(GEO)} -> {len(COLS)} features")   # esperado 18 / 11 / 5

train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
test = df[df["CONJUNTO"] == "PREDICCION"].reset_index(drop=True)
yv = train[Y].to_numpy(); n = len(train)
coords = train[["x_m", "y_m"]].to_numpy()

# ---------- 2. Validación (modelo y prior mixto de referencia, con los mismos folds)
def cv(splits):
    # Línea 30 corregida:
    se, se_b, oof, cnt, abs_err = [np.zeros(n) for _ in range(4)] + [[]]
    for tr, te in splits:
        p, base, _ = predecir(train.iloc[tr], train.iloc[te], COLS)
        se[te] += (yv[te] - p) ** 2; se_b[te] += (yv[te] - base) ** 2
        oof[te] += p; cnt[te] += 1; abs_err.append(np.abs(yv[te] - p))
    se, se_b, oof = se / cnt, se_b / cnt, oof / cnt
    r2 = 1 - ((yv - oof) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return dict(rmse=float(np.sqrt(se.mean())), rmse_prior=float(np.sqrt(se_b.mean())),
                r2=float(r2)), oof, np.concatenate(abs_err)

split_alea = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))
split_esp = []
for rep in range(3):
    gr = KMeans(15, n_init=10, random_state=rep).fit_predict(coords)
    split_esp += list(GroupKFold(5).split(coords, yv, gr))

m_alea, oof, abs_err = cv(split_alea)
m_esp, _, _ = cv(split_esp)
m_alea["mae"] = float(abs_err.mean())
q80, q90 = (float(q) for q in np.quantile(abs_err, [0.80, 0.90]))
m_media = float(yv.std())
print(f"Aleatorio: {m_alea}\nEspacial:  {m_esp}\nMedia global RMSE: {m_media:.3f}")
print(f"Intervalos: 80% = ±{q80:.2f}, 90% = ±{q90:.2f} t/ha")

# ---------- 3. Modelo final con las 138 parcelas -> predicción de las 59
pred, prior, corr = predecir(train, test, COLS)
modelo = ajustar(train, COLS)
joblib.dump({"modelo": modelo, "cols": COLS}, OUT / "modelo.joblib")

pd.DataFrame({"ID_POLIGONO": test["ID_POLIGONO"], "Municipio": test["Municipio"],
              "PREDICCION": pred, "lo80": pred - q80, "hi80": pred + q80,
              "lo90": pred - q90, "hi90": pred + q90}).to_csv("predicciones_59.csv", index=False)
pd.DataFrame({"ID_POLIGONO": train["ID_POLIGONO"], "Municipio": train["Municipio"],
              "real": yv, "oof_pred": oof}).to_csv("oof_train.csv", index=False)

# ---------- 4. Datos para el dashboard
cnt_mun = train["Municipio"].value_counts()
tab = df[["ID_POLIGONO", "CONJUNTO", "Estado", "Municipio", "lon", "lat", "x_m", "y_m", "area_ha_geom", Y] + COLS].copy()
tab["pred"] = np.nan
tab.loc[tab.CONJUNTO == "ENTRENAMIENTO", "pred"] = pd.Series(oof, index=train["ID_POLIGONO"]).reindex(
    tab.loc[tab.CONJUNTO == "ENTRENAMIENTO", "ID_POLIGONO"]).to_numpy()   # fuera de muestra
tab.loc[tab.CONJUNTO == "PREDICCION", "pred"] = pd.Series(pred, index=test["ID_POLIGONO"]).reindex(
    tab.loc[tab.CONJUNTO == "PREDICCION", "ID_POLIGONO"]).to_numpy()
tab["n_entren_municipio"] = tab["Municipio"].map(cnt_mun).fillna(0).astype(int)
tab["baja_confianza"] = tab["n_entren_municipio"] < 3
tab.to_parquet(OUT / "parcelas.parquet", index=False)

gdf = gpd.read_file(SHP)[["ID_POLIGON", "geometry"]].rename(columns={"ID_POLIGON": "ID_POLIGONO"})
gdf["geometry"] = gdf.geometry.simplify(0.00003, preserve_topology=True)      # ~3 m
gdf.to_file(OUT / "parcelas.geojson", driver="GeoJSON")

serie = pd.read_csv("serie_s2.csv", parse_dates=["fecha"])[["ID_POLIGONO", "fecha", "ndvi_promedio", "evi_promedio"]]
serie.to_parquet(OUT / "serie_s2.parquet", index=False)

meta = dict(config="res ridge S2+FEN+GEO (prior municipio + vecinos, corrección ridge)",
            cols=COLS, q80=q80, q90=q90, n_train=int(n), n_pred=int(len(test)),
            cv_aleatorio=m_alea, cv_espacial=m_esp, rmse_media_global=m_media)
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))

print("\nArchivos en app_data:", sorted(p.name for p in OUT.iterdir()))
print("Predicciones 59: media %.2f, std %.2f | min %.2f, max %.2f" % (pred.mean(), pred.std(), pred.min(), pred.max()))
print("Parcelas de baja confianza (municipio con <3 entrenamiento):\n",
      tab.loc[tab.baja_confianza, ["ID_POLIGONO", "Municipio", "CONJUNTO"]].to_string(index=False))